"""Audit sharpness around frozen prior-season context anchors.

Unlike batch recentering, every anchor is fitted from the previous labelled
season and is therefore a fixed lookup at validation/inference time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import TARGET_COL
from research_inferred_pitch_priors import bss


CONTEXTS = {
    "global": (),
    "regime": ("game_type",),
    "regime_count": ("game_type", "count_state"),
    "regime_count_hands": (
        "game_type", "count_state", "pitcher_hand", "batter_hand",
    ),
}


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def frozen_anchor(context, target, source, valid, keys):
    if not keys:
        return np.full(valid.sum(), float(target[source].mean()))
    work = context.loc[source, list(keys)].copy()
    work["target"] = target[source]
    table = work.groupby(list(keys), observed=True)["target"].agg(
        target_sum="sum", target_n="count",
    ).reset_index()
    global_rate = float(target[source].mean())
    regime = work.assign(target=target[source]).groupby(
        "game_type", observed=True,
    )["target"].mean().to_dict()
    query = context.loc[valid, list(keys)].copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(table, on=list(keys), how="left", sort=False).sort_values("_order")
    fallback = query["game_type"].map(regime).fillna(global_rate).to_numpy(float)
    count = query["target_n"].fillna(0.0).to_numpy(float)
    total = query["target_sum"].fillna(0.0).to_numpy(float)
    # Context rates are deliberately heavily shrunk to the prior regime level.
    return (total + 500.0 * fallback) / (count + 500.0)


def segment_masks(rows):
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": np.arange(len(rows)) < len(rows) // 2,
        "second_half": np.arange(len(rows)) >= len(rows) // 2,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def gain_curve(target, base, direction, mask):
    reference = float(target[mask].mean() * (1.0 - target[mask].mean()))
    residual = target[mask] - base[mask]
    selected = direction[mask]
    return (
        200000.0 * float(np.mean(residual * selected)) / reference,
        100000.0 * float(np.mean(selected * selected)) / reference,
    )


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target = raw[TARGET_COL].to_numpy(float)
    context = pd.DataFrame({
        "game_type": raw["game_type"].astype(str),
        "count_state": raw["balls_before"] * 3 + raw["strikes_before"],
        "pitcher_hand": raw["pitcher_hand"],
        "batter_hand": raw["batter_hand"],
    })
    seasons = raw["season"].to_numpy(np.int16)
    with np.load(root / "outputs/v23_oof_predictions.npz") as z:
        oof = {key: z[key] for key in z.files}
    years = {}
    for year in (2023, 2024):
        fold = oof["season"] == year
        valid = seasons == year
        source = seasons == year - 1
        if not np.allclose(oof["target"][fold], target[valid]):
            raise ValueError(f"v23 OOF rows do not align for {year}")
        rows = raw.loc[valid].reset_index(drop=True)
        years[year] = {
            "target": target[valid], "base": oof["blended"][fold].astype(float),
            "masks": segment_masks(rows),
            "anchors": {
                name: frozen_anchor(context, target, source, valid, keys)
                for name, keys in CONTEXTS.items()
            },
        }

    approximate = []
    for anchor_name in CONTEXTS:
        for transform in ("linear", "logit"):
            for gate_name in ("all", "regular", "futures"):
                curves = {}
                for year, item in years.items():
                    anchor = item["anchors"][anchor_name]
                    if transform == "linear":
                        direction = item["base"] - anchor
                    else:
                        direction = (
                            item["base"] * (1.0 - item["base"])
                            * (logit(item["base"]) - logit(anchor))
                        )
                    direction = direction * item["masks"][gate_name]
                    curves[str(year)] = {
                        name: gain_curve(
                            item["target"], item["base"], direction, mask,
                        )
                        for name, mask in item["masks"].items() if mask.any()
                    }
                for alpha in np.arange(.90, 1.501, .01):
                    scale = float(alpha - 1.0)
                    gains = {}
                    temporal = []
                    for year in (2023, 2024):
                        gains[str(year)] = {
                            name: scale * linear - scale * scale * quadratic
                            for name, (linear, quadratic) in curves[str(year)].items()
                        }
                        temporal.extend(
                            gains[str(year)][name]
                            for name in (
                                "first_half", "second_half", "months_3_5", "months_6_7",
                                "months_8_11",
                            )
                        )
                    approximate.append({
                        "anchor": anchor_name, "transform": transform,
                        "gate": gate_name, "alpha": float(alpha), "gains": gains,
                        "min_temporal": min(temporal),
                        "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
                        "mean_year": np.mean([gains["2023"]["all"], gains["2024"]["all"]]),
                    })
    approximate.sort(
        key=lambda row: (row["min_temporal"], row["min_year"], row["mean_year"]),
        reverse=True,
    )
    reports = []
    for report in approximate[:160]:
        gains = {}
        temporal = []
        for year, item in years.items():
            anchor = item["anchors"][report["anchor"]]
            base = item["base"]
            if report["transform"] == "linear":
                changed = anchor + report["alpha"] * (base - anchor)
            else:
                changed = sigmoid(
                    logit(anchor)
                    + report["alpha"] * (logit(base) - logit(anchor))
                )
            active = item["masks"][report["gate"]]
            candidate = np.clip(np.where(active, changed, base), .005, .995)
            gains[str(year)] = {
                name: bss(item["target"][mask], candidate[mask]) - bss(
                    item["target"][mask], base[mask],
                )
                for name, mask in item["masks"].items() if mask.any()
            }
            temporal.extend(
                gains[str(year)][name]
                for name in (
                    "first_half", "second_half", "months_3_5", "months_6_7",
                    "months_8_11",
                )
            )
        reports.append({
            **{key: report[key] for key in ("anchor", "transform", "gate", "alpha")},
            "gains": gains, "min_temporal": min(temporal),
            "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
            "mean_year": np.mean([gains["2023"]["all"], gains["2024"]["all"]]),
        })
    reports.sort(
        key=lambda row: (row["min_temporal"], row["min_year"], row["mean_year"]),
        reverse=True,
    )
    output = root / "research/v23_frozen_sharpness.json"
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps({"top": reports[:100]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
