"""Forward-screen frozen-anchor sharpness separately for R and F over v24."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
CONTEXTS = {
    "global": (),
    "count": ("count_state",),
    "count_hands": ("count_state", "pitcher_hand", "batter_hand"),
    "count_runners": ("count_state", "runner_gate"),
    "count_hands_runners": (
        "count_state", "pitcher_hand", "batter_hand", "runner_gate",
    ),
}
SHRINKS = (100., 500., 2000., 6400.)
ALPHAS = np.round(np.arange(.90, 1.201, .01), 2)


def logit(probability):
    p = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(p / (1. - p))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def add_context(raw):
    out = raw.copy()
    out["count_state"] = out["balls_before"] * 3 + out["strikes_before"]
    out["runner_gate"] = out["num_runners_on"].gt(0).astype(np.int8)
    return out


def anchor(source, query, target, keys, shrink):
    global_rate = float(np.mean(target))
    if not keys:
        return np.full(len(query), global_rate)
    work = source[list(keys)].copy()
    work["_target"] = target
    table = work.groupby(list(keys), observed=True, sort=False)["_target"].agg(
        ["sum", "count"]
    ).reset_index()
    table["_anchor"] = (table["sum"] + shrink * global_rate) / (table["count"] + shrink)
    left = query[list(keys)].copy()
    left["_order"] = np.arange(len(left))
    merged = left.merge(
        table[[*keys, "_anchor"]], on=list(keys), how="left", sort=False,
    ).sort_values("_order")
    return merged["_anchor"].fillna(global_rate).to_numpy(float)


def changed(base, center, alpha, transform):
    if transform == "linear":
        return np.clip(center + alpha * (base - center), .005, .995)
    return sigmoid(logit(center) + alpha * (logit(base) - logit(center)))


def gain(y, base, candidate):
    return bss(y, candidate) - bss(y, base)


def masks(rows):
    position = np.arange(len(rows))
    result = {
        "all": np.ones(len(rows), dtype=bool),
        "half_1": position < len(rows) // 2,
        "half_2": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
    }
    for month in sorted(rows["game_month"].unique()):
        active = rows["game_month"].eq(month).to_numpy()
        if active.sum() >= 40:
            result[f"month_{int(month)}"] = active
    return result


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    data = add_context(raw)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    positions = np.concatenate([
        np.flatnonzero(data["season"].to_numpy() == year) for year in (2023, 2024)
    ])
    rows = data.iloc[positions].reset_index(drop=True)
    y = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    year = oof["season"].astype(int)
    if not np.allclose(y, raw["control_success"].to_numpy(float)[positions]):
        raise ValueError("v24 OOF rows do not align")

    reports = []
    for regime in ("R", "F"):
        indices = {
            value: np.flatnonzero(rows["game_type"].eq(regime).to_numpy() & (year == value))
            for value in (2023, 2024)
        }
        halves = {
            (value, half): index[:len(index)//2] if half == 1 else index[len(index)//2:]
            for value, index in indices.items() for half in (1, 2)
        }
        transfers = (
            ("23h1_to_23h2", halves[(2023, 1)], halves[(2023, 2)]),
            ("23_to_24h1", indices[2023], halves[(2024, 1)]),
            ("23_to_24h2", indices[2023], halves[(2024, 2)]),
            ("24h1_to_24h2", halves[(2024, 1)], halves[(2024, 2)]),
        )
        for context_name, keys in CONTEXTS.items():
            for shrink, transform in itertools.product(SHRINKS, ("linear", "logit")):
                centers = {
                    label: anchor(
                        rows.iloc[source], rows.iloc[valid], y[source], keys, shrink,
                    )
                    for label, source, valid in transfers
                }
                for alpha in ALPHAS:
                    if alpha == 1.:
                        continue
                    values = {
                        label: gain(
                            y[valid], base[valid],
                            changed(base[valid], centers[label], alpha, transform),
                        )
                        for label, _source, valid in transfers
                    }
                    if min(values.values()) > 0:
                        reports.append({
                            "regime": regime, "context": context_name,
                            "keys": list(keys), "shrink": shrink,
                            "transform": transform, "alpha": float(alpha),
                            "gains": values, "min_transfer": min(values.values()),
                            "mean_transfer": float(np.mean(list(values.values()))),
                        })

    reports.sort(key=lambda item: (item["min_transfer"], item["mean_transfer"]), reverse=True)
    strongest = {}
    for report in reports:
        strongest.setdefault(report["regime"], report)
    audits = []
    for report in strongest.values():
        regime = report["regime"]
        source = np.flatnonzero(rows["game_type"].eq(regime).to_numpy() & (year == 2023))
        valid = np.flatnonzero(rows["game_type"].eq(regime).to_numpy() & (year == 2024))
        center = anchor(
            rows.iloc[source], rows.iloc[valid], y[source], tuple(report["keys"]),
            float(report["shrink"]),
        )
        candidate = changed(
            base[valid], center, float(report["alpha"]), report["transform"],
        )
        detail = {
            name: gain(y[valid][active], base[valid][active], candidate[active])
            for name, active in masks(rows.iloc[valid].reset_index(drop=True)).items()
        }
        audits.append({**report, "detail_2024": detail, "min_2024_segment": min(detail.values())})
    result = {"positive_count": len(reports), "top": reports[:200], "audits": audits}
    output = ROOT / "research/v25_regime_sharpness.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"positive_count": len(reports), "top": reports[:30], "audits": audits}, indent=2))
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
