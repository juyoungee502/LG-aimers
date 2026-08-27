"""Audit prior-season anchored calibration over the v38 OOF prediction.

Every anchor is learned from 2023 labels and applied unchanged to 2024 rows.
The screen never recenters from a validation batch, so surviving policies are
compatible with row-independent inference after refitting anchors on 2024.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CONTEXTS = {
    "global": (),
    "regime": ("game_type",),
    "regime_count": ("game_type", "count_state"),
    "regime_count_hands": (
        "game_type", "count_state", "pitcher_hand", "batter_hand",
    ),
}


def bss(target, prediction):
    target = np.asarray(target, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), .005, .995)
    rate = float(target.mean())
    return float(100000. * (
        1. - np.mean((target - prediction) ** 2) / (rate * (1. - rate))
    ))


def logit(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def frozen_anchor(source, valid, keys, shrink=500.):
    global_rate = float(source["control_success"].mean())
    if not keys:
        return np.full(len(valid), global_rate, dtype=float)
    regimes = source.groupby("game_type", observed=True)[
        "control_success"
    ].mean().to_dict()
    table = source.groupby(list(keys), observed=True)["control_success"].agg(
        target_sum="sum", target_n="count",
    ).reset_index()
    query = valid[list(keys)].copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(table, on=list(keys), how="left", sort=False).sort_values(
        "_order",
    )
    prior = valid["game_type"].map(regimes).fillna(global_rate).to_numpy(float)
    count = query["target_n"].fillna(0.).to_numpy(float)
    total = query["target_sum"].fillna(0.).to_numpy(float)
    return (total + shrink * prior) / (count + shrink)


def masks(rows):
    position = np.arange(len(rows))
    output = {
        "all": np.ones(len(rows), dtype=bool),
        "h1": position < len(rows) // 2,
        "h2": position >= len(rows) // 2,
        "R": rows["game_type"].eq("R").to_numpy(),
        "F": rows["game_type"].eq("F").to_numpy(),
    }
    for index, part in enumerate(np.array_split(position, 4), 1):
        active = np.zeros(len(rows), dtype=bool)
        active[part] = True
        output[f"q{index}"] = active
    for month in sorted(rows["game_month"].unique()):
        output[f"m{int(month)}"] = rows["game_month"].eq(month).to_numpy()
    return output


def apply_policy(base, anchor, rows, transform, alpha_r, alpha_f):
    alpha = np.where(rows["game_type"].eq("R").to_numpy(), alpha_r, alpha_f)
    if transform == "linear":
        return np.clip(anchor + alpha * (base - anchor), .005, .995)
    return np.clip(sigmoid(
        logit(anchor) + alpha * (logit(base) - logit(anchor))
    ), .005, .995)


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
        usecols=[
            "season", "game_month", "game_type", "balls_before",
            "strikes_before", "pitcher_hand", "batter_hand", "control_success",
        ],
    )
    raw["count_state"] = raw["balls_before"] * 3 + raw["strikes_before"]
    source = raw.loc[raw["season"].eq(2023)].reset_index(drop=True)
    valid = raw.loc[raw["season"].eq(2024)].reset_index(drop=True)
    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    active = oof["season"] == 2024
    target = oof["target"][active].astype(float)
    base = np.clip(oof["blended"][active].astype(float), .005, .995)
    if not np.allclose(target, valid["control_success"]):
        raise ValueError("v38 OOF and train rows do not align")
    segments = masks(valid)
    base_scores = {
        name: bss(target[mask], base[mask])
        for name, mask in segments.items() if mask.any()
    }
    base_squared = (target - base) ** 2
    segment_stats = {
        name: {
            "n": int(mask.sum()),
            "reference": float(target[mask].mean() * (1. - target[mask].mean())),
            "base_sse": float(base_squared[mask].sum()),
        }
        for name, mask in segments.items() if mask.any()
    }
    regular = valid["game_type"].eq("R").to_numpy()

    reports = []
    # A coarse grid is enough for screening; refine only a policy that passes
    # every temporal and regime gate.
    alpha_grid = np.round(np.arange(.94, 1.181, .02), 4)
    for context_name, keys in CONTEXTS.items():
        anchor = frozen_anchor(source, valid, keys)
        for transform in ("linear", "logit"):
            losses = {name: {"R": [], "F": []} for name in segment_stats}
            for alpha in alpha_grid:
                if transform == "linear":
                    changed = np.clip(anchor + alpha * (base - anchor), .005, .995)
                else:
                    changed = np.clip(sigmoid(
                        logit(anchor) + alpha * (logit(base) - logit(anchor))
                    ), .005, .995)
                squared = (target - changed) ** 2
                for name, mask in segments.items():
                    if name not in segment_stats:
                        continue
                    losses[name]["R"].append(float(squared[mask & regular].sum()))
                    losses[name]["F"].append(float(squared[mask & ~regular].sum()))
            for alpha_r in alpha_grid:
                for alpha_f in alpha_grid:
                    r_index = int(np.flatnonzero(alpha_grid == alpha_r)[0])
                    f_index = int(np.flatnonzero(alpha_grid == alpha_f)[0])
                    gains = {}
                    for name, stats in segment_stats.items():
                        candidate_sse = (
                            losses[name]["R"][r_index]
                            + losses[name]["F"][f_index]
                        )
                        gains[name] = float(
                            100000. * (stats["base_sse"] - candidate_sse)
                            / stats["n"] / stats["reference"]
                        )
                    quarters = [gains[f"q{i}"] for i in range(1, 5)]
                    halves = [gains["h1"], gains["h2"]]
                    months = [
                        value for name, value in gains.items() if name.startswith("m")
                    ]
                    reports.append({
                        "context": context_name, "transform": transform,
                        "alpha_R": float(alpha_r), "alpha_F": float(alpha_f),
                        "gains": gains,
                        "min_quarter": float(min(quarters)),
                        "min_half": float(min(halves)),
                        "min_month": float(min(months)),
                    })
    robust = sorted(
        reports,
        key=lambda row: (
            min(row["min_quarter"], row["min_half"], row["gains"]["R"],
                row["gains"]["F"]),
            row["gains"]["all"], row["min_month"],
        ),
        reverse=True,
    )
    overall = sorted(reports, key=lambda row: row["gains"]["all"], reverse=True)
    output = ROOT / "research/v39_v38_calibration.json"
    output.write_text(json.dumps({
        "base_bss": base_scores["all"],
        "selection": (
            "Prior-season anchors only; no validation-batch recentering. "
            "Robust ranking maximizes the worst quarter, half, R, and F gain."
        ),
        "best_robust": robust[:100], "best_overall": overall[:100],
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "base_bss": base_scores["all"],
        "best_robust": robust[:10], "best_overall": overall[:10],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
