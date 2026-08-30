"""Audit a sample-size hierarchical reliability correction over v64.

The method is an independent implementation of a public high-score concept:
use only row-local official history counts to decide how much a residual
correction can be trusted.  Every lookup is learned strictly before the
validation block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v65_hierarchical_context_lookup import cluster_bootstrap, gain


ROOT = Path(__file__).resolve().parent
COUNT_COLUMNS = (
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n",
)
BIN_EDGES = np.asarray([-np.inf, 0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, np.inf])
HIERARCHY = [
    ["game_type", "pitcher_n_bin", "batter_n_bin"],
    ["game_type", "pitcher_n_bin", "pitchmix_n_bin"],
    ["game_type", "pitcher_n_bin"],
    ["game_type", "batter_n_bin"],
    ["game_type", "pitchmix_n_bin"],
    ["pitcher_n_bin", "batter_n_bin"],
    ["game_type"],
    [],
]
SMOOTHING_GRID = (100.0, 300.0, 500.0, 1000.0)
MIN_COUNT_GRID = (50, 100, 250)
SCALE_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0)
CLIP = (0.005, 0.995)


def add_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for source, target in zip(
        COUNT_COLUMNS, ("pitcher_n_bin", "batter_n_bin", "pitchmix_n_bin"),
    ):
        values = pd.to_numeric(out[source], errors="coerce").fillna(0.0).clip(lower=0.0)
        out[target] = pd.cut(values, BIN_EDGES, labels=False, include_lowest=True).astype(str)
    out["game_type"] = out["game_type"].fillna("missing").astype(str)
    return out


def build(source: pd.DataFrame, smoothing: float, min_count: int) -> dict[str, object]:
    prior = float(source["residual"].mean())
    tables = []
    for keys in HIERARCHY:
        if not keys:
            table = pd.DataFrame({"correction": [prior], "count": [len(source)]})
        else:
            table = source.groupby(keys, observed=True, sort=False)["residual"].agg(
                ["sum", "count"],
            ).reset_index()
            table = table.loc[table["count"].ge(min_count)].copy()
            table["correction"] = (
                table["sum"] + smoothing * prior
            ) / (table["count"] + smoothing)
            table = table[keys + ["correction", "count"]]
        tables.append((keys, table))
    return {"prior": prior, "tables": tables}


def apply(frame: pd.DataFrame, artifact: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    correction = np.full(len(frame), np.nan, dtype=float)
    level = np.full(len(frame), -1, dtype=np.int8)
    for position, (keys, table) in enumerate(artifact["tables"]):
        if keys:
            merged = frame[keys].merge(
                table, on=keys, how="left", validate="many_to_one", sort=False,
            )
            values = merged["correction"].to_numpy(float)
        else:
            values = np.full(len(frame), float(table.iloc[0]["correction"]))
        fill = np.isnan(correction) & np.isfinite(values)
        correction[fill] = values[fill]
        level[fill] = position
    if np.isnan(correction).any():
        raise RuntimeError("unmatched reliability rows")
    return correction, level


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=3000)
    args = parser.parse_args()
    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        y = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    raw = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=["season", "game_type", "pitcher_id", "control_success", *COUNT_COLUMNS],
        encoding="utf-8-sig", low_memory=False,
    )
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(y) or not np.array_equal(rows["control_success"].to_numpy(float), y):
        raise ValueError("v64 OOF rows are not aligned")
    rows = add_bins(rows)
    rows["target"] = y
    rows["baseline"] = base
    rows["residual"] = y - base
    positions23 = np.flatnonzero(season == 2023)
    split23 = len(positions23) // 2
    folds = [
        {
            "name": "2023_second_half",
            "source": rows.iloc[positions23[:split23]].copy(),
            "valid": rows.iloc[positions23[split23:]].copy(),
        },
        {
            "name": "2024_forward",
            "source": rows.loc[rows["season"].eq(2023)].copy(),
            "valid": rows.loc[rows["season"].eq(2024)].copy(),
        },
    ]
    prepared: dict[tuple[float, int], list[dict[str, object]]] = {}
    for smoothing in SMOOTHING_GRID:
        for min_count in MIN_COUNT_GRID:
            key = (smoothing, min_count)
            prepared[key] = []
            for fold in folds:
                artifact = build(fold["source"], smoothing, min_count)
                correction, level = apply(fold["valid"], artifact)
                prepared[key].append({**fold, "correction": correction, "level": level})

    candidates = []
    for (smoothing, min_count), blocks in prepared.items():
        for r_scale in SCALE_GRID:
            for f_scale in SCALE_GRID:
                evaluations = []
                for block in blocks:
                    valid = block["valid"]
                    target = valid["target"].to_numpy(float)
                    baseline = valid["baseline"].to_numpy(float)
                    regular = valid["game_type"].eq("R").to_numpy()
                    scale = np.where(regular, r_scale, f_scale)
                    prediction = np.clip(baseline + scale * block["correction"], *CLIP)
                    halves = np.array_split(np.arange(len(valid)), 2)
                    groups = {
                        label: gain(target[mask], baseline[mask], prediction[mask])
                        for label, mask in (("R", regular), ("F", ~regular))
                    }
                    evaluations.append({
                        "fold": block["name"],
                        "gain": gain(target, baseline, prediction),
                        "half_gains": [
                            gain(target[index], baseline[index], prediction[index])
                            for index in halves
                        ],
                        "group_gains": groups,
                        "mean_absolute_change": float(np.mean(np.abs(prediction - baseline))),
                    })
                preliminary = bool(
                    min(item["gain"] for item in evaluations) > 0.0
                    and min(v for item in evaluations for v in item["half_gains"]) >= 0.0
                    and min(v for item in evaluations for v in item["group_gains"].values()) >= 0.0
                )
                gains = [item["gain"] for item in evaluations]
                candidates.append({
                    "smoothing": smoothing, "min_count": min_count,
                    "r_scale": r_scale, "f_scale": f_scale,
                    "evaluations": evaluations,
                    "preliminary_gate": preliminary,
                    "min_gain": float(min(gains)),
                    "mean_gain": float(np.mean(gains)),
                })
    candidates.sort(
        key=lambda item: (item["preliminary_gate"], item["min_gain"], item["mean_gain"]),
        reverse=True,
    )
    best = candidates[0]
    blocks = prepared[(float(best["smoothing"]), int(best["min_count"]))]
    bootstraps = {}
    level_usage = {}
    for block in blocks:
        valid = block["valid"]
        target = valid["target"].to_numpy(float)
        baseline = valid["baseline"].to_numpy(float)
        regular = valid["game_type"].eq("R").to_numpy()
        scale = np.where(regular, best["r_scale"], best["f_scale"])
        prediction = np.clip(baseline + scale * block["correction"], *CLIP)
        bootstraps[block["name"]] = cluster_bootstrap(
            target, baseline, prediction, valid["pitcher_id"].to_numpy(),
            args.bootstrap, 651146 + int(valid["season"].iloc[-1]),
        )
        unique, counts = np.unique(block["level"], return_counts=True)
        level_usage[block["name"]] = {
            str(int(level)): int(count) for level, count in zip(unique, counts)
        }
    strict = bool(
        best["preliminary_gate"]
        and min(item["ci_low"] for item in bootstraps.values()) > 0.0
    )
    report = {
        "baseline": "v64_public_method_transfer",
        "method": "independent sample-size hierarchical reliability correction",
        "best": best,
        "bootstrap": bootstraps,
        "level_usage": level_usage,
        "strict_gate": strict,
        "selected": best if strict else None,
        "top_candidates": candidates[:20],
        "rules": {
            "official_data_only": True,
            "external_model_or_prediction_used": False,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    (ROOT / "research").mkdir(exist_ok=True)
    path = ROOT / "research/v65_sample_reliability.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
