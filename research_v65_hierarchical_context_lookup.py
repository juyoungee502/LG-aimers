"""Strictly audit a train-derived hierarchical residual lookup over v64.

The structure is independently implemented from a publicly described idea.
Only official training rows and v64's own forward OOF residuals are used.  The
2023 second half is predicted from the first half, and 2024 is predicted from
2023, so no validation target can enter its own lookup table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
KEYS = ["pitcher_id", "game_type", "count_state", "batter_hand"]
HIERARCHY = [
    KEYS,
    ["pitcher_id", "game_type", "count_state"],
    ["pitcher_id", "game_type", "batter_hand"],
    ["pitcher_id", "count_state"],
    ["pitcher_id", "game_type"],
    ["pitcher_id", "batter_hand"],
    ["pitcher_id"],
    [],
]
SMOOTHING_GRID = (50.0, 100.0, 200.0)
LAMBDA_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)
MIN_COUNT = 20
CLIP = (0.005, 0.995)


def gain(y: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> float:
    return float(bss(y, candidate) - bss(y, base))


def context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["pitcher_id", "game_type", "batter_hand"]].copy()
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(-1).astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(-1).astype(int)
    out["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
    for column in KEYS:
        out[column] = out[column].fillna("missing").astype(str)
    return out[KEYS]


def build_tables(source: pd.DataFrame, smoothing: float) -> dict[str, object]:
    prior = float(source["residual"].mean())
    tables: list[tuple[list[str], pd.DataFrame]] = []
    for keys in HIERARCHY:
        if not keys:
            table = pd.DataFrame({"correction": [prior], "count": [len(source)]})
        else:
            table = (
                source.groupby(keys, dropna=False, observed=True, sort=False)["residual"]
                .agg(["sum", "count"]).reset_index()
            )
            table = table.loc[table["count"].ge(MIN_COUNT)].copy()
            table["correction"] = (
                table["sum"] + smoothing * prior
            ) / (table["count"] + smoothing)
            table = table[keys + ["correction", "count"]]
        tables.append((keys, table))
    return {"prior": prior, "tables": tables, "smoothing": smoothing}


def lookup(frame: pd.DataFrame, artifact: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys_frame = context_frame(frame)
    correction = np.full(len(frame), np.nan, dtype=float)
    count = np.full(len(frame), np.nan, dtype=float)
    level = np.full(len(frame), -1, dtype=np.int8)
    for position, (keys, table) in enumerate(artifact["tables"]):
        if keys:
            merged = keys_frame[keys].merge(
                table, on=keys, how="left", validate="many_to_one", sort=False,
            )
            values = merged["correction"].to_numpy(float)
            sizes = merged["count"].to_numpy(float)
        else:
            values = np.full(len(frame), float(table.iloc[0]["correction"]))
            sizes = np.full(len(frame), float(table.iloc[0]["count"]))
        fill = np.isnan(correction) & np.isfinite(values)
        correction[fill] = values[fill]
        count[fill] = sizes[fill]
        level[fill] = position
    if np.isnan(correction).any():
        raise RuntimeError("hierarchical lookup left unmatched rows")
    multiplier = np.sqrt(count / (count + float(artifact["smoothing"])))
    return correction * multiplier, count, level


def cluster_bootstrap(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    pitcher: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    reference = float(y.mean() * (1.0 - y.mean()))
    row_gain = np.square(base - y) - np.square(candidate - y)
    grouped = pd.DataFrame({"pitcher": pitcher.astype(str), "gain": row_gain}).groupby(
        "pitcher", sort=False,
    )["gain"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 64):
        size = min(64, repetitions - start)
        sampled = rng.integers(0, len(grouped), size=(size, len(grouped)))
        values[start:start + size] = (
            100_000.0 * sums[sampled].sum(axis=1)
            / sizes[sampled].sum(axis=1) / reference
        )
    return {
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
        "pitchers": int(len(grouped)),
    }


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
        usecols=[
            "season", "game_month", "pitcher_id", "game_type", "batter_hand",
            "balls_before", "strikes_before", "control_success",
        ],
        encoding="utf-8-sig", low_memory=False,
    )
    rows = pd.concat(
        [raw.loc[raw["season"].eq(year)] for year in (2023, 2024)],
        ignore_index=True,
    )
    if len(rows) != len(y):
        raise ValueError("v64 OOF length and official rows differ")
    if not np.array_equal(rows["season"].to_numpy(int), season):
        raise ValueError("v64 OOF season order is not aligned")
    if not np.array_equal(rows["control_success"].to_numpy(float), y):
        raise ValueError("v64 OOF targets are not aligned")
    rows = rows.reset_index(drop=True)
    normalized_context = context_frame(rows)
    for column in KEYS:
        rows[column] = normalized_context[column].to_numpy()
    rows["target"] = y
    rows["baseline"] = base
    rows["residual"] = y - base

    positions23 = np.flatnonzero(season == 2023)
    split23 = len(positions23) // 2
    folds = [
        {
            "name": "2023_second_half",
            "source": rows.iloc[positions23[:split23]].copy(),
            "validation": rows.iloc[positions23[split23:]].copy(),
        },
        {
            "name": "2024_forward",
            "source": rows.loc[rows["season"].eq(2023)].copy(),
            "validation": rows.loc[rows["season"].eq(2024)].copy(),
        },
    ]

    prepared: dict[float, list[dict[str, object]]] = {}
    for smoothing in SMOOTHING_GRID:
        prepared[smoothing] = []
        for fold in folds:
            artifact = build_tables(fold["source"], smoothing)
            correction, count, level = lookup(fold["validation"], artifact)
            prepared[smoothing].append({
                **fold,
                "correction": correction,
                "count": count,
                "level": level,
                "prior": float(artifact["prior"]),
            })

    candidates: list[dict[str, object]] = []
    for smoothing in SMOOTHING_GRID:
        for r_lambda in LAMBDA_GRID:
            for f_lambda in LAMBDA_GRID:
                evaluations: list[dict[str, object]] = []
                for fold in prepared[smoothing]:
                    valid = fold["validation"]
                    target = valid["target"].to_numpy(float)
                    baseline = valid["baseline"].to_numpy(float)
                    regular = valid["game_type"].astype(str).eq("R").to_numpy()
                    scale = np.where(regular, r_lambda, f_lambda)
                    prediction = np.clip(
                        baseline + scale * fold["correction"], *CLIP,
                    )
                    halves = np.array_split(np.arange(len(valid)), 2)
                    groups = {
                        label: gain(target[mask], baseline[mask], prediction[mask])
                        for label, mask in (("R", regular), ("F", ~regular))
                    }
                    evaluations.append({
                        "fold": fold["name"],
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
                    and min(value for item in evaluations for value in item["half_gains"]) >= 0.0
                    and min(value for item in evaluations for value in item["group_gains"].values()) >= 0.0
                )
                gains = [item["gain"] for item in evaluations]
                candidates.append({
                    "smoothing": smoothing,
                    "r_lambda": r_lambda,
                    "f_lambda": f_lambda,
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
    bootstraps: dict[str, object] = {}
    level_usage: dict[str, object] = {}
    for fold in prepared[float(best["smoothing"])]:
        valid = fold["validation"]
        target = valid["target"].to_numpy(float)
        baseline = valid["baseline"].to_numpy(float)
        regular = valid["game_type"].astype(str).eq("R").to_numpy()
        scale = np.where(regular, best["r_lambda"], best["f_lambda"])
        prediction = np.clip(baseline + scale * fold["correction"], *CLIP)
        bootstraps[fold["name"]] = cluster_bootstrap(
            target, baseline, prediction,
            valid["pitcher_id"].to_numpy(), args.bootstrap,
            651135 + int(valid["season"].iloc[-1]),
        )
        unique, counts = np.unique(fold["level"], return_counts=True)
        level_usage[fold["name"]] = {
            str(int(level)): int(count) for level, count in zip(unique, counts)
        }
    strict = bool(
        best["preliminary_gate"]
        and min(value["ci_low"] for value in bootstraps.values()) > 0.0
    )
    report = {
        "baseline": "v64_public_method_transfer",
        "method": "independent hierarchical context residual reliability correction",
        "source_folds": {
            "2023_second_half": "first half of 2023 only",
            "2024_forward": "all 2023 rows only",
        },
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
    path = ROOT / "research/v65_hierarchical_context_lookup.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
