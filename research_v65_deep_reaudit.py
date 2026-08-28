"""Re-audit legacy neural and auxiliary models against the public-best v54.

The original experiments used older OOF anchors.  This script keeps each saved
model direction fixed across 2023 and 2024, applies it to v54 in log-odds space,
and rejects directions that do not improve every chronological slice.  It is
research-only and never reads evaluation rows or 2025 Trackman data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v40_failure_seed_stability import logit, sigmoid
from research_v58_fraction_roster_stability import clustered_interval


ROOT = Path(__file__).resolve().parent
YEARS = (2023, 2024)
GATES = ("all", "R", "F")
WEIGHTS = np.round(np.arange(-.25, .2501, .01), 3)


def load_sources() -> dict[str, dict[int, np.ndarray]]:
    sources: dict[str, dict[int, np.ndarray]] = {}
    for year in YEARS:
        with np.load(ROOT / f"research/multitask_catboost_{year}.npz") as archive:
            for index, name in enumerate(archive["variants"].astype(str)):
                sources.setdefault(f"multitask_{name}", {})[year] = (
                    archive["predictions"][:, index].astype(float)
                )

    with np.load(ROOT / "research/v23_embedding_mlp.npz") as archive:
        for index, name in enumerate(archive["names"].astype(str)):
            for year in YEARS:
                sources.setdefault(f"embedding_{name}", {})[year] = (
                    archive[f"prediction_{year}"][:, index].astype(float)
                )

    for year in YEARS:
        with np.load(ROOT / f"research/v24_tabm_{year}.npz") as archive:
            for name in sorted(key for key in archive.files if key.startswith("epoch_")):
                sources.setdefault(f"tabm_{name}", {})[year] = archive[name].astype(float)
    return sources


def chronological_masks(length: int) -> dict[str, np.ndarray]:
    indices = np.arange(length)
    boundaries = np.linspace(0, length, 5, dtype=int)
    result = {}
    for quarter in range(4):
        result[f"q{quarter + 1}"] = (
            (indices >= boundaries[quarter]) & (indices < boundaries[quarter + 1])
        )
    midpoint = length // 2
    result["h1"] = indices < midpoint
    result["h2"] = indices >= midpoint
    return result


def gain_grid(target, base, prediction, selected, segment_masks):
    """Evaluate every fixed blend weight in one vectorized probability matrix."""
    chosen = np.flatnonzero(selected)
    y_selected = target[chosen]
    base_selected = base[chosen]
    direction = (
        logit(np.clip(prediction[chosen], .005, .995)) - logit(base_selected)
    )
    probabilities = sigmoid(
        logit(base_selected)[None, :] + WEIGHTS[:, None] * direction[None, :]
    )
    delta = (
        (y_selected - base_selected)[None, :] ** 2
        - (y_selected[None, :] - probabilities) ** 2
    )
    gains = {}
    for name, mask in segment_masks.items():
        if int(mask.sum()) < 500:
            continue
        affected = mask[chosen]
        prevalence = float(target[mask].mean())
        denominator = prevalence * (1. - prevalence)
        gains[name] = (
            100000. * delta[:, affected].sum(axis=1) / float(mask.sum()) / denominator
        )
    return gains


def main() -> None:
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
        usecols=["season", "game_type", "pitcher_id", "control_success"],
    )
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        season_archive = archive["season"].astype(int)
        target_archive = archive["target"].astype(float)
        base_archive = np.clip(archive["blended"].astype(float), .005, .995)

    frames: dict[int, pd.DataFrame] = {}
    targets: dict[int, np.ndarray] = {}
    bases: dict[int, np.ndarray] = {}
    segments: dict[int, dict[str, np.ndarray]] = {}
    for year in YEARS:
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        active = season_archive == year
        target = target_archive[active]
        if len(rows) != active.sum() or not np.array_equal(
            rows["control_success"].to_numpy(float), target
        ):
            raise ValueError(f"v54 rows do not align with train.csv for {year}")
        game_type = rows["game_type"].astype(str).to_numpy()
        frames[year] = rows
        targets[year] = target
        bases[year] = base_archive[active]
        segments[year] = {
            "all": np.ones(len(rows), dtype=bool),
            "R": game_type == "R",
            "F": game_type == "F",
            **chronological_masks(len(rows)),
        }

    sources = load_sources()
    reports = []
    for model, by_year in sources.items():
        if set(by_year) != set(YEARS):
            continue
        for year in YEARS:
            if len(by_year[year]) != len(targets[year]):
                raise ValueError(f"{model} rows do not align for {year}")
        for gate in GATES:
            grids = {
                year: gain_grid(
                    targets[year], bases[year], by_year[year],
                    segments[year][gate], segments[year],
                )
                for year in YEARS
            }
            for weight_index, weight in enumerate(WEIGHTS):
                year_metrics = {}
                slice_gains = []
                for year in YEARS:
                    gains = {
                        name: float(values[weight_index])
                        for name, values in grids[year].items()
                    }
                    year_metrics[str(year)] = {
                        "gain_all": gains["all"],
                        "gain_gate": gains[gate],
                        "slice_gains": {
                            name: gains[name] for name in ("h1", "h2", "q1", "q2", "q3", "q4")
                        },
                    }
                    slice_gains.extend(year_metrics[str(year)]["slice_gains"].values())
                reports.append({
                    "model": model,
                    "gate": gate,
                    "weight": float(weight),
                    "years": year_metrics,
                    "minimum_year_gain": float(min(
                        year_metrics[str(year)]["gain_all"] for year in YEARS
                    )),
                    "mean_year_gain": float(np.mean([
                        year_metrics[str(year)]["gain_all"] for year in YEARS
                    ])),
                    "minimum_slice_gain": float(min(slice_gains)),
                })

    nonzero = [row for row in reports if abs(row["weight"]) > 1e-12]
    ranked = sorted(
        nonzero,
        key=lambda row: (
            row["minimum_year_gain"], row["minimum_slice_gain"], row["mean_year_gain"],
        ),
        reverse=True,
    )
    audit = {id(row): row for row in ranked[:30]}
    for row in audit.values():
        intervals = {}
        for year in YEARS:
            target = targets[year]
            base = bases[year]
            selected = segments[year][row["gate"]]
            prediction = base.copy()
            direction = (
                logit(np.clip(sources[row["model"]][year], .005, .995)) - logit(base)
            )
            prediction[selected] = sigmoid(
                logit(base[selected]) + row["weight"] * direction[selected]
            )
            intervals[str(year)] = clustered_interval(
                target, base, prediction,
                frames[year]["pitcher_id"].to_numpy(), repeats=2000,
            )
        row["pitcher_clustered_bootstrap"] = intervals

    safe = sorted([
        row for row in audit.values()
        if row["minimum_year_gain"] > 0.
        and row["minimum_slice_gain"] > 0.
        and min(
            row["pitcher_clustered_bootstrap"][str(year)]["p05"] for year in YEARS
        ) > 0.
    ], key=lambda row: row["mean_year_gain"], reverse=True)
    output = {
        "anchor": {"version": "v54", "public_score": 1113},
        "years": list(YEARS),
        "weights": [float(WEIGHTS[0]), float(WEIGHTS[-1]), .01],
        "safe": safe,
        "ranked": ranked[:60],
        "row_independent": True,
        "current_pitch_information_used_at_inference": False,
        "trackman_2025_used": False,
    }
    path = ROOT / "research/v65_deep_reaudit.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"safe": safe[:12], "top": ranked[:12]}, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
