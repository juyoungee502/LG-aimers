"""Strict cross-year audit of paired multi-task TabM directions over v54."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_v40_failure_seed_stability import logit, sigmoid
from research_v58_fraction_roster_stability import (
    clustered_interval, roster_masks,
)


ROOT = Path(__file__).resolve().parent
YEARS = (2023, 2024)
GATES = ("all", "R", "F")
WEIGHTS = np.round(np.arange(-2., 2.001, .10), 2)


def time_masks(length):
    position = np.arange(length)
    boundaries = np.linspace(0, length, 5, dtype=int)
    result = {
        "all": np.ones(length, dtype=bool),
        "h1": position < length // 2,
        "h2": position >= length // 2,
    }
    for index in range(4):
        result[f"q{index + 1}"] = (
            (position >= boundaries[index]) & (position < boundaries[index + 1])
        )
    return result


def gain_grid(target, base, direction, selected, masks):
    chosen = np.flatnonzero(selected)
    y = target[chosen].astype(np.float32)
    before = base[chosen].astype(np.float32)
    logits = (
        logit(before).astype(np.float32)[None, :]
        + WEIGHTS.astype(np.float32)[:, None] * direction[chosen].astype(np.float32)[None, :]
    )
    prediction = sigmoid(logits).astype(np.float32)
    delta = (y - before)[None, :] ** 2 - (y[None, :] - prediction) ** 2
    gains = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        if count < 500:
            continue
        affected = mask[chosen]
        rate = float(target[mask].mean())
        gains[name] = (
            100000. * delta[:, affected].sum(axis=1, dtype=np.float64)
            / count / (rate * (1. - rate))
        )
    return gains


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}

    rows_by_year = {}
    target_by_year = {}
    base_by_year = {}
    masks_by_year = {}
    direction_by_epoch = {epoch: {} for epoch in range(1, 5)}
    auxiliary_weight = None
    for year in YEARS:
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        active = oof["season"] == year
        target = oof["target"][active].astype(float)
        base = np.clip(oof["blended"][active].astype(float), .005, .995)
        if len(rows) != len(target) or not np.array_equal(
            rows["control_success"].to_numpy(float), target
        ):
            raise ValueError(f"v54 rows do not align for {year}")
        with np.load(ROOT / f"research/v66_multitask_tabm_{year}.npz") as archive:
            if not np.allclose(archive["target"], target):
                raise ValueError(f"v66 rows do not align for {year}")
            fold_aux_weight = float(archive["aux_weight"])
            if auxiliary_weight is None:
                auxiliary_weight = fold_aux_weight
            elif not np.isclose(auxiliary_weight, fold_aux_weight):
                raise ValueError("Fold auxiliary weights differ")
            for epoch in range(1, 5):
                main_prediction = np.clip(
                    archive[f"main_epoch_{epoch}"].astype(float), .005, .995,
                )
                aux_prediction = np.clip(
                    archive[f"aux_epoch_{epoch}"].astype(float), .005, .995,
                )
                direction_by_epoch[epoch][year] = (
                    logit(aux_prediction) - logit(main_prediction)
                )
        game_type = rows["game_type"].astype(str).to_numpy()
        masks = {
            **time_masks(len(rows)),
            "R": game_type == "R",
            "F": game_type == "F",
            **{f"roster_{name}": mask for name, mask in roster_masks(raw, rows, year).items()},
        }
        rows_by_year[year] = rows
        target_by_year[year] = target
        base_by_year[year] = base
        masks_by_year[year] = masks

    reports = []
    chronological_names = ("h1", "h2", "q1", "q2", "q3", "q4")
    for epoch, directions in direction_by_epoch.items():
        for gate in GATES:
            grids = {
                year: gain_grid(
                    target_by_year[year], base_by_year[year], directions[year],
                    masks_by_year[year][gate], masks_by_year[year],
                )
                for year in YEARS
            }
            for weight_index, weight in enumerate(WEIGHTS):
                years = {}
                chronological = []
                roster = []
                for year in YEARS:
                    gains = {
                        name: float(values[weight_index])
                        for name, values in grids[year].items()
                    }
                    year_chronological = {
                        name: gains[name] for name in chronological_names
                    }
                    year_roster = {
                        name: value for name, value in gains.items()
                        if name.startswith("roster_")
                    }
                    years[str(year)] = {
                        "gain_all": gains["all"],
                        "gain_gate": gains[gate],
                        "chronological": year_chronological,
                        "roster": year_roster,
                    }
                    chronological.extend(year_chronological.values())
                    roster.extend(year_roster.values())
                reports.append({
                    "epoch": epoch,
                    "gate": gate,
                    "weight": float(weight),
                    "years": years,
                    "minimum_year_gain": float(min(
                        years[str(year)]["gain_all"] for year in YEARS
                    )),
                    "mean_year_gain": float(np.mean([
                        years[str(year)]["gain_all"] for year in YEARS
                    ])),
                    "minimum_chronological_gain": float(min(chronological)),
                    "minimum_roster_gain": float(min(roster)),
                })

    ranked = sorted(
        [row for row in reports if abs(row["weight"]) > 1e-12],
        key=lambda row: (
            row["minimum_year_gain"], row["minimum_chronological_gain"],
            row["minimum_roster_gain"], row["mean_year_gain"],
        ),
        reverse=True,
    )
    audited = ranked[:30]
    for report in audited:
        intervals = {}
        for year in YEARS:
            target = target_by_year[year]
            base = base_by_year[year]
            selected = masks_by_year[year][report["gate"]]
            prediction = base.copy()
            prediction[selected] = sigmoid(
                logit(base[selected])
                + report["weight"] * direction_by_epoch[report["epoch"]][year][selected]
            )
            intervals[str(year)] = clustered_interval(
                target, base, prediction,
                rows_by_year[year]["pitcher_id"].to_numpy(), repeats=2000,
            )
        report["pitcher_clustered_bootstrap"] = intervals

    safe = sorted([
        row for row in audited
        if row["minimum_year_gain"] > 0.
        and row["minimum_chronological_gain"] > 0.
        and row["minimum_roster_gain"] > 0.
        and min(
            row["pitcher_clustered_bootstrap"][str(year)]["p05"]
            for year in YEARS
        ) > 0.
    ], key=lambda row: row["mean_year_gain"], reverse=True)
    output = {
        "anchor": {"version": "v54", "public_score": 1113},
        "auxiliary_weight": auxiliary_weight,
        "safe": safe,
        "ranked": ranked[:60],
        "selection_rule": (
            "One fixed epoch/gate/weight must improve both years, every half and "
            "quarter, every sufficiently large roster cohort, and both pitcher-"
            "clustered bootstrap lower bounds."
        ),
        "row_independent": True,
        "current_pitch_information_used_at_inference": False,
        "forbidden_2025_trackman_used": False,
    }
    path = ROOT / "research/v66_multitask_audit.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"safe": safe[:12], "top": ranked[:12]}, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
