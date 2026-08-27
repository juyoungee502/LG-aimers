"""Jointly screen deployable command, F-resolution, and fine-failure axes.

All axes are frozen from seasons earlier than each validation year.  The search
uses quadratic Brier geometry only as a fast screen; every reported finalist is
then scored with the exact sigmoid/additive inference formula.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss, reconstruct_labels
from research_trackman_failure_prior import FAILURES, aligned_history, selection_delta
from research_v23_combined_candidate import add_pitcher_season_exposure, logit, sigmoid
from research_v23_context_deviation import deviation


AXIS_NAMES = (
    "command_no_month", "command_full", "command_recent",
    "global_logit_shift",
    "f_count", "f_hands", "f_runners",
    "fine_reverse", "fine_middle", "fine_wayoff",
    "pressure_hand", "platoon_hand",
)
LOW = np.asarray([
    .45, -.20, -.20, -.020, -.20, -.20, -.40, -1.00, -.50, -1.00, 0., 0.,
])
HIGH = np.asarray([
    1.80, 1.40, 1.50, .010, .90, .30, .20, .50, 1.00, .50, .60, .30,
])
DEFAULT = np.asarray([
    1.70, .15, 1.20, -.002, .55, -.25, -.25, 0., 0., 0., .25, 0.,
])
RANDOM_CANDIDATES = 180_000
RNG_SEED = 72301


def masks(rows):
    position = np.arange(len(rows))
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": position < len(rows) // 2,
        "second_half": position >= len(rows) // 2,
        "q1": position < len(rows) // 4,
        "q2": (position >= len(rows) // 4) & (position < len(rows) // 2),
        "q3": (position >= len(rows) // 2) & (position < 3 * len(rows) // 4),
        "q4": position >= 3 * len(rows) // 4,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def geometry(target, base, axes, mask):
    uncertainty = float(target[mask].mean() * (1. - target[mask].mean()))
    residual = target[mask] - base[mask]
    selected = axes[mask]
    return (
        200000. * np.mean(selected * residual[:, None], axis=0) / uncertainty,
        100000. * (selected.T @ selected) / (mask.sum() * uncertainty),
    )


def approximate(weights, geometries):
    output = np.empty((len(weights), len(geometries)), dtype=np.float64)
    for index, (_, linear, quadratic) in enumerate(geometries):
        output[:, index] = (
            weights @ linear
            - np.einsum("bi,ij,bj->b", weights, quadratic, weights, optimize=True)
        )
    return output


def candidate_weights():
    rng = np.random.default_rng(RNG_SEED)
    uniform = rng.uniform(LOW, HIGH, size=(RANDOM_CANDIDATES, len(AXIS_NAMES)))
    # Concentrate half the search near independently robust settings while still
    # retaining a broad uniform exploration of interactions.
    local = DEFAULT + rng.normal(
        scale=np.asarray([
            .18, .25, .25, .005, .16, .10, .12, .20, .18, .20, .10, .06,
        ]),
        size=(RANDOM_CANDIDATES, len(AXIS_NAMES)),
    )
    local = np.clip(local, LOW, HIGH)
    constrained_no_fine = DEFAULT + rng.normal(
        scale=np.asarray([
            .22, .30, .30, .006, .18, .12, .14, 0., 0., 0., .12, .07,
        ]),
        size=(RANDOM_CANDIDATES, len(AXIS_NAMES)),
    )
    constrained_no_fine = np.clip(constrained_no_fine, LOW, HIGH)
    constrained_no_fine[:, 7:10] = 0.
    constrained_uniform = rng.uniform(
        np.asarray([
            .80, -.10, .20, -.012, .10, -.30, -.40,
            0., 0., 0., .10, 0.,
        ]),
        np.asarray([
            1.80, 1.20, 1.50, .004, .90, .15, .10,
            0., 0., 0., .60, .10,
        ]),
        size=(RANDOM_CANDIDATES, len(AXIS_NAMES)),
    )
    no_fine = DEFAULT.copy()
    no_fine[7:10] = 0.
    no_context = DEFAULT.copy()
    no_context[10:12] = 0.
    command_f = DEFAULT.copy()
    command_f[7:] = 0.
    command_only = np.zeros(len(AXIS_NAMES))
    command_only[:4] = DEFAULT[:4]
    names = (
        "default", "no_fine", "no_context", "command_f", "command_only",
        "zero",
    )
    ablations = np.vstack([
        DEFAULT, no_fine, no_context, command_f, command_only,
        np.zeros(len(AXIS_NAMES)),
    ])
    weights = np.vstack([
        uniform, local, constrained_no_fine, constrained_uniform, ablations,
    ])
    named_indices = {
        name: len(weights) - len(ablations) + index
        for index, name in enumerate(names)
    }
    return weights, named_indices


def exact_prediction(item, weights):
    command = item["logit_axes"] @ weights[:4]
    additive = item["additive_axes"] @ weights[4:]
    return np.clip(sigmoid(logit(item["base"]) + command) + additive, .005, .995)


def exact_report(item, weights):
    prediction = exact_prediction(item, weights)
    return {
        name: bss(item["target"][mask], prediction[mask])
        - bss(item["target"][mask], item["base"][mask])
        for name, mask in item["masks"].items() if mask.any()
    }


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(
        root / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    data["count_state"] = (
        data["balls_before"].to_numpy(np.int16) * 3
        + data["strikes_before"].to_numpy(np.int16)
    )
    balls = data["balls_before"].to_numpy(np.int16)
    strikes = data["strikes_before"].to_numpy(np.int16)
    data["pressure_state"] = np.where(
        (balls == 3) & (strikes == 2), 2,
        np.where((balls == 3) | (strikes == 2), 1, 0),
    ).astype(np.int8)
    data["season_relative_target"] = (
        data["control_success"]
        - data.groupby(["season", "game_type"], observed=True)[
            "control_success"
        ].transform("mean")
    )
    exposure = add_pitcher_season_exposure(data[[
        "season", "pitcher_id", "asof_pitcher_n",
    ]])
    labels = reconstruct_labels(data)
    history = aligned_history(root, data, labels)
    years = {}
    centers = {}
    for year in (2023, 2024):
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        season_exposure = exposure.loc[
            exposure["season"].eq(year), "pitcher_season_n"
        ].to_numpy(float)
        with np.load(root / f"research/v23_trackman_no_month_{year}.npz") as z:
            target = z["target"].astype(float)
            base = z["base"].astype(float)
            no_month = z["direction"].astype(float)
        with np.load(root / f"research/v23_prior_command_context_{year}.npz") as z:
            full = z["command_direction"].astype(float)
        with np.load(root / f"research/v23_prior_command_context_{year}_w1.npz") as z:
            recent = z["command_direction"].astype(float)
        with np.load(root / f"research/v23_conditional_resolution_{year}.npz") as z:
            resolution_names = z["names"].astype(str).tolist()
            resolution = z["directions"].astype(float)
        expected_resolution = ["regime_count", "regime_count_hands", "regime_count_runners"]
        order = [resolution_names.index(name) for name in expected_resolution]
        resolution = resolution[:, order]

        regular = rows["game_type"].eq("R").to_numpy(float)
        futures = rows["game_type"].eq("F").to_numpy(float)
        early = (season_exposure <= 600).astype(float)
        logit_axes = np.column_stack([
            regular * no_month,
            regular * early * full,
            regular * early * recent,
            np.ones(len(rows), dtype=float),
        ])

        source = history.loc[history["season"].lt(year)]
        proxy = data.loc[data["season"].eq(year - 1)].reset_index(drop=True)
        proxy_regular = proxy["game_type"].eq("R").to_numpy()
        fine_members, fine_centers = [], []
        for failure in FAILURES:
            raw_signal = selection_delta(source, rows, failure, 20., 600.)
            proxy_signal = selection_delta(source, proxy, failure, 20., 600.)
            center = float(proxy_signal[proxy_regular].mean())
            fine_members.append(regular * (raw_signal - center))
            fine_centers.append(center)
        fine = np.column_stack(fine_members)
        source_all = data.loc[data["season"].lt(year)]
        pressure = deviation(
            source_all, rows,
            ("pitcher_id", "pressure_state", "batter_hand"),
            1200., "season_relative_target",
        )
        source_recent_regular = data.loc[
            data["season"].between(year - 3, year - 1)
            & data["game_type"].eq("R")
        ]
        platoon = deviation(
            source_recent_regular, rows, ("pitcher_id", "batter_hand"),
            100., "control_success",
        )
        additive_axes = np.column_stack([
            futures[:, None] * resolution, fine, pressure, platoon,
        ])
        if len(rows) != len(target):
            raise ValueError(f"row alignment failed for {year}")
        years[year] = {
            "rows": rows, "target": target, "base": base,
            "logit_axes": logit_axes, "additive_axes": additive_axes,
            "masks": masks(rows),
        }
        centers[str(year)] = fine_centers
        print(f"Prepared joint axes for {year}", flush=True)

    geometries = []
    labels_for_geometry = []
    for year, item in years.items():
        derivative = item["base"] * (1. - item["base"])
        approximate_axes = np.column_stack([
            derivative[:, None] * item["logit_axes"], item["additive_axes"],
        ])
        for name, mask in item["masks"].items():
            if not mask.any():
                continue
            linear, quadratic = geometry(
                item["target"], item["base"], approximate_axes, mask,
            )
            labels_for_geometry.append((str(year), name))
            geometries.append(((str(year), name), linear, quadratic))

    weights, named_indices = candidate_weights()
    gains = approximate(weights, geometries)
    label_index = {label: index for index, label in enumerate(labels_for_geometry)}
    year_columns = [label_index[(str(year), "all")] for year in (2023, 2024)]
    core_columns = [
        index for index, (_, name) in enumerate(labels_for_geometry)
        if name in (
            "all", "first_half", "second_half", "months_3_5",
            "months_6_7", "months_8_11", "regular", "futures",
        )
    ]
    temporal_columns = [
        index for index, (_, name) in enumerate(labels_for_geometry)
        if name not in ("all", "regular", "futures")
    ]
    objectives = {
        "maximin_core": np.min(gains[:, core_columns], axis=1),
        "maximin_temporal": np.min(gains[:, temporal_columns], axis=1),
        "maximin_year": np.min(gains[:, year_columns], axis=1),
        "mean_year": np.mean(gains[:, year_columns], axis=1),
    }
    selected = set()
    for values in objectives.values():
        count = min(300, len(values))
        selected.update(np.argpartition(values, -count)[-count:].tolist())
        no_fine_indices = np.flatnonzero(
            np.all(np.isclose(weights[:, 7:10], 0.), axis=1)
        )
        constrained_values = values[no_fine_indices]
        constrained_count = min(300, len(constrained_values))
        local_indices = np.argpartition(
            constrained_values, -constrained_count,
        )[-constrained_count:]
        selected.update(no_fine_indices[local_indices].tolist())
    # Retain independently chosen policies even if a quadratic approximation
    # ranks them below interaction-tuned candidates.
    selected.update(named_indices.values())
    print(
        f"Exact scoring {len(selected)} finalists from {len(weights)} candidates",
        flush=True,
    )

    reports = []
    for index in selected:
        candidate = weights[index]
        by_year = {
            str(year): exact_report(item, candidate)
            for year, item in years.items()
        }
        core = [
            value for year_gains in by_year.values()
            for name, value in year_gains.items()
            if name in (
                "all", "first_half", "second_half", "months_3_5",
                "months_6_7", "months_8_11", "regular", "futures",
            )
        ]
        temporal = [
            value for year_gains in by_year.values()
            for name, value in year_gains.items()
            if name not in ("all", "regular", "futures")
        ]
        reports.append({
            "weights": dict(zip(AXIS_NAMES, candidate.tolist())),
            "gains": by_year,
            "min_core": min(core), "min_temporal": min(temporal),
            "min_year": min(by_year["2023"]["all"], by_year["2024"]["all"]),
            "mean_year": np.mean([
                by_year["2023"]["all"], by_year["2024"]["all"],
            ]),
        })

    ranking_functions = {
        "maximin_core": lambda row: (
            row["min_core"], row["min_year"], row["mean_year"],
        ),
        "maximin_temporal": lambda row: (
            row["min_temporal"], row["min_year"], row["mean_year"],
        ),
        "maximin_year": lambda row: (
            row["min_year"], row["min_temporal"], row["mean_year"],
        ),
        "best_mean": lambda row: (
            row["mean_year"], row["min_year"], row["min_temporal"],
        ),
    }
    rankings = {
        name: sorted(reports, key=key, reverse=True)[:100]
        for name, key in ranking_functions.items()
    }
    no_fine_reports = [
        row for row in reports
        if all(abs(row["weights"][name]) < 1e-12 for name in (
            "fine_reverse", "fine_middle", "fine_wayoff",
        ))
    ]
    no_fine_rankings = {
        name: sorted(no_fine_reports, key=key, reverse=True)[:100]
        for name, key in ranking_functions.items()
    }
    report_by_weights = {
        tuple(row["weights"][name] for name in AXIS_NAMES): row
        for row in reports
    }
    named_policies = {
        name: report_by_weights[tuple(weights[index].tolist())]
        for name, index in named_indices.items()
    }
    output = root / "research/v23_joint_candidate_portfolio.json"
    output.write_text(json.dumps({
        "axis_names": AXIS_NAMES, "fine_configuration": {
            "outcome_k": 20., "selection_k": 600., "centers": centers,
        }, "rankings": rankings, "no_fine_rankings": no_fine_rankings,
        "named_policies": named_policies,
    }, indent=2), encoding="utf-8")
    print(json.dumps({name: rows[:12] for name, rows in rankings.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
