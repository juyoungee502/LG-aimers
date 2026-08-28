"""Audit post-v38 candidates under roster and team turnover in 2024."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss
from research_v40_failure_seed_stability import logit, masks, sigmoid


ROOT = Path(__file__).resolve().parent


def candidate_score(target, prediction, active):
    if int(active.sum()) < 500:
        return None
    return float(bss(target[active], prediction[active]))


def clustered_interval(target, base, prediction, pitcher_ids, repeats=2000):
    delta = (target - base) ** 2 - (target - prediction) ** 2
    frame = pd.DataFrame({"pitcher": pitcher_ids, "delta": delta})
    groups = frame.groupby("pitcher", observed=True)["delta"].agg(["sum", "count"])
    values = groups[["sum", "count"]].to_numpy(float)
    rng = np.random.default_rng(5300)
    draws = np.empty(repeats, dtype=float)
    denominator = float(target.mean() * (1. - target.mean()))
    for index in range(repeats):
        selected = rng.integers(0, len(values), size=len(values))
        sampled = values[selected].sum(axis=0)
        draws[index] = 100000. * sampled[0] / sampled[1] / denominator
    return {
        "p05": float(np.quantile(draws, .05)),
        "median": float(np.quantile(draws, .50)),
        "p95": float(np.quantile(draws, .95)),
        "positive_probability": float(np.mean(draws > 0.)),
        "pitcher_clusters": int(len(values)),
    }


def main():
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    valid = raw["season"].eq(2024).to_numpy()
    rows = raw.loc[valid].reset_index(drop=True)
    target = rows["control_success"].to_numpy(float)
    game_type = rows["game_type"].astype(str).to_numpy()
    active_f = game_type == "F"

    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        base = np.clip(
            archive["blended"][archive["season"] == 2024].astype(float),
            .005, .995,
        )
    with np.load(
        ROOT / "research/v34_categorical_failure_lowcard_no_ids_hl2_2024.npz"
    ) as archive:
        failure = archive["new_failure"].astype(float)
    with np.load(ROOT / "research/v43_multiclass_hl2_s1_2024.npz") as archive:
        multiclass = archive["prediction"].astype(float)
    with np.load(ROOT / "research/v45_overlap_hl2_2024.npz") as archive:
        overlap = archive["overlap_probability"].astype(float)
    with np.load(ROOT / "research/v48_regime_command_s3_2024.npz") as archive:
        recent_a = archive["multiclass"].astype(float)
    with np.load(
        ROOT / "research/v49_regime_multiclass_complexity_2024.npz"
    ) as archive:
        recent_b = archive["d6_i1000"].astype(float)
    with np.load(ROOT / "research/v52_pitch_command_joint_s3_2024.npz") as archive:
        joint_history = archive["history"].astype(float)
        joint_no_team = archive["history_no_team"].astype(float)
    if not all(len(value) == len(base) for value in (
        target, failure, multiclass, overlap, recent_a, recent_b,
        joint_history, joint_no_team,
    )):
        raise ValueError("candidate rows do not align")

    corrected = np.clip(failure + .45 * overlap, .005, .995)
    v46 = base.copy()
    v46[active_f] = sigmoid(
        logit(base[active_f])
        + .10 * (logit(multiclass[active_f]) - logit(base[active_f]))
        + .075 * (logit(corrected[active_f]) - logit(base[active_f]))
    )
    recent = .5 * (recent_a + recent_b)
    recent_direction = logit(recent) - logit(base)
    joint_average = .5 * (joint_history + joint_no_team)
    joint_directions = {
        "joint_history": logit(joint_history) - logit(base),
        "joint_no_team": logit(joint_no_team) - logit(base),
        "joint_average": logit(joint_average) - logit(base),
    }
    candidates = {"v38": base, "v46": v46}
    for weight in (.025, .05, .075):
        value = v46.copy()
        value[active_f] = sigmoid(
            logit(v46[active_f]) + weight * recent_direction[active_f]
        )
        candidates[f"v46_recent_{weight:g}"] = value
    anchor = candidates["v46_recent_0.05"]
    for name, direction in joint_directions.items():
        for weight in (.0125, .01875, .025):
            candidates[f"anchor_{name}_{weight:g}"] = sigmoid(
                logit(anchor) + weight * direction
            )

    previous = raw.loc[raw["season"].eq(2023)]
    previous_pitchers = set(previous["pitcher_id"].astype(int))
    previous_batters = set(previous["batter_id"].astype(int))
    pitcher_returning = rows["pitcher_id"].astype(int).isin(previous_pitchers).to_numpy()
    batter_returning = rows["batter_id"].astype(int).isin(previous_batters).to_numpy()
    last_pitcher_team = (
        previous.groupby("pitcher_id", observed=True, sort=False).tail(1)
        .set_index("pitcher_id")["pitcher_team_id"]
    )
    last_batter_team = (
        previous.groupby("batter_id", observed=True, sort=False).tail(1)
        .set_index("batter_id")["batter_team_id"]
    )
    prior_pitcher_team = rows["pitcher_id"].map(last_pitcher_team)
    prior_batter_team = rows["batter_id"].map(last_batter_team)
    same_pitcher_team = (
        prior_pitcher_team.eq(rows["pitcher_team_id"]) & pitcher_returning
    ).to_numpy()
    same_batter_team = (
        prior_batter_team.eq(rows["batter_team_id"]) & batter_returning
    ).to_numpy()

    previous_pitcher_end = (
        previous.groupby("pitcher_id", observed=True, sort=False).tail(1)
        .set_index("pitcher_id")["asof_pitcher_n"] + 1.
    )
    pitcher_origin = rows["pitcher_id"].map(previous_pitcher_end).fillna(0.).to_numpy(float)
    pitcher_exposure = np.maximum(
        0., rows["asof_pitcher_n"].to_numpy(float) - pitcher_origin,
    )
    cohorts = {
        "all": np.ones(len(rows), dtype=bool),
        "R": game_type == "R", "F": active_f,
        "F_returning_both": active_f & pitcher_returning & batter_returning,
        "F_roster_change": active_f & ~(pitcher_returning & batter_returning),
        "F_same_teams": active_f & same_pitcher_team & same_batter_team,
        "F_player_or_team_change": active_f & ~(same_pitcher_team & same_batter_team),
        "F_low_pitcher_exposure": active_f & (pitcher_exposure <= 100.),
        "F_high_pitcher_exposure": active_f & (pitcher_exposure > 100.),
    }
    for name, active in masks(len(rows)).items():
        cohorts[name] = active

    base_scores = {
        name: candidate_score(target, base, active)
        for name, active in cohorts.items()
    }
    reports = {}
    for name, prediction in candidates.items():
        scores = {
            cohort: candidate_score(target, prediction, active)
            for cohort, active in cohorts.items()
        }
        gains = {
            cohort: scores[cohort] - base_scores[cohort]
            for cohort in cohorts if scores[cohort] is not None
        }
        team_gains = {}
        for team, index in rows.loc[active_f].groupby(
            "pitcher_team_id", observed=True,
        ).groups.items():
            active = np.zeros(len(rows), dtype=bool)
            active[np.asarray(list(index), dtype=int)] = True
            if active.sum() >= 500:
                team_gains[str(team)] = float(
                    bss(target[active], prediction[active])
                    - bss(target[active], base[active])
                )
        reports[name] = {
            "scores": scores, "gains": gains,
            "minimum_roster_gain": float(min(
                gains[key] for key in (
                    "F_returning_both", "F_roster_change", "F_same_teams",
                    "F_player_or_team_change", "F_low_pitcher_exposure",
                    "F_high_pitcher_exposure",
                ) if key in gains
            )),
            "team_gains": team_gains,
            "minimum_team_gain": float(min(team_gains.values())),
            "median_team_gain": float(np.median(list(team_gains.values()))),
            "clustered_bootstrap": clustered_interval(
                target, base, prediction, rows["pitcher_id"].to_numpy(),
            ),
        }

    output = ROOT / "research/v53_roster_stability.json"
    output.write_text(json.dumps({
        "cohort_rows": {name: int(active.sum()) for name, active in cohorts.items()},
        "base_scores": base_scores, "reports": reports,
    }, indent=2), encoding="utf-8")
    compact = {
        name: {
            "gain_all": report["gains"]["all"],
            "gain_F": report["gains"]["F"],
            "quarters": [report["gains"][f"q{i}"] for i in range(1, 5)],
            "minimum_roster_gain": report["minimum_roster_gain"],
            "minimum_team_gain": report["minimum_team_gain"],
            "bootstrap": report["clustered_bootstrap"],
        } for name, report in reports.items()
    }
    print(json.dumps({
        "cohort_rows": {name: int(active.sum()) for name, active in cohorts.items()},
        "candidates": compact,
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
