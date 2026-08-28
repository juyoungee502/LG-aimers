"""Validate and promote one conservative R residual table over v56.

The table is learned only from prior training rows.  At inference it is a
frozen row-local lookup, so neither 2025 Trackman data nor aggregation across
test rows is required.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)
from research_v40_failure_seed_stability import logit, masks, sigmoid
from research_v53_roster_stability import clustered_interval
from train_v25_temporal_portfolio import bss
from v25_temporal_portfolio import apply_regime, freeze_regime


ROOT = Path(__file__).resolve().parent
VERSION = "v57_conservative_r_table"
F_SCALE = 1.25
POLICY = ({
    "type": "one_d",
    "kind": "numeric",
    "column": "pitcher_success_x_runners",
    "bins": 8,
    "shrink": 6400.0,
    "scale": 1.0,
    "weight": 0.5,
},)
CLIP = (0.005, 0.995)
EXPOSURE_FEATURE = "pitcher_season_n"
MIN_EXPOSURE = 100.0
warnings.filterwarnings("ignore", category=PerformanceWarning)


def score(target: np.ndarray, prediction: np.ndarray, active: np.ndarray):
    if int(active.sum()) < 500:
        return None
    return bss(target[active], prediction[active])


def gain_report(
    target: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    cohorts: dict[str, np.ndarray],
) -> tuple[dict, dict]:
    scores = {}
    gains = {}
    for name, active in cohorts.items():
        base_score = score(target, base, active)
        candidate_score = score(target, candidate, active)
        if base_score is not None:
            scores[name] = {
                "base": float(base_score),
                "candidate": float(candidate_score),
            }
            gains[name] = float(candidate_score - base_score)
    return scores, gains


def main() -> None:
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    target_all = target_series.to_numpy(float)
    history = training_history_arrays(raw, target_series)
    features_all = engineer_features(
        raw, *history, global_prior=float(target_series.mean()),
    )
    add_training_component_features(features_all, raw)
    features_all = add_state_interactions(features_all)

    with np.load(ROOT / "outputs/v38_oof_predictions.npz") as archive:
        v38 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54 = {key: archive[key] for key in archive.files}

    seasons = raw["season"].to_numpy(np.int16)
    positions = np.concatenate([
        np.flatnonzero(seasons == year) for year in (2023, 2024)
    ])
    rows = raw.iloc[positions].reset_index(drop=True)
    features = features_all.iloc[positions].reset_index(drop=True)
    target = v38["target"].astype(float)
    year = v38["season"].astype(int)
    if not np.allclose(target_all[positions], target):
        raise ValueError("OOF rows do not align with train.csv")

    # Reconstruct the public v56 anchor exactly from v38 and v54.
    anchor = v54["blended"].astype(float).copy()
    active_2024 = year == 2024
    active_2024_f = (
        active_2024 & rows["game_type"].astype(str).eq("F").to_numpy()
    )
    anchor[active_2024_f] = sigmoid(
        logit(v38["blended"][active_2024_f].astype(float))
        + F_SCALE * (
            logit(v54["blended"][active_2024_f].astype(float))
            - logit(v38["blended"][active_2024_f].astype(float))
        )
    )

    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    source = np.flatnonzero((year == 2023) & regular)
    valid = np.flatnonzero((year == 2024) & regular)
    validation_table = freeze_regime(
        rows.iloc[source], features.iloc[source], anchor[source], target[source],
        POLICY, (),
    )
    prediction = anchor.copy()
    validation_gate = (
        features.iloc[valid][EXPOSURE_FEATURE].to_numpy(float) > MIN_EXPOSURE
    ).astype(float)
    prediction[valid] = np.clip(
        anchor[valid] + validation_gate * apply_regime(
            rows.iloc[valid], features.iloc[valid], anchor[valid],
            validation_table,
        ),
        *CLIP,
    )

    current = rows.loc[active_2024].reset_index(drop=True)
    current_target = target[active_2024]
    current_base = anchor[active_2024]
    current_prediction = prediction[active_2024]
    current_regular = current["game_type"].astype(str).eq("R").to_numpy()
    cohorts: dict[str, np.ndarray] = {
        "all": np.ones(len(current), dtype=bool),
        "R": current_regular,
        "F": ~current_regular,
    }
    for name, active in masks(len(current)).items():
        cohorts[name] = active
        cohorts[f"R_{name}"] = current_regular & active

    previous = raw.loc[raw["season"].eq(2023)]
    previous_pitchers = set(previous["pitcher_id"].astype(str))
    previous_batters = set(previous["batter_id"].astype(str))
    pitcher_returning = current["pitcher_id"].astype(str).isin(
        previous_pitchers,
    ).to_numpy()
    batter_returning = current["batter_id"].astype(str).isin(
        previous_batters,
    ).to_numpy()
    last_pitcher_team = (
        previous.groupby("pitcher_id", observed=True, sort=False).tail(1)
        .set_index("pitcher_id")["pitcher_team_id"]
    )
    last_batter_team = (
        previous.groupby("batter_id", observed=True, sort=False).tail(1)
        .set_index("batter_id")["batter_team_id"]
    )
    same_pitcher_team = (
        current["pitcher_id"].map(last_pitcher_team).eq(
            current["pitcher_team_id"],
        ) & pitcher_returning
    ).to_numpy()
    same_batter_team = (
        current["batter_id"].map(last_batter_team).eq(
            current["batter_team_id"],
        ) & batter_returning
    ).to_numpy()
    previous_pitcher_end = (
        previous.groupby("pitcher_id", observed=True, sort=False).tail(1)
        .set_index("pitcher_id")["asof_pitcher_n"] + 1.0
    )
    pitcher_origin = current["pitcher_id"].map(
        previous_pitcher_end,
    ).fillna(0.0).to_numpy(float)
    pitcher_exposure = np.maximum(
        0.0, current["asof_pitcher_n"].to_numpy(float) - pitcher_origin,
    )
    returning_both = pitcher_returning & batter_returning
    same_teams = same_pitcher_team & same_batter_team
    cohorts.update({
        "R_returning_both": current_regular & returning_both,
        "R_roster_change": current_regular & ~returning_both,
        "R_same_teams": current_regular & same_teams,
        "R_player_or_team_change": current_regular & ~same_teams,
        "R_low_pitcher_exposure": current_regular & (pitcher_exposure <= 100.0),
        "R_high_pitcher_exposure": current_regular & (pitcher_exposure > 100.0),
    })
    scores, gains = gain_report(
        current_target, current_base, current_prediction, cohorts,
    )

    team_gains = {}
    for team, indices in current.loc[current_regular].groupby(
        "pitcher_team_id", observed=True,
    ).groups.items():
        active = np.zeros(len(current), dtype=bool)
        active[np.asarray(list(indices), dtype=int)] = True
        if active.sum() >= 500:
            team_gains[str(team)] = float(
                bss(current_target[active], current_prediction[active])
                - bss(current_target[active], current_base[active])
            )

    bootstrap = clustered_interval(
        current_target[current_regular],
        current_base[current_regular],
        current_prediction[current_regular],
        current.loc[current_regular, "pitcher_id"].to_numpy(),
    )
    robust_names = (
        "R_returning_both", "R_roster_change", "R_same_teams",
        "R_player_or_team_change", "R_low_pitcher_exposure",
        "R_high_pitcher_exposure",
    )
    report = {
        "anchor": "v56_v54_regime_scaling",
        "public_anchor_score": 1113.86,
        "policy": POLICY[0],
        "exposure_gate": {
            "feature": EXPOSURE_FEATURE,
            "minimum_exclusive": MIN_EXPOSURE,
        },
        "cohort_rows": {
            name: int(active.sum()) for name, active in cohorts.items()
        },
        "scores": scores,
        "gains": gains,
        "minimum_r_quarter_gain": float(min(
            gains[f"R_q{index}"] for index in range(1, 5)
        )),
        "minimum_roster_gain": float(min(
            gains[name] for name in robust_names if name in gains
        )),
        "team_gains": team_gains,
        "minimum_team_gain": float(min(team_gains.values())),
        "median_team_gain": float(np.median(list(team_gains.values()))),
        "clustered_bootstrap_R": bootstrap,
        "selection_note": (
            "Single no-ID table selected after conservative chronological "
            "screening; no multi-table portfolio or global sharpening."
        ),
        "row_independent_inference": True,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    report_path = ROOT / "research/v57_conservative_r_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    np.savez_compressed(
        ROOT / "outputs/v57_oof_predictions.npz",
        **{key: value for key, value in v54.items() if key != "blended"},
        blended=prediction,
    )

    # The deploy table uses all legal 2024 R rows and their strictly OOF anchor
    # predictions.  It is frozen before any 2025 evaluation rows are observed.
    deploy_source = np.flatnonzero((year == 2024) & regular)
    deploy = freeze_regime(
        rows.iloc[deploy_source], features.iloc[deploy_source],
        anchor[deploy_source], target[deploy_source], POLICY, (),
    )
    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = metadata.get("model_names", [])
    if not {
        "v54_roster_robust_command", "v56_v54_regime_scaling",
    }.issubset(names):
        raise ValueError("v57 requires complete v56 model artifacts")
    if VERSION not in names:
        names.append(VERSION)
    metadata["model_names"] = names
    metadata["version"] = VERSION
    metadata[VERSION] = {
        "regular": deploy,
        "exposure_gate": {
            "feature": EXPOSURE_FEATURE,
            "minimum_exclusive": MIN_EXPOSURE,
        },
        "source_season": 2024,
        "target_season": 2025,
        "game_type_regular": "R",
        "anchor": "v56_v54_regime_scaling",
        "public_anchor_score": 1113.86,
        "validation": report,
        "row_independent_inference": True,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps({
        "gain_2024": gains["all"],
        "gain_2024_R": gains["R"],
        "r_quarters": [gains[f"R_q{i}"] for i in range(1, 5)],
        "minimum_roster_gain": report["minimum_roster_gain"],
        "minimum_team_gain": report["minimum_team_gain"],
        "bootstrap": bootstrap,
    }, indent=2), flush=True)
    print(f"Saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
