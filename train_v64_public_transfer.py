"""Train and promote v64 from the clean v61 baseline.

v64 combines two disjoint, publicly documented structures rebuilt on official
data: an F-only conditional residual model and an R-only prior-season pitcher
state.  No v62/v63 component is retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from research_v64_dynamic_pitcher_state import (
    career_before,
    dynamic_deltas,
    fit_ar1,
    latest_state,
    season_states,
)
from research_v64_public_f_regime import SEEDS, parameters, prior_game_type_table
from train_v25_temporal_portfolio import bss
from v64_public_transfer import F_CATEGORICAL, build_f_features


ROOT = Path(__file__).resolve().parent
VERSION = "v64_public_method_transfer"
F_SCALE = 0.15
R_STRENGTH = 100.0
R_WEIGHT = 0.25
CLIP = (0.005, 0.995)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    return parser.parse_args()


def fit_members(
    train_x: pd.DataFrame,
    residual: np.ndarray,
    args: argparse.Namespace,
    *,
    sample_weight: np.ndarray | None = None,
    save: bool = False,
) -> tuple[list[CatBoostRegressor], list[float]]:
    if sample_weight is None:
        target_center = float(np.mean(residual))
    else:
        target_center = float(np.average(residual, weights=sample_weight))
    target = np.asarray(residual, dtype=float) - target_center
    models, prediction_centers = [], []
    for index, seed in enumerate(SEEDS):
        model = CatBoostRegressor(**parameters(args, seed))
        model.fit(
            train_x, target, sample_weight=sample_weight,
            cat_features=list(F_CATEGORICAL),
        )
        source = model.predict(train_x)
        center = float(
            np.mean(source) if sample_weight is None
            else np.average(source, weights=sample_weight)
        )
        models.append(model)
        prediction_centers.append(center)
        if save:
            model.save_model(
                str(ROOT / "submit/model" / f"catboost_v64_f_residual_{index}.cbm")
            )
    return models, prediction_centers


def model_correction(
    models: list[CatBoostRegressor],
    centers: list[float],
    features: pd.DataFrame,
) -> np.ndarray:
    return np.mean([
        model.predict(features) - center
        for model, center in zip(models, centers)
    ], axis=0)


def json_series(series: pd.Series) -> dict[str, float | str]:
    return {str(key): value.item() if hasattr(value, "item") else value
            for key, value in series.items()}


def score_gain(target: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> float:
    return float(bss(target, np.clip(candidate, *CLIP)) - bss(target, base))


def main() -> None:
    args = arguments()
    raw = pd.read_csv(
        ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False,
    )
    missing_pitcher_rate = raw["asof_pitcher_success_rate"].isna()
    if not raw.loc[missing_pitcher_rate, "asof_pitcher_n"].eq(0).all():
        raise ValueError("positive-count pitcher has a missing success rate")
    raw["asof_pitcher_success_rate"] = raw[
        "asof_pitcher_success_rate"
    ].fillna(0.0)
    with np.load(ROOT / "outputs/v61_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    seasons = oof["season"].astype(int)
    target = oof["target"].astype(float)
    base = oof["blended"].astype(float)
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(base) or not np.array_equal(
        rows["season"].to_numpy(int), seasons,
    ):
        raise ValueError("v61 OOF and train.csv are not aligned")

    # Strict 2023 -> 2024 F residual confirmation and OOF correction.
    active23, active24 = seasons == 2023, seasons == 2024
    rows23 = rows.loc[active23].reset_index(drop=True)
    rows24 = rows.loc[active24].reset_index(drop=True)
    f23 = rows23["game_type"].eq("F").to_numpy()
    f24 = rows24["game_type"].eq("F").to_numpy()
    x23 = build_f_features(
        rows23, base[active23], prior_game_type_table(raw, 2023).to_dict(),
    ).loc[f23].reset_index(drop=True)
    x24 = build_f_features(
        rows24, base[active24], prior_game_type_table(raw, 2024).to_dict(),
    ).loc[f24].reset_index(drop=True)
    validation_models, validation_centers = fit_members(
        x23, target[active23][f23] - base[active23][f23], args,
    )
    f24_correction = F_SCALE * model_correction(
        validation_models, validation_centers, x24,
    )

    states, league_rates = season_states(raw)
    corrected = base.copy()
    fold_audits = {}
    for year in (2023, 2024):
        active = seasons == year
        year_rows = rows.loc[active].reset_index(drop=True)
        deltas, audit = dynamic_deltas(
            year_rows, year, states, league_rates, career_before(raw, year),
        )
        regular = year_rows["game_type"].eq("R").to_numpy()
        r_correction = R_WEIGHT * deltas[f"ar_k{int(R_STRENGTH)}"]
        block = corrected[active].copy()
        block[regular] += r_correction[regular]
        if year == 2024:
            block[f24] += f24_correction
        corrected[active] = np.clip(block, *CLIP)
        fold_audits[str(year)] = {
            "rho": audit["rho"],
            "total_gain": score_gain(target[active], base[active], corrected[active]),
            "r_dynamic_gain": score_gain(
                target[active][regular], base[active][regular],
                np.clip(base[active][regular] + r_correction[regular], *CLIP),
            ),
            "f_residual_gain": (
                score_gain(
                    target[active][f24], base[active][f24],
                    np.clip(base[active][f24] + f24_correction, *CLIP),
                ) if year == 2024 else None
            ),
        }
    if not np.isfinite(corrected).all():
        raise ValueError("v64 OOF correction contains non-finite values")

    # Production F residual: both strict OOF years, with the newer year weighted
    # twice as much.  The residual level is removed before training.
    feature_blocks = []
    f_positions = []
    for year in (2023, 2024):
        active = seasons == year
        year_rows = rows.loc[active].reset_index(drop=True)
        feature_blocks.append(build_f_features(
            year_rows, base[active], prior_game_type_table(raw, year).to_dict(),
        ))
        f_positions.append(year_rows["game_type"].eq("F").to_numpy())
    all_features = pd.concat(feature_blocks, ignore_index=True)
    f_all = np.concatenate(f_positions)
    production_x = all_features.loc[f_all].reset_index(drop=True)
    production_residual = target[f_all] - base[f_all]
    production_weights = np.where(seasons[f_all] == 2024, 1.0, 0.5)
    _, production_centers = fit_members(
        production_x, production_residual, args,
        sample_weight=production_weights, save=True,
    )

    # Freeze the 2025 dynamic state.  These maps come only from 2019-2024.
    rho, transition_pairs = fit_ar1(states, 2025)
    latest = latest_state(states, 2025)
    career = career_before(raw, 2025)
    prior_type = prior_game_type_table(raw, 2025)
    dynamic_configuration = {
        "prediction_year": 2025,
        "league_prior": float(league_rates[2024]),
        "rho": rho,
        "transition_pairs": transition_pairs,
        "current_prior_strength": R_STRENGTH,
        "weight": R_WEIGHT,
        "game_type_gate": "R",
        "prior_n": json_series(career.n),
        "prior_success": json_series(career.successes),
        "latent": json_series(latest["latent"]),
        "latent_year": json_series(latest["season"]),
    }

    np.savez_compressed(
        ROOT / "outputs/v64_oof_predictions.npz",
        **{key: value for key, value in oof.items() if key != "blended"},
        blended=corrected,
    )
    metadata_path = ROOT / "submit/model/metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "v61_public_complete_shape" not in metadata.get("model_names", []):
        raise ValueError("v64 requires the complete v61 bundle")
    rejected = {
        "v62_public_residual_frontier", "v63_train_trend_calibration", VERSION,
    }
    names = [name for name in metadata["model_names"] if name not in rejected]
    metadata["model_names"] = names + [VERSION]
    metadata.pop("v62_public_residual_frontier", None)
    metadata.pop("v63_train_trend_calibration", None)
    metadata["version"] = VERSION
    metadata[VERSION] = {
        "baseline": "v61_public_complete_shape",
        "f_residual": {
            "model_files": [
                f"catboost_v64_f_residual_{index}.cbm" for index in range(len(SEEDS))
            ],
            "model_prediction_centers": production_centers,
            "scale": F_SCALE,
            "categorical_columns": list(F_CATEGORICAL),
            "feature_columns": list(production_x.columns),
            "prior_game_type": {
                str(key): str(value) for key, value in prior_type.items()
            },
            "source_year_weights": {"2023": 0.5, "2024": 1.0},
            "source_residual_level_removed": True,
        },
        "r_dynamic_state": dynamic_configuration,
        "validation": fold_audits,
        "row_independent_inference": True,
        "external_model_or_prediction_in_bundle": False,
        "forbidden_2025_trackman_used": False,
        "test_row_aggregation_used": False,
        "v62_or_v63_component_used": False,
        "projected_public_range": [1132.0, 1139.0],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    promotion = {
        "version": VERSION,
        "baseline": "v61_public_complete_shape",
        "f_scale": F_SCALE,
        "r_strength": R_STRENGTH,
        "r_weight": R_WEIGHT,
        "folds": fold_audits,
        "production_f_rows": int(f_all.sum()),
        "rho_2025": rho,
        "projected_public_range": [1132.0, 1139.0],
        "rules": {
            "external_model_or_prediction_used": False,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    path = ROOT / "research/v64_promotion.json"
    path.write_text(json.dumps(promotion, indent=2), encoding="utf-8")
    print(json.dumps(promotion, indent=2), flush=True)


if __name__ == "__main__":
    main()
