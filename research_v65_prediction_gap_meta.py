"""Audit a low-dimensional prediction-gap residual over v64.

This independently implements the public anchor-plus-small-residual concept.
Only this project's own forward OOF predictions are meta inputs.  v62/v63 are
excluded explicitly.  Ridge regularisation prevents the sidecar from replacing
the trusted anchor with an unstable second-stage model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from research_v65_hierarchical_context_lookup import cluster_bootstrap, gain
from v65_prediction_gap import build_prediction_gap_features


ROOT = Path(__file__).resolve().parent
# These historical predictions correspond to exact checkpoints in the current
# inference graph.  v57/v58 are deliberately omitted because their superseded
# scaling configurations are not retained by the clean v61/v64 bundle.
# Keep only checkpoints whose inference state is still reproduced exactly by
# submit/script.py.  Later archives inherited now-removed temporal portfolio
# stages, so using them would make research and submission features diverge.
VERSIONS = ("v16", "v17", "v18", "v19", "v23", "v24")
RIDGES = (1000.0, 10000.0, 30000.0, 100000.0, 300000.0)
SCALES = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
CLIP = (0.005, 0.995)


def load_meta(
    base: np.ndarray, target: np.ndarray, season: np.ndarray, rows: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    stage_predictions: dict[str, np.ndarray] = {}
    lineage = []
    for version in VERSIONS:
        path = ROOT / f"outputs/{version}_oof_predictions.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=True) as archive:
            if not np.array_equal(archive["target"].astype(float), target):
                raise ValueError(f"{version} target alignment mismatch")
            if not np.array_equal(archive["season"].astype(int), season):
                raise ValueError(f"{version} season alignment mismatch")
            prediction = archive["blended"].astype(float)
        stage_predictions[version] = prediction
        lineage.append(version)
    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        members = archive["predictions"].astype(float)
        names = archive["model_names"].astype(str).tolist()
        member_predictions = {
            name: members[:, index] for index, name in enumerate(names)
        }
        features = build_prediction_gap_features(
            rows, base, member_predictions, stage_predictions,
            archive["base_blended"].astype(float),
            archive["trackman_context"].astype(float),
            archive["f_specialist"].astype(float),
            member_names=names, stage_names=lineage,
        )
    return features, lineage


def fit_predict(
    features: pd.DataFrame,
    fit_index: np.ndarray,
    valid_index: np.ndarray,
    residual: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train = scaler.fit_transform(features.iloc[fit_index])
    valid = scaler.transform(features.iloc[valid_index])
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(train, residual)
    return model.predict(valid)


def centered_residual(y: np.ndarray, base: np.ndarray, game: np.ndarray) -> np.ndarray:
    residual = y - base
    output = residual.copy()
    for group in ("R", "F"):
        active = game == group
        output[active] -= float(residual[active].mean())
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        y = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    raw = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=[
            "season", "game_type", "pitcher_id", "control_success",
            "asof_pitcher_success_rate", "asof_batter_success_rate",
            "asof_pitcher_n", "asof_batter_n", "balls_before", "strikes_before",
        ], encoding="utf-8-sig", low_memory=False,
    )
    rows = pd.concat([raw.loc[raw["season"].eq(year)] for year in (2023, 2024)], ignore_index=True)
    if len(rows) != len(y) or not np.array_equal(rows["control_success"].to_numpy(float), y):
        raise ValueError("v64 OOF rows are not aligned")
    features, lineage = load_meta(base, y, season, rows)
    game = rows["game_type"].astype(str).to_numpy()
    positions23 = np.flatnonzero(season == 2023)
    positions24 = np.flatnonzero(season == 2024)
    first23, second23 = np.array_split(positions23, 2)
    folds = [
        ("2023_second_half", first23, second23),
        ("2024_forward", positions23, positions24),
    ]
    corrections: dict[tuple[float, str], np.ndarray] = {}
    for alpha in RIDGES:
        for name, fit_index, valid_index in folds:
            residual = centered_residual(y[fit_index], base[fit_index], game[fit_index])
            corrections[(alpha, name)] = fit_predict(
                features, fit_index, valid_index, residual, alpha,
            )
    candidates = []
    for alpha in RIDGES:
        for r_scale in SCALES:
            for f_scale in SCALES:
                evaluations = []
                for name, _, valid_index in folds:
                    regular = game[valid_index] == "R"
                    scale = np.where(regular, r_scale, f_scale)
                    prediction = np.clip(
                        base[valid_index] + scale * corrections[(alpha, name)], *CLIP,
                    )
                    halves = np.array_split(np.arange(len(valid_index)), 2)
                    groups = {
                        label: gain(
                            y[valid_index][mask], base[valid_index][mask], prediction[mask],
                        ) for label, mask in (("R", regular), ("F", ~regular))
                    }
                    evaluations.append({
                        "fold": name,
                        "gain": gain(y[valid_index], base[valid_index], prediction),
                        "half_gains": [
                            gain(y[valid_index][i], base[valid_index][i], prediction[i])
                            for i in halves
                        ],
                        "group_gains": groups,
                        "mean_absolute_change": float(np.mean(np.abs(prediction - base[valid_index]))),
                    })
                preliminary = bool(
                    min(item["gain"] for item in evaluations) > 0.0
                    and min(v for item in evaluations for v in item["half_gains"]) >= 0.0
                    and min(v for item in evaluations for v in item["group_gains"].values()) >= 0.0
                )
                gains = [item["gain"] for item in evaluations]
                candidates.append({
                    "ridge": alpha, "r_scale": r_scale, "f_scale": f_scale,
                    "evaluations": evaluations, "preliminary_gate": preliminary,
                    "min_gain": float(min(gains)), "mean_gain": float(np.mean(gains)),
                })
    candidates.sort(
        key=lambda item: (item["preliminary_gate"], item["min_gain"], item["mean_gain"]),
        reverse=True,
    )
    def bootstrap_candidate(candidate: dict[str, object], repetitions: int) -> dict[str, object]:
        result = {}
        for name, _, valid_index in folds:
            regular = game[valid_index] == "R"
            scale = np.where(regular, candidate["r_scale"], candidate["f_scale"])
            prediction = np.clip(
                base[valid_index] + scale * corrections[(candidate["ridge"], name)], *CLIP,
            )
            result[name] = cluster_bootstrap(
                y[valid_index], base[valid_index], prediction,
                rows.iloc[valid_index]["pitcher_id"].to_numpy(), repetitions,
                651150 + int(rows.iloc[valid_index]["season"].iloc[-1]),
            )
        return result

    # The best mean-gain point need not have the tightest uncertainty. Screen
    # the leading temporally stable settings, then rerun the selected setting
    # with the full bootstrap count.
    screened = []
    for candidate in [item for item in candidates if item["preliminary_gate"]]:
        screening_bootstrap = bootstrap_candidate(candidate, min(1000, args.bootstrap))
        screened.append({
            "candidate": candidate,
            "bootstrap": screening_bootstrap,
            "min_ci_low": float(min(item["ci_low"] for item in screening_bootstrap.values())),
        })
    screened.sort(
        key=lambda item: (
            item["min_ci_low"] > 0.0,
            item["min_ci_low"],
            item["candidate"]["min_gain"],
        ), reverse=True,
    )
    best = screened[0]["candidate"] if screened else candidates[0]
    bootstraps = bootstrap_candidate(best, args.bootstrap)
    strict = bool(
        best["preliminary_gate"]
        and min(item["ci_low"] for item in bootstraps.values()) > 0.0
    )
    report = {
        "baseline": "v64_public_method_transfer",
        "method": "low-dimensional prediction-gap Ridge residual",
        "oof_lineage": lineage,
        "excluded_lineage": ["v62", "v63"],
        "feature_count": int(features.shape[1]),
        "best": best,
        "bootstrap": bootstraps,
        "bootstrap_screen": screened[:20],
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
    path = ROOT / "research/v65_prediction_gap_meta.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
