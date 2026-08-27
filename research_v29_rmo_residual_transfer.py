"""Test whether train-only R/M/O meta predictions improve v23 residual transfer.

This is deliberately stricter than a pooled random split: the final decision is
based on a corrector trained on 2023 OOF residuals and transferred unchanged to
2024.  Pitcher-disjoint 2023 cross-fitting is reported only as a secondary
diagnostic.  A no-meta ExtraTrees corrector is evaluated beside the R/M/O
version so gains can be attributed to the auxiliary task rather than capacity.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesRegressor

from feature_engineering import (
    TARGET_COL,
    add_state_interactions,
    add_training_component_features,
    engineer_features,
    training_history_arrays,
)
from research_inferred_pitch_priors import bss


ROOT = Path(__file__).resolve().parent
# One deterministic member is enough for the initial transfer screen.  A
# promoted candidate is re-fit as a multi-seed ensemble in the packaging step.
SEEDS = (8101,)
warnings.filterwarnings("ignore", category=PerformanceWarning)


def segment(rows: pd.DataFrame) -> np.ndarray:
    hybrid = rows["pitcher_team_id"].eq(13) | rows["batter_team_id"].eq(13)
    return np.where(rows["game_type"].eq("F"), 2,
                    np.where(hybrid, 1, 0)).astype(np.int8)


def parameters(seed):
    return dict(
        n_estimators=120, max_depth=10, min_samples_leaf=200,
        max_features=.70, bootstrap=False, n_jobs=-1, random_state=seed,
    )


def fit_segmented(x_train, residual, train_segment, x_valid, valid_segment,
                  seed):
    prediction = np.zeros(len(x_valid), dtype=np.float64)
    for value in (0, 1, 2):
        train = train_segment == value
        valid = valid_segment == value
        model = ExtraTreesRegressor(**parameters(seed + value * 100))
        model.fit(x_train[train], residual[train])
        prediction[valid] = model.predict(x_valid[valid])
    return prediction


def masks(rows: pd.DataFrame):
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


def audit(target, base, direction, rows):
    result = []
    groups = masks(rows)
    for scale in np.arange(-.20, 1.001, .025):
        candidate = np.clip(base + scale * direction, .005, .995)
        gains = {
            name: bss(target[active], candidate[active])
            - bss(target[active], base[active])
            for name, active in groups.items() if active.any()
        }
        result.append({
            "scale": float(scale), "gains": gains,
            "min_quarter": min(gains[f"q{i}"] for i in range(1, 5)),
            "min_half": min(gains["first_half"], gains["second_half"]),
        })
    result.sort(key=lambda row: (
        min(row["min_quarter"], row["min_half"]), row["gains"]["all"],
    ), reverse=True)
    return result


def main():
    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig",
                      low_memory=False)
    target_series = raw.pop(TARGET_COL).astype(np.float32)
    history = training_history_arrays(raw, target_series)
    features = engineer_features(
        raw, *history, global_prior=float(target_series.mean()),
    )
    add_training_component_features(features, raw)
    features = add_state_interactions(features).copy()
    # Year is constant within each side of the transfer and cannot extrapolate.
    features.drop(columns=["season"], inplace=True)
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    matrix = np.nan_to_num(features.to_numpy(np.float32), nan=-999.,
                           posinf=999., neginf=-999.)

    with np.load(ROOT / "outputs/v23_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    years = {}
    for year in (2023, 2024):
        raw_active = raw["season"].eq(year).to_numpy()
        oof_active = oof["season"] == year
        rows = raw.loc[raw_active].reset_index(drop=True)
        target = oof["target"][oof_active].astype(float)
        if not np.allclose(target, target_series.to_numpy()[raw_active]):
            raise ValueError(f"v23 OOF rows do not align for {year}")
        with np.load(ROOT / f"research/v29_rmo_multitask_{year}.npz") as z:
            meta = z["meta"].astype(np.float32)
            names = z["names"].astype(str)
        years[year] = {
            "rows": rows,
            "target": target,
            "base": oof["blended"][oof_active].astype(float),
            "x": matrix[raw_active],
            "meta": meta,
            "names": names,
            "segment": segment(rows),
        }

    variants = {
        "plain": (),
        "cat_rmo": tuple(range(10)),
        "shape_only": tuple(range(1, 6)) + tuple(range(7, 10)),
    }
    reports = {}
    stored = {}
    for variant, columns in variants.items():
        data = {}
        for year in (2023, 2024):
            base_column = years[year]["base"].astype(np.float32)[:, None]
            pieces = [years[year]["x"], base_column]
            if columns:
                pieces.append(years[year]["meta"][:, columns])
            data[year] = np.column_stack(pieces).astype(np.float32)

        transfer_members = []
        for seed in SEEDS:
            transfer_members.append(fit_segmented(
                data[2023], years[2023]["target"] - years[2023]["base"],
                years[2023]["segment"], data[2024], years[2024]["segment"],
                seed,
            ))
        transfer = np.mean(transfer_members, axis=0)

        # Same-year pitcher-disjoint cross-fit is a secondary capacity check.
        crossfit_members = []
        pitcher = years[2023]["rows"]["pitcher_id"].to_numpy()
        unique = np.unique(pitcher)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            shuffled = unique.copy()
            rng.shuffle(shuffled)
            fold_map = {value: index % 3 for index, value in enumerate(shuffled)}
            fold = np.asarray([fold_map[value] for value in pitcher], np.int8)
            correction = np.zeros(len(pitcher), dtype=float)
            for heldout in range(3):
                train = fold != heldout
                valid = ~train
                correction[valid] = fit_segmented(
                    data[2023][train],
                    (years[2023]["target"] - years[2023]["base"])[train],
                    years[2023]["segment"][train], data[2023][valid],
                    years[2023]["segment"][valid], seed + heldout * 1000,
                )
            crossfit_members.append(correction)
        crossfit = np.mean(crossfit_members, axis=0)

        reports[variant] = {
            "crossfit_2023": audit(
                years[2023]["target"], years[2023]["base"], crossfit,
                years[2023]["rows"],
            )[:20],
            "transfer_2024": audit(
                years[2024]["target"], years[2024]["base"], transfer,
                years[2024]["rows"],
            )[:20],
        }
        stored[f"{variant}_crossfit_2023"] = crossfit.astype(np.float32)
        stored[f"{variant}_transfer_2024"] = transfer.astype(np.float32)
        print(json.dumps({variant: reports[variant]}, indent=2), flush=True)

    output = ROOT / "research/v29_rmo_residual_transfer.npz"
    np.savez_compressed(
        output, reports_json=np.asarray(json.dumps(reports)), **stored,
    )
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
