"""Test a low-capacity hazard/disagreement residual gate over v23.

The source gate is fitted only on 2023 OOF residuals and is evaluated on 2024.
Its features are row-local predictions already produced by the submission plus
official pre-pitch state.  Recovered failure labels are never inference inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from research_inferred_pitch_priors import bss


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def hazard_matrix(root, year):
    with np.load(
        root / "research" / f"failure_specialists_{year}_prior_context.npz"
    ) as loaded:
        variants = list(loaded["variants"].astype(str))
        index = variants.index("uniform_depth8")
        hazards = loaded["predictions"][index, :, :3].astype(np.float64)
    hazards = np.clip(hazards, 1e-5, 1. - 1e-5)
    return hazards


def meta_features(raw, oof, mask, hazards, year):
    component = oof["predictions"][mask].astype(np.float64)
    component_names = list(oof["model_names"].astype(str))
    output = pd.DataFrame(component, columns=[f"model_{name}" for name in component_names])
    output["base_blended"] = oof["base_blended"][mask].astype(float)
    output["trackman_context"] = oof["trackman_context"][mask].astype(float)
    output["model_mean"] = component.mean(axis=1)
    output["model_std"] = component.std(axis=1)
    output["model_range"] = component.max(axis=1) - component.min(axis=1)
    names = ("reverse", "middle", "wayoff")
    for index, name in enumerate(names):
        output[f"hazard_{name}"] = hazards[:, index]
        output[f"hazard_logit_{name}"] = logit(hazards[:, index])
    output["hazard_sum"] = hazards.sum(axis=1)
    output["hazard_max"] = hazards.max(axis=1)
    output["hazard_std"] = hazards.std(axis=1)
    output["hazard_log_middle_reverse"] = np.log(hazards[:, 1] / hazards[:, 0])
    output["hazard_log_wayoff_reverse"] = np.log(hazards[:, 2] / hazards[:, 0])
    output["hazard_log_wayoff_middle"] = np.log(hazards[:, 2] / hazards[:, 1])

    # OOF arrays only contain the held-out 2023/2024 folds, whereas ``raw``
    # contains every season.  Select raw rows by year instead of applying the
    # shorter OOF mask to the full frame.
    frame = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
    if len(frame) != len(output):
        raise ValueError(
            f"Raw/OOF alignment length differs for {year}: "
            f"raw={len(frame)} oof={len(output)}"
        )
    numeric = (
        "balls_before", "strikes_before", "outs_before", "inning",
        "num_runners_on", "li", "score_diff_pitcher_team",
        "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n",
        "asof_pitcher_success_rate", "asof_batter_success_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate",
    )
    for column in numeric:
        output[f"raw_{column}"] = pd.to_numeric(
            frame[column], errors="coerce",
        ).to_numpy(float)
    output["log_pitcher_n"] = np.log1p(frame["asof_pitcher_n"].fillna(0).clip(lower=0))
    output["log_batter_n"] = np.log1p(frame["asof_batter_n"].fillna(0).clip(lower=0))
    output["is_regular"] = frame["game_type"].eq("R").astype(np.int8).to_numpy()
    output["same_hand"] = frame["pitcher_hand"].eq(frame["batter_hand"]).astype(np.int8).to_numpy()
    output["count_state"] = (
        frame["balls_before"] * 3 + frame["strikes_before"]
    ).to_numpy(np.int8)
    return output.replace([np.inf, -np.inf], np.nan), frame


def segment_masks(rows):
    return {
        "all": np.ones(len(rows), dtype=bool),
        "first_half": np.arange(len(rows)) < len(rows) // 2,
        "second_half": np.arange(len(rows)) >= len(rows) // 2,
        "months_3_5": rows["game_month"].between(3, 5).to_numpy(),
        "months_6_7": rows["game_month"].between(6, 7).to_numpy(),
        "months_8_11": rows["game_month"].between(8, 11).to_numpy(),
        "regular": rows["game_type"].eq("R").to_numpy(),
        "futures": rows["game_type"].eq("F").to_numpy(),
    }


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    with np.load(root / "outputs/v23_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    folds = {}
    for year in (2023, 2024):
        mask = oof["season"] == year
        hazard = hazard_matrix(root, year)
        features, rows = meta_features(raw, oof, mask, hazard, year)
        folds[year] = {
            "features": features,
            "rows": rows,
            "target": oof["target"][mask].astype(np.float64),
            "base": oof["blended"][mask].astype(np.float64),
        }
        if not np.allclose(folds[year]["target"], rows["control_success"]):
            raise ValueError(f"OOF alignment differs for {year}")

    source = folds[2023]
    valid = folds[2024]
    predictions = {}
    variants = (
        ("d2_all", 2, False),
        ("d3_all", 3, False),
        ("d2_regular", 2, True),
        ("d3_regular", 3, True),
    )
    for offset, (name, depth, regular_only) in enumerate(variants):
        fit = np.ones(len(source["target"]), dtype=bool)
        if regular_only:
            fit &= source["rows"]["game_type"].eq("R").to_numpy()
        residual = source["target"] - source["base"]
        member = []
        centers = []
        for seed_offset in range(3):
            model = CatBoostRegressor(
                iterations=220, depth=depth, learning_rate=.025,
                loss_function="RMSE", eval_metric="RMSE", l2_leaf_reg=150.,
                random_strength=.2, bootstrap_type="Bernoulli", subsample=.8,
                border_count=32, task_type="GPU", devices="0",
                random_seed=9100 + 100 * offset + seed_offset,
                allow_writing_files=False, verbose=0,
            )
            model.fit(source["features"].loc[fit], residual[fit])
            member.append(model.predict(valid["features"]))
            centers.append(
                float(model.predict(source["features"].loc[fit]).mean())
            )
        correction = np.mean(member, axis=0)
        # Remove only the source prediction mean.  This is a frozen train-side
        # constant and does not inspect validation/test prediction distribution.
        center = float(np.mean(centers))
        correction -= center
        predictions[name] = correction
        print(
            f"Hazard meta gate complete: {name} rows={fit.sum()} center={center:.7f}",
            flush=True,
        )

    reports = []
    masks = segment_masks(valid["rows"])
    regular = valid["rows"]["game_type"].eq("R").to_numpy()
    for name, correction in predictions.items():
        for apply_gate in ("all", "regular"):
            value = correction.copy()
            if apply_gate == "regular":
                value[~regular] = 0.
            for weight in np.arange(-.25, 1.501, .025):
                candidate = np.clip(valid["base"] + weight * value, .005, .995)
                gains = {
                    label: bss(valid["target"][mask], candidate[mask])
                    - bss(valid["target"][mask], valid["base"][mask])
                    for label, mask in masks.items() if mask.any()
                }
                reports.append({
                    "name": name, "apply_gate": apply_gate,
                    "weight": float(weight), "gains": gains,
                    "min_half": min(gains["first_half"], gains["second_half"]),
                    "min_month": min(
                        gains["months_3_5"], gains["months_6_7"], gains["months_8_11"],
                    ),
                    "correction_mean": float(value.mean()),
                    "correction_std": float(value.std()),
                })
    reports.sort(
        key=lambda row: (
            min(row["min_half"], row["min_month"]), row["gains"]["all"],
        ), reverse=True,
    )
    output = root / "research/v23_hazard_meta_gate.npz"
    np.savez_compressed(
        output, names=np.asarray(list(predictions)),
        predictions=np.column_stack(list(predictions.values())).astype(np.float32),
        target=valid["target"].astype(np.float32),
        base=valid["base"].astype(np.float32),
        reports_json=np.asarray(json.dumps(reports)),
    )
    print(json.dumps({"top": reports[:80]}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
