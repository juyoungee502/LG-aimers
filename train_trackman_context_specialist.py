"""Train the v17 regular-season Trackman context specialist."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import bss
from trackman_context import (
    FEATURE_COLUMNS, attach_context, freeze_context, pitcher_mapping,
    prepare_trackman,
)


SEEDS = (42, 2025, 3407)
BLEND_WEIGHT = .35
CAT_COLUMNS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "count_state", "hand_matchup", "team_matchup",
]


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def parameters(seed):
    return dict(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function="Logloss",
        eval_metric="Logloss", task_type="GPU", devices="0",
        random_seed=seed, allow_writing_files=False, verbose=0,
    )


def fit_predict(features, target, train_mask, valid_mask, seeds, save_dir=None):
    predictions = []
    for index, seed in enumerate(seeds):
        model = CatBoostClassifier(**parameters(seed))
        model.fit(features.loc[train_mask], target[train_mask])
        if valid_mask is not None:
            predictions.append(model.predict_proba(features.loc[valid_mask])[:, 1])
        if save_dir is not None:
            model.save_model(save_dir / f"catboost_trackman_context_{index}.cbm")
        print(f"Trackman context model complete: seed={seed}", flush=True)
    return np.mean(predictions, axis=0) if predictions else None


def main():
    root = Path(__file__).resolve().parent
    metadata_path = root / "submit" / "model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v16_pitch_failure_prior":
        raise ValueError(f"Expected v16 artifacts, found {metadata.get('version')}")
    data = pd.read_csv(root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    trackman = pd.read_csv(
        root / "data" / "trackman_history.csv",
        usecols=[
            "trackman_id", "season", "pitcher_trackman_id", "pitch_type_group",
            "balls_before", "strikes_before", "batter_hand", "rel_speed",
        ], encoding="utf-8-sig", low_memory=False,
    )
    mapping, mapping_report = pitcher_mapping(root, data, trackman)
    trackman = prepare_trackman(trackman, mapping)
    context = attach_context(data, trackman)
    bases = training_history_arrays(data, target_series)
    features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(features, data)
    features = add_state_interactions(features)
    features = pd.concat([features, context], axis=1)
    for column in CAT_COLUMNS:
        features[column] = features[column].fillna(-1).astype(np.int32)
    expected_base = metadata["feature_columns"]
    expected = [*expected_base, *FEATURE_COLUMNS]
    if list(features.columns) != expected:
        raise ValueError("Trackman specialist feature schema differs from metadata")
    seasons = data["season"].to_numpy(np.int16)
    with np.load(root / "outputs" / "v16_oof_predictions.npz") as loaded:
        v16 = {key: loaded[key] for key in loaded.files}
    oof_predictions, reports = [], {}
    for year in (2023, 2024):
        train_mask = seasons < year
        valid_mask = seasons == year
        specialist = fit_predict(
            features, target, train_mask, valid_mask, SEEDS,
        )
        base_mask = v16["season"] == year
        y = v16["target"][base_mask].astype(float)
        base = v16["blended"][base_mask].astype(float)
        if not np.allclose(y, target[valid_mask]):
            raise ValueError(f"v16 OOF rows do not align for {year}")
        rows = data.loc[valid_mask]
        regular = rows["game_type"].eq("R").to_numpy()
        blended_logit = logit(base)
        blended_logit[regular] = (
            (1.0 - BLEND_WEIGHT) * blended_logit[regular]
            + BLEND_WEIGHT * logit(specialist[regular])
        )
        blended = sigmoid(blended_logit)
        half = len(y) // 2
        report = {
            "v16_bss": bss(y, base), "v17_bss": bss(y, blended),
            "gain": bss(y, blended) - bss(y, base),
            "gain_first_half": (
                bss(y[:half], blended[:half]) - bss(y[:half], base[:half])
            ),
            "gain_second_half": (
                bss(y[half:], blended[half:]) - bss(y[half:], base[half:])
            ),
            "regular_rows": int(regular.sum()),
        }
        reports[str(year)] = report
        oof_predictions.append((year, specialist, blended))
        print(f"v17 Trackman validation {year}: {json.dumps(report)}", flush=True)

    model_dir = root / "submit" / "model"
    fit_predict(
        features, target, np.ones(len(data), dtype=bool), None, SEEDS,
        save_dir=model_dir,
    )
    final_context = freeze_context(trackman, int(data["season"].max()) + 1)
    final_context.update({
        "blend_weight": BLEND_WEIGHT, "game_type": "R",
        "model_feature_columns": expected,
    })
    metadata["version"] = "v17_trackman_context"
    model_names = metadata.setdefault("model_names", [])
    if "trackman_context_specialist" not in model_names:
        model_names.append("trackman_context_specialist")
    metadata["trackman_context"] = final_context
    metadata["training_info"]["v17_validation"] = {
        **reports, "seeds": list(SEEDS), "numeric_id_encoding": True,
        "mapped_pitchers": len(mapping),
        "minimum_mapping_confidence": float(mapping_report["confidence"].min()),
        "aligned_pitch_rows": 1_160_395,
        "row_independent_inference": True,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8",
    )
    specialist_all = np.concatenate([item[1] for item in oof_predictions])
    blended_all = np.concatenate([item[2] for item in oof_predictions])
    np.savez_compressed(
        root / "outputs" / "v17_oof_predictions.npz",
        **{key: value for key, value in v16.items() if key != "blended"},
        trackman_context=specialist_all.astype(np.float32),
        blended=blended_all.astype(np.float64),
    )
    print(
        f"Stored Trackman lookups: count={len(final_context['count_keys'])}, "
        f"hand={len(final_context['hand_keys'])}", flush=True,
    )


if __name__ == "__main__":
    main()
