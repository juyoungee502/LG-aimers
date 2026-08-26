"""Train decomposed, prior-context failure experts for v19."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from failure_context import freeze_prior_context, prior_season_context
from feature_engineering import (
    TARGET_COL, add_state_interactions, add_training_component_features,
    engineer_features, training_history_arrays,
)
from research_inferred_pitch_priors import reconstruct_labels


LABELS = ("reverse", "middle", "wayoff")
WEIGHT_ALL = .20
WEIGHT_MIDDLE = .015


def bss(target, prediction):
    rate = float(target.mean())
    return 100000. * (
        1. - np.mean((target - np.clip(prediction, .005, .995)) ** 2)
        / (rate * (1. - rate))
    )


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


def parameters(seed):
    return dict(
        iterations=1600, learning_rate=.01631820635235777, depth=8,
        l2_leaf_reg=509.6419153575998, random_strength=2.9151912613602535,
        bagging_temperature=.36881602504480515, border_count=32,
        bootstrap_type="Bayesian", loss_function="Logloss", eval_metric="Logloss",
        task_type="GPU", devices="0", random_seed=seed,
        allow_writing_files=False, verbose=0,
    )


def combine(base, prediction_matrix, regular):
    p_all = np.clip(1. - prediction_matrix[:, 0] - prediction_matrix[:, 1]
                    - prediction_matrix[:, 2], 1e-5, 1. - 1e-5)
    p_middle = np.clip(1. - prediction_matrix[:, 1], 1e-5, 1. - 1e-5)
    output = base.copy()
    output[regular] = sigmoid(
        (1. - WEIGHT_ALL - WEIGHT_MIDDLE) * logit(base[regular])
        + WEIGHT_ALL * logit(p_all[regular])
        + WEIGHT_MIDDLE * logit(p_middle[regular])
    )
    return output


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    target_series = data.pop(TARGET_COL).astype(np.float32)
    target = target_series.to_numpy()
    full = pd.concat([data, target_series.rename(TARGET_COL)], axis=1)
    labels = reconstruct_labels(full)
    bases = training_history_arrays(data, target_series)
    base_features = engineer_features(data, *bases, global_prior=float(target.mean()))
    add_training_component_features(base_features, data)
    base_features = add_state_interactions(base_features)
    context = prior_season_context(full, labels)
    features = pd.concat([base_features, context], axis=1)

    model_dir = root / "submit/model"
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("version") != "v18_f_regime":
        raise ValueError(f"Expected v18 metadata, got {metadata.get('version')}")
    for column in metadata["cat_features"]:
        features[column] = features[column].fillna(-1).astype(np.int32)
    if list(base_features.columns) != metadata["feature_columns"]:
        raise ValueError("Failure specialist base features differ from v18")

    with np.load(root / "outputs/v18_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    upgraded = oof["blended"].astype(np.float64).copy()
    reports = {}
    for year in (2023, 2024):
        with np.load(root / f"research/failure_specialists_{year}_prior_context.npz") as loaded:
            stored = {key: loaded[key] for key in loaded.files}
        index = list(stored["variants"].astype(str)).index("uniform_depth8")
        matrix = stored["predictions"][index, :, :3].astype(np.float64)
        fold = oof["season"] == year
        if not np.allclose(stored["target"], oof["target"][fold]):
            raise ValueError(f"Failure OOF rows differ for {year}")
        rows = data.loc[data["season"].eq(year)]
        regular = rows["game_type"].eq("R").to_numpy()
        base = upgraded[fold].copy()
        prediction = combine(base, matrix, regular)
        y = oof["target"][fold].astype(np.float64)
        midpoint = len(y) // 2
        report = {
            "base_bss": bss(y, base), "v19_bss": bss(y, prediction),
            "gain": bss(y, prediction) - bss(y, base),
            "gain_first_half": bss(y[:midpoint], prediction[:midpoint]) - bss(y[:midpoint], base[:midpoint]),
            "gain_second_half": bss(y[midpoint:], prediction[midpoint:]) - bss(y[midpoint:], base[midpoint:]),
        }
        if min(report["gain"], report["gain_first_half"], report["gain_second_half"]) <= 0.:
            raise RuntimeError(f"Failure specialist failed promotion for {year}: {report}")
        reports[str(year)] = report
        upgraded[fold] = prediction
    print(f"v19 validation: {json.dumps(reports)}", flush=True)

    usable = labels[list(LABELS)].notna().all(axis=1).to_numpy()
    for offset, label in enumerate(LABELS):
        model = CatBoostClassifier(**parameters(250 + offset))
        model.fit(features.loc[usable], labels.loc[usable, label].to_numpy(np.int8))
        model.save_model(str(model_dir / f"catboost_failure_{label}.cbm"))
        print(f"Failure model complete: {label}", flush=True)

    frozen = freeze_prior_context(full, labels, int(data["season"].max()) + 1)
    frozen.update({
        "weight_all": WEIGHT_ALL, "weight_middle": WEIGHT_MIDDLE,
        "game_type": "R", "labels": list(LABELS),
        "model_feature_columns": list(features.columns),
    })
    metadata["version"] = "v19_failure_specialist"
    names = metadata.setdefault("model_names", [])
    if "failure_specialist" not in names:
        names.append("failure_specialist")
    metadata["failure_specialist"] = frozen
    metadata["training_info"]["v19_validation"] = reports
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

    output = root / "outputs/v19_oof_predictions.npz"
    np.savez_compressed(
        output, **{key: value for key, value in oof.items() if key != "blended"},
        blended=upgraded,
    )
    print(f"Stored v19 models and diagnostics: {output}", flush=True)


if __name__ == "__main__":
    main()
