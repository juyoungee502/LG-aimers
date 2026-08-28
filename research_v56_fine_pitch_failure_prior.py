"""Screen a fine-pitch failure prior without using the current pitch type.

Only aligned pitches from seasons before the validation year define (a) each
pitcher's failure tendency by historical tagged pitch type and (b) historical
pitch-selection probabilities by batter hand and count.  The current row's
actual pitch type is never read.  Signals are centered on the latest available
source season so they do not act as a hidden target-rate calibration.
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss, reconstruct_labels
from research_v40_failure_seed_stability import masks


ROOT = Path(__file__).resolve().parent
PITCH_TYPES = (
    "Fastball", "Slider", "Curveball", "ChangeUp", "Splitter",
    "Sinker", "Cutter", "Other",
)
FAILURES = ("reverse", "middle", "wayoff")
WEIGHT_GRID = (-1.5, -1., -.75, -.5, -.25, 0., .25, .5, .75, 1., 1.5)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome-k", default="20,50,100,200")
    parser.add_argument("--selection-k", default="100,300,600")
    return parser.parse_args()


def wide_counts(frame, keys):
    table = (
        frame.groupby([*keys, "pitch_type_fine"], observed=True, sort=False)
        .size().unstack(fill_value=0).reindex(columns=PITCH_TYPES, fill_value=0)
    )
    table.columns = [f"n_{name}" for name in PITCH_TYPES]
    return table.reset_index()


def fine_history(data, labels):
    with np.load(ROOT / "outputs/trackman_pitch_alignment.npz") as archive:
        links = pd.DataFrame({
            "row_id": archive["row_id"].astype(str),
            "trackman_id": archive["trackman_id"].astype(str),
        })
    trackman = pd.read_csv(
        ROOT / "data/trackman_history.csv", encoding="utf-8-sig",
        usecols=["trackman_id", "tagged_pitch_type"], low_memory=False,
    )
    trackman["trackman_id"] = trackman["trackman_id"].astype(str)
    replacements = {"Changeup": "ChangeUp", "Four-Seam": "Fastball", "SInker": "Sinker"}
    fine = trackman["tagged_pitch_type"].replace(replacements)
    trackman["pitch_type_fine"] = fine.where(fine.isin(PITCH_TYPES[:-1]), "Other")
    links = links.merge(
        trackman[["trackman_id", "pitch_type_fine"]], on="trackman_id",
        how="left", validate="one_to_one",
    )

    history = data[[
        "row_id", "season", "pitcher_id", "batter_hand",
        "balls_before", "strikes_before", "control_success",
    ]].copy()
    history["row_id"] = history["row_id"].astype(str)
    history["reverse"] = labels["reverse"].to_numpy(np.float32)
    history["middle"] = labels["middle"].to_numpy(np.float32)
    complete = (
        history["reverse"].isin((0., 1.))
        & history["middle"].isin((0., 1.))
    )
    history["wayoff"] = np.where(
        complete,
        history["control_success"].eq(0)
        & history["reverse"].eq(0) & history["middle"].eq(0),
        np.nan,
    ).astype(np.float32)
    history = history.merge(
        links[["row_id", "pitch_type_fine"]], on="row_id", how="inner",
        validate="one_to_one",
    ).dropna(subset=list(FAILURES))
    if not history[list(FAILURES)].isin((0., 1.)).all().all():
        raise ValueError("Reconstructed failure labels must be binary")
    history["count_state"] = (
        history["balls_before"] * 3 + history["strikes_before"]
    ).astype(np.int8)
    return history


def build_tables(source, label, outcome_k, selection_k):
    work = source.copy()
    work["relative"] = (
        work[label] - work.groupby("season", observed=True)[label].transform("mean")
    )
    grouped = work.groupby("pitch_type_fine", observed=True)["relative"].agg(["sum", "count"])
    global_outcome = {
        name: float(grouped.loc[name, "sum"] / grouped.loc[name, "count"])
        if name in grouped.index else 0. for name in PITCH_TYPES
    }
    outcome = work.groupby(
        ["pitcher_id", "pitch_type_fine"], observed=True, sort=False,
    )["relative"].agg(["sum", "count"]).reset_index()
    outcome["value"] = [
        (total + outcome_k * global_outcome[name]) / (count + outcome_k)
        for total, count, name in zip(
            outcome["sum"], outcome["count"], outcome["pitch_type_fine"],
        )
    ]
    outcome = outcome.pivot(
        index="pitcher_id", columns="pitch_type_fine", values="value",
    ).reindex(columns=PITCH_TYPES)
    outcome.columns = [f"v_{name}" for name in PITCH_TYPES]
    outcome = outcome.reset_index()

    overall = wide_counts(work, ["pitcher_id"])
    global_mix = work["pitch_type_fine"].value_counts(normalize=True)
    total = overall[[f"n_{name}" for name in PITCH_TYPES]].sum(axis=1).to_numpy(float)
    for name in PITCH_TYPES:
        overall[f"p_{name}"] = (
            overall[f"n_{name}"].to_numpy(float)
            + selection_k * float(global_mix.get(name, 0.))
        ) / (total + selection_k)

    keys = ["pitcher_id", "batter_hand", "count_state"]
    selection = wide_counts(work, keys)
    selection_total = selection[[f"n_{name}" for name in PITCH_TYPES]].sum(axis=1).to_numpy(float)
    selection = selection.merge(
        overall[["pitcher_id", *[f"p_{name}" for name in PITCH_TYPES]]],
        on="pitcher_id", how="left",
    )
    for name in PITCH_TYPES:
        fallback = selection[f"p_{name}"].fillna(float(global_mix.get(name, 0.)))
        selection[f"p_{name}"] = (
            selection[f"n_{name}"].to_numpy(float)
            + selection_k * fallback.to_numpy(float)
        ) / (selection_total + selection_k)
    return keys, selection, overall, outcome, global_mix, global_outcome


def apply_tables(rows, tables):
    keys, selection, overall, outcome, global_mix, global_outcome = tables
    query = rows[["pitcher_id", "batter_hand", "count_state"]].copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(
        selection[[*keys, *[f"p_{name}" for name in PITCH_TYPES]]],
        on=keys, how="left", sort=False,
    ).merge(outcome, on="pitcher_id", how="left", sort=False)
    base = overall[["pitcher_id", *[f"p_{name}" for name in PITCH_TYPES]]].merge(
        outcome, on="pitcher_id", how="left",
    )
    for name in PITCH_TYPES:
        base[f"p_{name}"] = base[f"p_{name}"].fillna(float(global_mix.get(name, 0.)))
        base[f"v_{name}"] = base[f"v_{name}"].fillna(global_outcome[name])
    base["base_expected"] = sum(
        base[f"p_{name}"] * base[f"v_{name}"] for name in PITCH_TYPES
    )
    query = query.merge(base[["pitcher_id", "base_expected"]], on="pitcher_id", how="left")
    for name in PITCH_TYPES:
        query[f"p_{name}"] = query[f"p_{name}"].fillna(float(global_mix.get(name, 0.)))
        query[f"v_{name}"] = query[f"v_{name}"].fillna(global_outcome[name])
    expected = sum(
        query[f"p_{name}"] * query[f"v_{name}"] for name in PITCH_TYPES
    ).to_numpy(float)
    baseline = query["base_expected"].fillna(0.).to_numpy(float)
    order = np.argsort(query["_order"].to_numpy())
    return (expected - baseline)[order]


def signal_matrix(history, data, year, outcome_k, selection_k):
    source = history.loc[history["season"].lt(year)].copy()
    query = data.loc[data["season"].eq(year)].copy().reset_index(drop=True)
    query["count_state"] = (
        query["balls_before"] * 3 + query["strikes_before"]
    ).astype(np.int8)
    reference = source.loc[source["season"].eq(source["season"].max())]
    values = []
    centers = []
    for label in FAILURES:
        tables = build_tables(source, label, outcome_k, selection_k)
        center = float(apply_tables(reference, tables).mean())
        values.append(apply_tables(query, tables) - center)
        centers.append(center)
    matrix = np.column_stack(values)
    matrix[query["game_type"].astype(str).ne("R").to_numpy()] = 0.
    return matrix, centers


def gain_quadratic(target, base, matrix, active, weights):
    y = target[active]
    p = base[active]
    x = matrix[active]
    residual = y - p
    linear = np.mean(x * residual[:, None], axis=0)
    quadratic = x.T @ x / len(x)
    denominator = float(y.mean() * (1. - y.mean()))
    return float(
        100000. * (2. * linear @ weights - weights @ quadratic @ weights)
        / denominator
    )


def cohort_masks(length, game_type):
    result = masks(length)
    result["R"] = game_type == "R"
    result["F"] = game_type == "F"
    return result


def main():
    args = arguments()
    outcome_values = tuple(float(value) for value in args.outcome_k.split(","))
    selection_values = tuple(float(value) for value in args.selection_k.split(","))
    data = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    labels = reconstruct_labels(data)
    history = fine_history(data, labels)
    with np.load(ROOT / "outputs/v24_oof_predictions.npz") as archive:
        v24 = {key: archive[key] for key in archive.files}
    with np.load(ROOT / "outputs/v54_oof_predictions.npz") as archive:
        v54 = {key: archive[key] for key in archive.files}

    fold = {}
    for year in (2023, 2024):
        active = v24["season"] == year
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        fold[year] = {
            "target": v24["target"][active].astype(float),
            "v24": np.clip(v24["blended"][active].astype(float), .005, .995),
            "v54": np.clip(v54["blended"][active].astype(float), .005, .995),
            "game_type": rows["game_type"].astype(str).to_numpy(),
        }
        if not np.allclose(fold[year]["target"], rows["control_success"]):
            raise ValueError(f"OOF rows do not align for {year}")

    reports = []
    matrices = {}
    centers = {}
    for outcome_k, selection_k in product(outcome_values, selection_values):
        config = f"o{outcome_k:g}_s{selection_k:g}"
        matrices[config] = {}
        centers[config] = {}
        for year in (2023, 2024):
            print(json.dumps({
                "config": config, "year": year,
                "aligned_source_rows": int((history["season"] < year).sum()),
            }), flush=True)
            matrix, center = signal_matrix(
                history, data, year, outcome_k, selection_k,
            )
            matrices[config][year] = matrix
            centers[config][year] = center

        for weight_tuple in product(WEIGHT_GRID, repeat=len(FAILURES)):
            weights = np.asarray(weight_tuple, float)
            gains = {}
            for year in (2023, 2024):
                item = fold[year]
                matrix = matrices[config][year]
                for base_name in ("v24", "v54"):
                    for cohort, active in cohort_masks(
                        len(item["target"]), item["game_type"],
                    ).items():
                        gains[f"{year}_{base_name}_{cohort}"] = gain_quadratic(
                            item["target"], item[base_name], matrix, active, weights,
                        )
            robust_keys = [
                f"{year}_v24_{cohort}"
                for year in (2023, 2024)
                for cohort in ("all", "h1", "h2", "q1", "q2", "q3", "q4", "R")
            ]
            reports.append({
                "config": config,
                "outcome_k": outcome_k, "selection_k": selection_k,
                "weights": dict(zip(FAILURES, map(float, weights))),
                "gains": gains,
                "minimum_crossyear_v24": float(min(gains[key] for key in robust_keys)),
                "mean_crossyear_v24": float(np.mean([gains[key] for key in robust_keys])),
                "gain_2024_v54": gains["2024_v54_all"],
                "minimum_quarter_2024_v54": float(min(
                    gains[f"2024_v54_q{index}"] for index in range(1, 5)
                )),
            })

    reports.sort(key=lambda row: (
        row["minimum_crossyear_v24"], row["minimum_quarter_2024_v54"],
        row["gain_2024_v54"], row["mean_crossyear_v24"],
    ), reverse=True)
    chosen = reports[0]
    config = chosen["config"]
    weights = np.asarray([chosen["weights"][name] for name in FAILURES])
    matrix = matrices[config][2024]
    base = fold[2024]["v54"]
    prediction = np.clip(base + matrix @ weights, .005, .995)
    exact_gain = float(bss(fold[2024]["target"], prediction) - bss(fold[2024]["target"], base))
    exact_gains = {}
    for year in (2023, 2024):
        item = fold[year]
        adjustment = matrices[config][year] @ weights
        for base_name in ("v24", "v54"):
            candidate = np.clip(item[base_name] + adjustment, .005, .995)
            for cohort, active in cohort_masks(
                len(item["target"]), item["game_type"],
            ).items():
                exact_gains[f"{year}_{base_name}_{cohort}"] = float(
                    bss(item["target"][active], candidate[active])
                    - bss(item["target"][active], item[base_name][active])
                )
    diagnostics = {
        "aligned_rows": int(len(history)),
        "alignment_coverage_train": float(len(history) / len(data)),
        "pitch_types": list(PITCH_TYPES), "failures": list(FAILURES),
        "current_pitch_type_used": False, "forbidden_2025_trackman_used": False,
        "centers": centers, "chosen": chosen, "exact_gains": exact_gains,
        "exact_gain_2024_v54": exact_gain,
        "top": reports[:100],
    }
    output = ROOT / "research/v56_fine_pitch_failure_prior_2024.npz"
    np.savez_compressed(
        output, target=fold[2024]["target"].astype(np.float32),
        base=base.astype(np.float32), prediction=prediction.astype(np.float32),
        signals=matrix.astype(np.float32), failure_names=np.asarray(FAILURES),
        diagnostics_json=np.asarray(json.dumps(diagnostics)),
    )
    (ROOT / "research/v56_fine_pitch_failure_prior.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "chosen": chosen, "exact_gain_2024_v54": exact_gain,
        "exact_gains": exact_gains,
        "top": reports[:20],
    }, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
