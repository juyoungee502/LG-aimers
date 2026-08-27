"""Audit paired historical research predictions as independent v23 axes."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss


def logit(probability):
    probability = np.clip(probability, 1e-5, 1. - 1e-5)
    return np.log(probability / (1. - probability))


def sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -30., 30.)))


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


def column_names(bundle, key, width):
    candidates = (
        f"{key}_names", "names", "model_names", "variants", "labels",
    )
    for candidate in candidates:
        if candidate not in bundle.files:
            continue
        values = np.asarray(bundle[candidate]).astype(str).ravel()
        if len(values) == width:
            return values.tolist()
    return [str(index) for index in range(width)]


def prediction_columns(path, expected_rows):
    output = {}
    with np.load(path, allow_pickle=False) as bundle:
        for key in bundle.files:
            if any(word in key.lower() for word in (
                "target", "season", "report", "name", "center", "weight",
                "direction", "label", "feature", "config", "row",
            )):
                continue
            values = np.asarray(bundle[key])
            if values.ndim not in (1, 2) or values.shape[0] != expected_rows:
                continue
            matrix = values[:, None] if values.ndim == 1 else values
            if matrix.shape[1] > 80 or not np.issubdtype(matrix.dtype, np.number):
                continue
            names = column_names(bundle, key, matrix.shape[1])
            for index, name in enumerate(names):
                column = matrix[:, index].astype(float)
                finite = np.isfinite(column)
                if finite.mean() < .995:
                    continue
                column = np.nan_to_num(column, nan=float(np.nanmean(column)))
                lower, upper = np.quantile(column, [.001, .999])
                if lower < -.001 or upper > 1.001:
                    continue
                if not (.03 < column.mean() < .97 and column.std() > .003):
                    continue
                output[f"{key}:{name}"] = np.clip(column, .005, .995)
    return output


def main():
    root = Path(__file__).resolve().parent
    research = root / "research"
    raw = pd.read_csv(
        root / "data/train.csv",
        usecols=["season", "game_month", "game_type"],
        encoding="utf-8-sig", low_memory=False,
    )
    with np.load(root / "outputs/v23_oof_predictions.npz") as source:
        oof = {key: source[key] for key in source.files}
    folds = {}
    for year in (2023, 2024):
        active = oof["season"] == year
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        folds[year] = {
            "target": oof["target"][active].astype(float),
            "base": oof["blended"][active].astype(float),
            "rows": rows, "masks": masks(rows),
        }

    pairs = []
    for path_2023 in sorted(research.glob("*.npz")):
        if "2023" not in path_2023.name:
            continue
        name_2024 = re.sub("2023", "2024", path_2023.name, count=1)
        path_2024 = path_2023.with_name(name_2024)
        if path_2024.is_file():
            pairs.append((path_2023, path_2024))

    candidates = []
    inventory = []
    weight_grid = np.arange(-.30, .501, .01)
    for path_2023, path_2024 in pairs:
        columns = {
            2023: prediction_columns(path_2023, len(folds[2023]["target"])),
            2024: prediction_columns(path_2024, len(folds[2024]["target"])),
        }
        common = sorted(set(columns[2023]) & set(columns[2024]))
        inventory.append({
            "file_2023": path_2023.name, "file_2024": path_2024.name,
            "accepted_columns": common,
        })
        for column_name in common:
            directions = {
                year: logit(columns[year][column_name]) - logit(folds[year]["base"])
                for year in (2023, 2024)
            }
            curves = {}
            for year in (2023, 2024):
                item = folds[year]
                axis = item["base"] * (1. - item["base"]) * directions[year]
                residual = item["target"] - item["base"]
                curves[str(year)] = {}
                for name, active in item["masks"].items():
                    if not active.any():
                        continue
                    uncertainty = float(
                        item["target"][active].mean()
                        * (1. - item["target"][active].mean())
                    )
                    curves[str(year)][name] = (
                        200000. * float(np.mean(
                            residual[active] * axis[active]
                        )) / uncertainty,
                        100000. * float(np.mean(axis[active] ** 2)) / uncertainty,
                    )
            approximate_reports = []
            for weight in weight_grid:
                gains = {}
                temporal, core = [], []
                for year in (2023, 2024):
                    gains[str(year)] = {}
                    for name, (linear, quadratic) in curves[str(year)].items():
                        gain = linear * weight - quadratic * weight**2
                        gains[str(year)][name] = gain
                        if name != "all":
                            temporal.append(gain)
                        if name in (
                            "all", "first_half", "second_half", "months_3_5",
                            "months_6_7", "months_8_11", "regular", "futures",
                        ):
                            core.append(gain)
                approximate_reports.append({
                    "file_pattern": re.sub("2023", "{year}", path_2023.name, count=1),
                    "column": column_name, "weight": float(weight), "gains": gains,
                    "min_core": min(core), "min_segment": min(temporal),
                    "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
                    "mean_year": np.mean([
                        gains["2023"]["all"], gains["2024"]["all"],
                    ]),
                })
            keys = (
                lambda row: (row["min_core"], row["min_year"], row["mean_year"]),
                lambda row: (row["min_segment"], row["min_year"], row["mean_year"]),
                lambda row: (row["min_year"], row["min_segment"], row["mean_year"]),
                lambda row: (row["mean_year"], row["min_year"], row["min_segment"]),
            )
            chosen_weights = {
                max(approximate_reports, key=key)["weight"] for key in keys
            }
            for weight in chosen_weights:
                gains = {}
                temporal, core = [], []
                for year in (2023, 2024):
                    item = folds[year]
                    candidate = sigmoid(
                        logit(item["base"]) + weight * directions[year]
                    )
                    gains[str(year)] = {}
                    for name, active in item["masks"].items():
                        if not active.any():
                            continue
                        gain = bss(
                            item["target"][active], candidate[active],
                        ) - bss(item["target"][active], item["base"][active])
                        gains[str(year)][name] = gain
                        if name != "all":
                            temporal.append(gain)
                        if name in (
                            "all", "first_half", "second_half", "months_3_5",
                            "months_6_7", "months_8_11", "regular", "futures",
                        ):
                            core.append(gain)
                candidates.append({
                    "file_pattern": re.sub("2023", "{year}", path_2023.name, count=1),
                    "column": column_name, "weight": float(weight), "gains": gains,
                    "min_core": min(core), "min_segment": min(temporal),
                    "min_year": min(gains["2023"]["all"], gains["2024"]["all"]),
                    "mean_year": np.mean([
                        gains["2023"]["all"], gains["2024"]["all"],
                    ]),
                })
        print(
            f"Audited {path_2023.name}: {len(common)} prediction columns",
            flush=True,
        )

    nonzero = [row for row in candidates if abs(row["weight"]) > 1e-8]
    ranking_functions = {
        "maximin_core": lambda row: (
            row["min_core"], row["min_year"], row["mean_year"],
        ),
        "maximin_segment": lambda row: (
            row["min_segment"], row["min_year"], row["mean_year"],
        ),
        "maximin_year": lambda row: (
            row["min_year"], row["min_segment"], row["mean_year"],
        ),
        "best_mean": lambda row: (
            row["mean_year"], row["min_year"], row["min_segment"],
        ),
    }
    rankings = {
        name: sorted(nonzero, key=key, reverse=True)[:150]
        for name, key in ranking_functions.items()
    }
    output = research / "v23_archive_ensemble.json"
    output.write_text(json.dumps({
        "inventory": inventory, "rankings": rankings,
    }, indent=2), encoding="utf-8")
    print(json.dumps({name: rows[:15] for name, rows in rankings.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
