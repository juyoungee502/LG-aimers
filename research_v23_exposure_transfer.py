"""Forward-audit frozen player-exposure corrections on top of v23.

Every lookup is fitted from an earlier training block and applied without
grouping the destination rows.  This mirrors production inference, where the
final tables are frozen from 2024 and each 2025 row is evaluated independently.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from component_residual_portfolio import apply_component_portfolio
from probability_residual_portfolio import apply_probability_portfolio
from research_residual_portfolio_v19 import prepare
from residual_portfolio import apply_frozen_portfolio
from train_component_residual_portfolio import fit_effects
from train_probability_residual_portfolio import fit_probability_effects
from train_robust_residual_portfolio import freeze as freeze_v21


def gain(y, base, correction):
    uncertainty = float(y.mean() * (1. - y.mean()))
    residual = y - base
    return float(100000. * np.mean(
        2. * residual * correction - correction * correction
    ) / uncertainty)


def component_dict(oof, mask):
    names = list(oof["model_names"].astype(str))
    matrix = oof["predictions"][mask].astype(np.float64)
    values = {name: matrix[:, index] for index, name in enumerate(names)}
    values["pre_specialist_blend"] = oof["base_blended"][mask].astype(np.float64)
    return values


def build_sources_and_blocks(raw, oof):
    rows = {
        year: prepare(raw.loc[raw["season"].eq(year)].reset_index(drop=True))
        for year in (2023, 2024)
    }
    folds = {}
    for year in rows:
        mask = oof["season"] == year
        folds[year] = {
            "y": oof["target"][mask].astype(np.float64),
            "anchor": oof["blended"][mask].astype(np.float64),
            "components": component_dict(oof, mask),
        }

    quarters = {
        year: np.linspace(0, len(rows[year]), 5, dtype=int) for year in rows
    }
    source_specs = {
        "23h1": (2023, slice(0, quarters[2023][2])),
        "23": (2023, slice(None)),
        "24h1": (2024, slice(0, quarters[2024][2])),
    }
    sources = {}
    for label, (year, section) in source_specs.items():
        frame = rows[year].iloc[section].reset_index(drop=True)
        y = folds[year]["y"][section]
        anchor = folds[year]["anchor"][section]
        components = {
            key: value[section] for key, value in folds[year]["components"].items()
        }
        v21_config = freeze_v21(frame, y - anchor)
        v21 = np.clip(
            anchor + apply_frozen_portfolio(frame, v21_config), .005, .995,
        )
        component_config = fit_effects(
            frame, y, anchor, v21, components,
        )
        v22 = np.clip(
            v21 + apply_component_portfolio(
                frame, components, anchor, component_config,
            ), .005, .995,
        )
        probability_config = fit_probability_effects(frame, y, v22)
        v23 = np.clip(
            v22 + apply_probability_portfolio(frame, v22, probability_config),
            .005, .995,
        )
        sources[label] = {
            "frame": frame, "y": y, "prediction": v23,
            "v21_config": v21_config,
            "component_config": component_config,
            "probability_config": probability_config,
        }

    block_specs = {}
    for index in (2, 3):
        block_specs[f"2023_q{index + 1}"] = (
            2023, slice(quarters[2023][index], quarters[2023][index + 1]), "23h1",
        )
    for index in range(4):
        block_specs[f"2023_to_2024_q{index + 1}"] = (
            2024, slice(quarters[2024][index], quarters[2024][index + 1]), "23",
        )
    for index in (2, 3):
        block_specs[f"2024_h1_to_q{index + 1}"] = (
            2024, slice(quarters[2024][index], quarters[2024][index + 1]), "24h1",
        )

    blocks = {}
    for label, (year, section, source_name) in block_specs.items():
        source = sources[source_name]
        frame = rows[year].iloc[section].reset_index(drop=True)
        y = folds[year]["y"][section]
        anchor = folds[year]["anchor"][section]
        components = {
            key: value[section] for key, value in folds[year]["components"].items()
        }
        v21 = np.clip(
            anchor + apply_frozen_portfolio(
                frame, source["v21_config"],
            ), .005, .995,
        )
        v22 = np.clip(
            v21 + apply_component_portfolio(
                frame, components, anchor, source["component_config"],
            ), .005, .995,
        )
        v23 = np.clip(
            v22 + apply_probability_portfolio(
                frame, v22, source["probability_config"],
            ), .005, .995,
        )
        blocks[label] = {
            "frame": frame, "y": y, "prediction": v23,
            "source": source_name,
        }
    return sources, blocks


def exposure_values(counts, transform):
    counts = np.asarray(counts, dtype=np.float64)
    if transform == "linear":
        return counts
    if transform == "sqrt":
        return np.sqrt(counts)
    if transform == "log":
        return np.log1p(counts)
    if transform.startswith("high_"):
        quantile = float(transform.split("_", 1)[1])
        threshold = float(np.quantile(counts, quantile))
        return (counts >= threshold).astype(np.float64)
    raise ValueError(transform)


def fit_exposure(frame, residual, column, transform):
    grouped = frame.groupby(column, observed=True, sort=False).size()
    keys = grouped.index.to_numpy(np.int64)
    raw = exposure_values(grouped.to_numpy(np.float64), transform)
    table = pd.Series(raw, index=keys)
    source_rows = frame[column].map(table).to_numpy(np.float64)
    center = float(source_rows.mean())
    scale = float(source_rows.std())
    if scale <= 0.:
        raise ValueError(f"Degenerate exposure direction: {column}/{transform}")
    values = (raw - center) / scale
    normalized = pd.Series(values, index=keys)
    direction = frame[column].map(normalized).fillna(0.).to_numpy(np.float64)
    coefficient = float(np.dot(residual, direction) / np.dot(direction, direction))
    return {
        "column": column, "transform": transform,
        "coefficient": coefficient,
        "table": {str(int(key)): float(value) for key, value in normalized.items()},
    }


def apply_exposure(frame, config):
    return (
        frame[config["column"]].astype(str).map(config["table"])
        .fillna(0.).to_numpy(np.float64)
    )


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    sources, blocks = build_sources_and_blocks(raw, oof)

    reports = []
    for column in ("batter_id", "pitcher_id"):
        for transform in ("linear", "sqrt", "log", "high_.50", "high_.75", "high_.90"):
            configs = {}
            coefficients = {}
            for source_name, source in sources.items():
                config = fit_exposure(
                    source["frame"], source["y"] - source["prediction"],
                    column, transform,
                )
                configs[source_name] = config
                coefficients[source_name] = config["coefficient"]
            for shrink in (.125, .25, .5, .75, 1.):
                gains = {}
                for label, block in blocks.items():
                    config = configs[block["source"]]
                    correction = (
                        float(shrink) * config["coefficient"]
                        * apply_exposure(block["frame"], config)
                    )
                    gains[label] = gain(
                        block["y"], block["prediction"], correction,
                    )
                reports.append({
                    "column": column, "transform": transform,
                    "shrink": shrink, "coefficients": coefficients,
                    "gains": gains, "min_transfer": min(gains.values()),
                    "mean_transfer": float(np.mean(list(gains.values()))),
                    "positive_blocks": int(sum(value > 0 for value in gains.values())),
                })
    reports.sort(
        key=lambda row: (
            row["positive_blocks"], row["min_transfer"], row["mean_transfer"],
        ), reverse=True,
    )
    output = {"top": reports[:100], "all": reports}
    path = root / "research/v23_exposure_transfer.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"top": reports[:30]}, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
