"""Audit fixed, row-independent probability sharpness on top of v23."""
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


def anchor(frame, name, fallback):
    def col(column):
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)

    if name == "half":
        value = np.full(len(frame), .5, dtype=np.float64)
    elif name in ("source_rate", "source_prediction_mean"):
        value = np.full(len(frame), float(fallback), dtype=np.float64)
    elif name == "prev1":
        value = col("asof_pitcher_prev1_game_success_rate")
    elif name == "prev3":
        value = col("asof_pitcher_prev3_game_success_rate")
    elif name == "prev5":
        value = col("asof_pitcher_prev5_game_success_rate")
    elif name == "recent_blend":
        value = (
            .2 * col("asof_pitcher_prev1_game_success_rate")
            + .3 * col("asof_pitcher_prev3_game_success_rate")
            + .5 * col("asof_pitcher_prev5_game_success_rate")
        )
    elif name == "career":
        value = col("asof_pitcher_success_rate")
    elif name == "pitcher_batter":
        value = .75 * col("asof_pitcher_success_rate") + .25 * col("asof_batter_success_rate")
    else:
        raise ValueError(name)
    return np.where(np.isfinite(value), value, float(fallback))


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    rows = {
        year: prepare(raw.loc[raw["season"].eq(year)].reset_index(drop=True))
        for year in (2023, 2024)
    }
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    names = list(oof["model_names"].astype(str))
    folds = {}
    for year in rows:
        mask = oof["season"] == year
        matrix = oof["predictions"][mask].astype(np.float64)
        components = {name: matrix[:, index] for index, name in enumerate(names)}
        components["pre_specialist_blend"] = oof["base_blended"][mask].astype(np.float64)
        folds[year] = {
            "y": oof["target"][mask].astype(np.float64),
            "anchor": oof["blended"][mask].astype(np.float64),
            "components": components,
        }

    q = {year: np.linspace(0, len(rows[year]), 5, dtype=int) for year in rows}
    source_specs = {
        "23h1": (2023, slice(0, q[2023][2])),
        "23": (2023, slice(None)),
        "24h1": (2024, slice(0, q[2024][2])),
    }
    sources = {}
    for label, (year, section) in source_specs.items():
        frame = rows[year].iloc[section]
        y = folds[year]["y"][section]
        base_anchor = folds[year]["anchor"][section]
        components = {key: value[section] for key, value in folds[year]["components"].items()}
        v21_config = freeze_v21(frame, y - base_anchor)
        v21 = np.clip(base_anchor + apply_frozen_portfolio(frame, v21_config), .005, .995)
        component_config = fit_effects(frame, y, base_anchor, v21, components)
        v22 = np.clip(
            v21 + apply_component_portfolio(frame, components, base_anchor, component_config),
            .005, .995,
        )
        probability_config = fit_probability_effects(frame, y, v22)
        v23 = np.clip(
            v22 + apply_probability_portfolio(frame, v22, probability_config),
            .005, .995,
        )
        sources[label] = {
            "frame": frame, "y": y, "base_anchor": base_anchor,
            "v21_config": v21_config, "component_config": component_config,
            "probability_config": probability_config, "v23": v23,
            "target_rate": float(y.mean()), "prediction_mean": float(v23.mean()),
        }

    block_specs = {}
    for index in (2, 3):
        block_specs[f"2023_q{index + 1}"] = (
            2023, slice(q[2023][index], q[2023][index + 1]), "23h1",
        )
    for index in range(4):
        block_specs[f"2023_to_2024_q{index + 1}"] = (
            2024, slice(q[2024][index], q[2024][index + 1]), "23",
        )
    for index in (2, 3):
        block_specs[f"2024_h1_to_q{index + 1}"] = (
            2024, slice(q[2024][index], q[2024][index + 1]), "24h1",
        )
    blocks = {}
    for label, (year, section, source_name) in block_specs.items():
        source = sources[source_name]
        frame = rows[year].iloc[section]
        y = folds[year]["y"][section]
        base_anchor = folds[year]["anchor"][section]
        components = {key: value[section] for key, value in folds[year]["components"].items()}
        v21 = np.clip(
            base_anchor + apply_frozen_portfolio(frame, source["v21_config"]), .005, .995,
        )
        v22 = np.clip(
            v21 + apply_component_portfolio(
                frame, components, base_anchor, source["component_config"],
            ), .005, .995,
        )
        v23 = np.clip(
            v22 + apply_probability_portfolio(
                frame, v22, source["probability_config"],
            ), .005, .995,
        )
        blocks[label] = {
            "frame": frame, "y": y, "v23": v23, "source": source_name,
        }

    reports = []
    for anchor_name in (
        "half", "source_rate", "source_prediction_mean", "prev1", "prev3",
        "prev5", "recent_blend", "career", "pitcher_batter",
    ):
        for gate_name in ("all", "regular", "other", "two_strike"):
            for alpha in np.arange(.80, 1.501, .01):
                gains = {}
                for label, block in blocks.items():
                    source = sources[block["source"]]
                    fallback = (
                        source["prediction_mean"] if anchor_name == "source_prediction_mean"
                        else source["target_rate"]
                    )
                    reference = anchor(block["frame"], anchor_name, fallback)
                    active = np.ones(len(reference), dtype=bool)
                    if gate_name == "regular":
                        active = block["frame"]["game_type"].eq("R").to_numpy()
                    elif gate_name == "other":
                        active = block["frame"]["strikes_before"].ne(2).to_numpy()
                    elif gate_name == "two_strike":
                        active = block["frame"]["strikes_before"].eq(2).to_numpy()
                    correction = (float(alpha) - 1.) * (block["v23"] - reference) * active
                    gains[label] = gain(block["y"], block["v23"], correction)
                reports.append({
                    "anchor": anchor_name, "gate": gate_name, "alpha": float(alpha),
                    "gains": gains, "min_transfer": min(gains.values()),
                    "mean_transfer": float(np.mean(list(gains.values()))),
                })
    reports.sort(key=lambda row: (row["min_transfer"], row["mean_transfer"]), reverse=True)
    best = reports[0]
    full_labels = [f"2023_to_2024_q{index}" for index in range(1, 5)]
    corrections = []
    for label in full_labels:
        block = blocks[label]
        source = sources[block["source"]]
        fallback = (
            source["prediction_mean"] if best["anchor"] == "source_prediction_mean"
            else source["target_rate"]
        )
        reference = anchor(block["frame"], best["anchor"], fallback)
        active = np.ones(len(reference), dtype=bool)
        if best["gate"] == "regular":
            active = block["frame"]["game_type"].eq("R").to_numpy()
        elif best["gate"] == "other":
            active = block["frame"]["strikes_before"].ne(2).to_numpy()
        elif best["gate"] == "two_strike":
            active = block["frame"]["strikes_before"].eq(2).to_numpy()
        corrections.append((best["alpha"] - 1.) * (block["v23"] - reference) * active)
    full_gain = gain(
        np.concatenate([blocks[label]["y"] for label in full_labels]),
        np.concatenate([blocks[label]["v23"] for label in full_labels]),
        np.concatenate(corrections),
    )
    output = {"top": reports[:200], "best": best, "full_2024_gain": full_gain}
    path = root / "research/v23_sharpness.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"top": reports[:50], "best": best,
                      "full_2024_gain": full_gain}, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
