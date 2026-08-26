"""Screen frozen state residual tables on top of v23."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from component_residual_portfolio import apply_component_portfolio
from probability_residual_portfolio import apply_probability_portfolio
from research_residual_portfolio_v19 import SPECS, apply_table, build_table, prepare
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
        anchor = folds[year]["anchor"][section]
        components = {key: value[section] for key, value in folds[year]["components"].items()}
        v21_config = freeze_v21(frame, y - anchor)
        v21 = np.clip(anchor + apply_frozen_portfolio(frame, v21_config), .005, .995)
        component_config = fit_effects(frame, y, anchor, v21, components)
        v22 = np.clip(
            v21 + apply_component_portfolio(frame, components, anchor, component_config),
            .005, .995,
        )
        probability_config = fit_probability_effects(frame, y, v22)
        v23 = np.clip(
            v22 + apply_probability_portfolio(frame, v22, probability_config),
            .005, .995,
        )
        sources[label] = {
            "frame": frame, "y": y, "anchor": anchor,
            "v21_config": v21_config, "component_config": component_config,
            "probability_config": probability_config,
            "v23": v23, "residual": y - v23,
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
        anchor = folds[year]["anchor"][section]
        components = {key: value[section] for key, value in folds[year]["components"].items()}
        v21 = np.clip(
            anchor + apply_frozen_portfolio(frame, source["v21_config"]), .005, .995,
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
        blocks[label] = {"frame": frame, "y": y, "v23": v23, "source": source_name}

    candidates, reports = {}, []
    for name, keys in SPECS.items():
        for shrink in (50., 100., 200., 400., 800., 1600., 3200., 6400.):
            fitted = {
                source_name: build_table(
                    source["frame"], source["residual"], keys, shrink,
                ) for source_name, source in sources.items()
            }
            for mode in ("regular", "all"):
                correction = {}
                for label, block in blocks.items():
                    value = apply_table(block["frame"], fitted[block["source"]], keys)
                    if mode == "regular":
                        value = value * block["frame"]["game_type"].eq("R").to_numpy()
                    correction[label] = value
                key = f"{name}:{int(shrink)}:{mode}"
                candidates[key] = correction
                for scale in (.125, .25, .5, .75, 1., 1.25):
                    gains = {
                        label: gain(block["y"], block["v23"], scale * correction[label])
                        for label, block in blocks.items()
                    }
                    reports.append({
                        "key": key, "name": name, "keys": keys,
                        "shrink": shrink, "mode": mode, "scale": scale,
                        "gains": gains, "min_transfer": min(gains.values()),
                        "mean_transfer": float(np.mean(list(gains.values()))),
                    })

    reports.sort(key=lambda row: (row["min_transfer"], row["mean_transfer"]), reverse=True)
    best = {}
    for row in reports:
        best.setdefault(row["key"], row)
    total = {label: np.zeros(len(block["y"]), dtype=np.float64)
             for label, block in blocks.items()}
    total_gains = {label: 0. for label in blocks}
    remaining, selected = dict(best), []
    for step in range(24):
        winner = None
        for key, config in remaining.items():
            new = {
                label: gain(
                    block["y"], block["v23"],
                    total[label] + float(config["scale"]) * candidates[key][label],
                ) for label, block in blocks.items()
            }
            if min(new.values()) < -1e-9:
                continue
            rank = (min(new.values()), float(np.mean(list(new.values()))))
            if winner is None or rank > winner[0]:
                winner = (rank, key, config, new)
        if winner is None or min(winner[3].values()) <= min(total_gains.values()) + .01:
            break
        _rank, key, config, new = winner
        for label in total:
            total[label] += float(config["scale"]) * candidates[key][label]
        total_gains = new
        selected.append({"step": step + 1, "key": key, "config": config,
                         "total_gains": new})
        remaining.pop(key)

    full_labels = [f"2023_to_2024_q{index}" for index in range(1, 5)]
    full_gain = gain(
        np.concatenate([blocks[label]["y"] for label in full_labels]),
        np.concatenate([blocks[label]["v23"] for label in full_labels]),
        np.concatenate([total[label] for label in full_labels]),
    )
    output = {
        "top_candidates": reports[:100], "selected": selected,
        "final_gains": total_gains, "full_2024_gain": full_gain,
    }
    path = root / "research/v23_state_residual.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "top_candidates": reports[:30], "selected": selected,
        "final_gains": total_gains, "full_2024_gain": full_gain,
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
