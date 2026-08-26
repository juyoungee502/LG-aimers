"""Screen deployable component reblends on top of the v21 portfolio."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_residual_portfolio_v19 import prepare
from residual_portfolio import apply_frozen_portfolio
from train_robust_residual_portfolio import freeze as freeze_v21


def gain(y, base, correction):
    uncertainty = float(y.mean() * (1. - y.mean()))
    residual = y - base
    return float(100000. * np.mean(
        2. * residual * correction - correction * correction
    ) / uncertainty)


def gates(frame):
    regular = frame["game_type"].eq("R").to_numpy()
    two = frame["strikes_before"].eq(2).to_numpy()
    result = {
        "all": np.ones(len(frame), dtype=bool),
        "regular": regular,
        "f_regime": frame["game_type"].eq("F").to_numpy(),
        "other": ~two,
        "two_strike": two,
        "regular_other": regular & ~two,
        "regular_two_strike": regular & two,
    }
    count = frame["count_state"].to_numpy()
    runners = frame["runner_count_code"].to_numpy()
    for value in range(12):
        result[f"regular_count_{value}"] = regular & (count == value)
    for value in range(4):
        result[f"regular_runners_{value}"] = regular & (runners == value)
    return result


def fit_weight(residual, direction):
    finite = np.isfinite(direction)
    if finite.sum() < 100:
        return 0.
    numerator = float(np.dot(residual[finite], direction[finite]))
    denominator = float(np.dot(direction[finite], direction[finite]))
    return float(np.clip(numerator / max(denominator, 1e-12), -1., 1.))


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
    for year in (2023, 2024):
        mask = oof["season"] == year
        length = int(mask.sum())
        folds[year] = {
            "y": oof["target"][mask].astype(np.float64),
            "base": oof["blended"][mask].astype(np.float64),
            "components": oof["predictions"][mask].astype(np.float64),
            "base_blended": oof["base_blended"][mask].astype(np.float64),
            "trackman": oof["trackman_context"][mask].astype(np.float64),
        }
        if length != len(rows[year]):
            raise ValueError(f"OOF rows differ for {year}")

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
        base = folds[year]["base"][section]
        residual = y - base
        configuration = freeze_v21(frame, residual)
        portfolio = apply_frozen_portfolio(frame, configuration)
        sources[label] = {
            "frame": frame, "residual": y - np.clip(base + portfolio, .005, .995),
            "base": base, "configuration": configuration,
            "components": folds[year]["components"][section],
            "base_blended": folds[year]["base_blended"][section],
            "trackman": folds[year]["trackman"][section],
            "gates": gates(frame),
        }

    blocks = {}
    for index in (2, 3):
        section = slice(q[2023][index], q[2023][index + 1])
        blocks[f"2023_q{index + 1}"] = (2023, section, "23h1")
    for index in range(4):
        section = slice(q[2024][index], q[2024][index + 1])
        blocks[f"2023_to_2024_q{index + 1}"] = (2024, section, "23")
    for index in (2, 3):
        section = slice(q[2024][index], q[2024][index + 1])
        blocks[f"2024_h1_to_q{index + 1}"] = (2024, section, "24h1")

    prepared_blocks = {}
    for label, (year, section, source) in blocks.items():
        frame = rows[year].iloc[section]
        base = folds[year]["base"][section]
        portfolio = apply_frozen_portfolio(frame, sources[source]["configuration"])
        prepared_blocks[label] = {
            "frame": frame,
            "y": folds[year]["y"][section],
            "v21": np.clip(base + portfolio, .005, .995),
            "base": base,
            "components": folds[year]["components"][section],
            "base_blended": folds[year]["base_blended"][section],
            "trackman": folds[year]["trackman"][section],
            "source": source,
            "gates": gates(frame),
        }

    candidate_predictions = {
        **{name: ("components", index) for index, name in enumerate(names)},
        "pre_specialist_blend": ("base_blended", None),
        "trackman_specialist": ("trackman", None),
    }
    candidates, reports = {}, []
    for prediction_name, (kind, index) in candidate_predictions.items():
        for gate_name in next(iter(sources.values()))["gates"]:
            weights = {}
            for source_name, source in sources.items():
                candidate = source[kind] if index is None else source[kind][:, index]
                direction = np.nan_to_num(candidate - source["base"], nan=0.)
                direction = direction * source["gates"][gate_name]
                weights[source_name] = fit_weight(source["residual"], direction)
            raw_corrections = {}
            for label, block in prepared_blocks.items():
                candidate = block[kind] if index is None else block[kind][:, index]
                direction = np.nan_to_num(candidate - block["base"], nan=0.)
                raw_corrections[label] = (
                    weights[block["source"]] * direction * block["gates"][gate_name]
                )
            for scale in (.25, .5, .75, 1.):
                gains = {
                    label: gain(block["y"], block["v21"], scale * raw_corrections[label])
                    for label, block in prepared_blocks.items()
                }
                reports.append({
                    "prediction": prediction_name, "gate": gate_name,
                    "scale": scale, "source_weights": weights,
                    "gains": gains, "min_transfer": min(gains.values()),
                    "mean_transfer": float(np.mean(list(gains.values()))),
                })
            key = f"{prediction_name}:{gate_name}"
            candidates[key] = raw_corrections

    reports.sort(key=lambda row: (row["min_transfer"], row["mean_transfer"]), reverse=True)
    best = {}
    for row in reports:
        best.setdefault((row["prediction"], row["gate"]), row)

    total = {label: np.zeros(len(block["y"]), dtype=np.float64)
             for label, block in prepared_blocks.items()}
    total_gains = {label: 0. for label in prepared_blocks}
    remaining = {
        f"{row['prediction']}:{row['gate']}": row
        for row in best.values() if row["min_transfer"] > -.5
    }
    selected = []
    for step in range(24):
        winner = None
        for key, config in remaining.items():
            correction = candidates[key]
            scale = float(config["scale"])
            new = {
                label: gain(
                    block["y"], block["v21"], total[label] + scale * correction[label]
                ) for label, block in prepared_blocks.items()
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
        selected.append({
            "step": step + 1, "key": key, "configuration": config,
            "total_gains": new,
        })
        remaining.pop(key)

    full_labels = [f"2023_to_2024_q{index}" for index in range(1, 5)]
    full_y = np.concatenate([prepared_blocks[label]["y"] for label in full_labels])
    full_base = np.concatenate([prepared_blocks[label]["v21"] for label in full_labels])
    full_correction = np.concatenate([total[label] for label in full_labels])
    full_2024_gain = gain(full_y, full_base, full_correction)

    deploy_frame = rows[2024]
    deploy_base = folds[2024]["base"]
    deploy_residual = folds[2024]["y"] - deploy_base
    deploy_configuration = freeze_v21(deploy_frame, deploy_residual)
    deploy_v21 = np.clip(
        deploy_base + apply_frozen_portfolio(deploy_frame, deploy_configuration),
        .005, .995,
    )
    deploy_residual = folds[2024]["y"] - deploy_v21
    deploy_gates = gates(deploy_frame)
    production_effects = []
    for chosen in selected:
        prediction_name, gate_name = chosen["key"].split(":", 1)
        kind, index = candidate_predictions[prediction_name]
        candidate = folds[2024][kind] if index is None else folds[2024][kind][:, index]
        direction = np.nan_to_num(candidate - deploy_base, nan=0.)
        direction *= deploy_gates[gate_name]
        raw_weight = fit_weight(deploy_residual, direction)
        production_effects.append({
            "prediction": prediction_name, "gate": gate_name,
            "scale": float(chosen["configuration"]["scale"]),
            "raw_weight": raw_weight,
            "effective_weight": raw_weight * float(chosen["configuration"]["scale"]),
        })

    output = {
        "model_names": names,
        "top_candidates": reports[:100],
        "selected": selected,
        "final_gains": total_gains,
        "full_2024_gain": full_2024_gain,
        "production_effects": production_effects,
    }
    path = root / "research/component_residual_portfolio_v21.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "model_names": names, "top_candidates": reports[:30],
        "selected": selected, "final_gains": total_gains,
        "full_2024_gain": full_2024_gain,
        "production_effects": production_effects,
    }, indent=2), flush=True)
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
