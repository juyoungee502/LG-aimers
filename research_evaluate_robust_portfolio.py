"""Evaluate the quarter-robust residual portfolio against v19 and v20."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_numeric_residual_tables import apply_effect, fit_effect
from research_pair_residual_tables import apply_pair, fit_pair
from research_residual_portfolio_v19 import apply_table, build_table, prepare
from residual_portfolio import apply_frozen_portfolio
from train_residual_portfolio import freeze as freeze_v20


def bss(y, prediction):
    uncertainty = float(y.mean() * (1. - y.mean()))
    return float(100000. * (1. - np.mean((y - prediction) ** 2) / uncertainty))


def first_by(rows, key):
    result = {}
    for row in rows:
        result.setdefault(key(row), row)
    return result


def build_correction(root, source_frame, source_residual, source_base,
                     target_frame, target_base, selected):
    numeric_reports = first_by(
        json.loads((root / "research/numeric_residual_tables_v19.json").read_text("utf-8")),
        lambda row: (row["feature"], row["context"]),
    )
    categorical_reports = first_by(
        json.loads((root / "research/residual_portfolio_v19.json").read_text("utf-8")),
        lambda row: row["name"],
    )
    pair_reports = first_by(
        json.loads((root / "research/pair_residual_tables_v19.json").read_text("utf-8")),
        lambda row: (row["name"], row["context"]),
    )
    correction = np.zeros(len(target_frame), dtype=np.float64)
    details = []
    for chosen in selected:
        kind, name, *context_part = chosen["name"].split(":")
        scale = float(chosen["scale"])
        if kind == "numeric":
            context = context_part[0]
            config = numeric_reports[(name, context)]
            fitted = fit_effect(
                source_frame, source_residual, name, source_base, context,
                int(config["n_bins"]), float(config["shrink"]),
            )
            value = apply_effect(target_frame, name, target_base, context, fitted)
        elif kind == "categorical":
            config = categorical_reports[name]
            keys = config["keys"]
            fitted = build_table(
                source_frame, source_residual, keys, float(config["shrink"]),
            )
            value = np.where(
                target_frame["game_type"].eq("R").to_numpy(),
                apply_table(target_frame, fitted, keys), 0.,
            )
        elif kind == "pair":
            context = context_part[0]
            config = pair_reports[(name, context)]
            pair = tuple(config["features"])
            fitted = fit_pair(
                source_frame, source_residual, source_base, pair, context,
                int(config["n_bins"]), float(config["shrink"]),
            )
            value = apply_pair(target_frame, target_base, pair, context, fitted)
        else:
            raise ValueError(f"Unknown effect kind: {kind}")
        effective_weight = scale * float(config["weight"])
        correction += effective_weight * value
        details.append({
            "name": chosen["name"], "scale": scale,
            "base_weight": float(config["weight"]),
            "effective_weight": effective_weight,
            "n_bins": config.get("n_bins"), "shrink": float(config["shrink"]),
            "keys": config.get("keys"), "features": config.get("features"),
        })
    return correction, details


def block_gains(y, base, correction):
    cuts = np.linspace(0, len(y), 5, dtype=int)
    prediction = np.clip(base + correction, .005, .995)
    return {
        "full": bss(y, prediction) - bss(y, base),
        "quarters": [
            bss(y[cuts[i]:cuts[i + 1]], prediction[cuts[i]:cuts[i + 1]])
            - bss(y[cuts[i]:cuts[i + 1]], base[cuts[i]:cuts[i + 1]])
            for i in range(4)
        ],
        "bss": bss(y, prediction),
    }


def main():
    root = Path(__file__).resolve().parent
    raw = pd.read_csv(root / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    rows = {
        year: prepare(raw.loc[raw["season"].eq(year)].reset_index(drop=True))
        for year in (2023, 2024)
    }
    with np.load(root / "outputs/v19_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    folds = {}
    for year in rows:
        mask = oof["season"] == year
        folds[year] = {
            "y": oof["target"][mask].astype(np.float64),
            "base": oof["blended"][mask].astype(np.float64),
        }
    selected = json.loads(
        (root / "research/joint_residual_portfolio_quarters_v19.json").read_text("utf-8")
    )["selected"]
    source_residual = folds[2023]["y"] - folds[2023]["base"]
    robust, details = build_correction(
        root, rows[2023], source_residual, folds[2023]["base"],
        rows[2024], folds[2024]["base"], selected,
    )
    v20_config = freeze_v20(rows[2023], source_residual)
    v20 = apply_frozen_portfolio(rows[2024], v20_config)
    y, base = folds[2024]["y"], folds[2024]["base"]
    report = {
        "v19_bss": bss(y, base),
        "robust": block_gains(y, base, robust),
        "v20": block_gains(y, base, v20),
        "robust_vs_v20": block_gains(y, np.clip(base + v20, .005, .995), robust - v20),
        "correction": {
            "robust_mean": float(robust.mean()),
            "robust_std": float(robust.std()),
            "v20_std": float(v20.std()),
            "correlation": float(np.corrcoef(robust, v20)[0, 1]),
        },
        "effects": details,
    }
    output = root / "research/robust_portfolio_evaluation_v19.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
