"""Test whether the unified command model adds independent signal to three axes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_inferred_pitch_priors import bss


def logit(probability):
    probability = np.clip(probability, 1e-5, 1.0 - 1e-5)
    return np.log(probability / (1.0 - probability))


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def masks(size):
    # Month masks are reconstructed from the ordered rows through the stored
    # reports only in the source script, so this joint audit uses quarters in
    # addition to halves.  The previous three-axis candidate has already passed
    # the separate month audit.
    position = np.arange(size)
    return {
        "all": np.ones(size, dtype=bool),
        "first_half": position < size // 2,
        "second_half": position >= size // 2,
        "q1": position < size // 4,
        "q2": (position >= size // 4) & (position < size // 2),
        "q3": (position >= size // 2) & (position < 3 * size // 4),
        "q4": position >= 3 * size // 4,
    }


def curve(target, base, axes, active):
    reference = float(target[active].mean() * (1.0 - target[active].mean()))
    residual = target[active] - base[active]
    selected = [axis[active] for axis in axes]
    linear = np.asarray([
        200000.0 * np.mean(axis * residual) / reference for axis in selected
    ])
    quadratic = 100000.0 * np.asarray([
        [np.mean(left * right) / reference for right in selected]
        for left in selected
    ])
    return linear, quadratic


def approximate(values, weights):
    linear, quadratic = values
    return float(linear @ weights - weights @ quadratic @ weights)


def main():
    root = Path(__file__).resolve().parent
    years = {}
    for year in (2023, 2024):
        with np.load(root / f"research/v23_trackman_no_month_{year}.npz") as z:
            target = z["target"].astype(float)
            base = z["base"].astype(float)
            no_month = z["direction"].astype(float)
        with np.load(root / f"research/v23_prior_command_context_{year}.npz") as z:
            full = z["command_direction"].astype(float)
        with np.load(root / f"research/v23_prior_command_context_{year}_w1.npz") as z:
            recent = z["command_direction"].astype(float)
        with np.load(root / f"research/v23_unified_command_specialist_{year}.npz") as z:
            unified = z["direction"].astype(float)
            exposure = z["pitcher_season_n"].astype(float)
        years[year] = {
            "target": target, "base": base, "no_month": no_month,
            "full": full, "recent": recent, "unified": unified,
            "exposure": exposure, "masks": masks(len(target)),
        }

    candidates = []
    for unified_gate in ("all", "pitch_n_400", "pitch_n_600"):
        curves = {}
        for year, item in years.items():
            command_gate = (item["exposure"] <= 600).astype(float)
            if unified_gate == "all":
                extra_gate = np.ones(len(item["target"]))
            else:
                threshold = int(unified_gate.rsplit("_", 1)[1])
                extra_gate = (item["exposure"] <= threshold).astype(float)
            derivative = item["base"] * (1.0 - item["base"])
            axes = [
                derivative * item["no_month"],
                derivative * command_gate * item["full"],
                derivative * command_gate * item["recent"],
                derivative * extra_gate * item["unified"],
            ]
            curves[year] = {
                name: curve(item["target"], item["base"], axes, active)
                for name, active in item["masks"].items()
            }
        for w0 in np.arange(.60, 1.401, .10):
            for w1 in np.arange(.20, 1.401, .10):
                for w2 in np.arange(.20, 1.201, .10):
                    for w3 in np.arange(-.50, 1.251, .10):
                        weights = np.asarray((w0, w1, w2, w3))
                        gains = {
                            year: {
                                name: approximate(value, weights)
                                for name, value in year_curves.items()
                            }
                            for year, year_curves in curves.items()
                        }
                        segments = [
                            gain for year_gain in gains.values()
                            for name, gain in year_gain.items() if name != "all"
                        ]
                        year_gains = [gains[year]["all"] for year in years]
                        candidates.append({
                            "unified_gate": unified_gate,
                            "weights": weights.tolist(), "gains": gains,
                            "min_segment": min(segments),
                            "min_year": min(year_gains),
                            "mean_year": float(np.mean(year_gains)),
                        })

    ranking_keys = {
        "maximin_segment": lambda row: (
            row["min_segment"], row["min_year"], row["mean_year"],
        ),
        "maximin_year": lambda row: (
            row["min_year"], row["min_segment"], row["mean_year"],
        ),
        "best_mean": lambda row: (
            row["mean_year"], row["min_segment"], row["min_year"],
        ),
    }
    selected = {}
    for ranking in ranking_keys.values():
        for row in sorted(candidates, key=ranking, reverse=True)[:60]:
            key = (row["unified_gate"], *(round(x, 4) for x in row["weights"]))
            selected[key] = row

    exact = []
    for row in selected.values():
        gains = {}
        for year, item in years.items():
            command_gate = (item["exposure"] <= 600).astype(float)
            if row["unified_gate"] == "all":
                extra_gate = np.ones(len(item["target"]))
            else:
                threshold = int(row["unified_gate"].rsplit("_", 1)[1])
                extra_gate = (item["exposure"] <= threshold).astype(float)
            w = row["weights"]
            candidate = sigmoid(
                logit(item["base"]) + w[0] * item["no_month"]
                + w[1] * command_gate * item["full"]
                + w[2] * command_gate * item["recent"]
                + w[3] * extra_gate * item["unified"]
            )
            gains[year] = {
                name: bss(item["target"][active], candidate[active])
                - bss(item["target"][active], item["base"][active])
                for name, active in item["masks"].items()
            }
        segments = [
            gain for year_gain in gains.values()
            for name, gain in year_gain.items() if name != "all"
        ]
        year_gains = [gains[year]["all"] for year in years]
        exact.append({
            "unified_gate": row["unified_gate"], "weights": row["weights"],
            "gains": gains, "min_segment": min(segments),
            "min_year": min(year_gains), "mean_year": float(np.mean(year_gains)),
        })

    result = {
        name: sorted(exact, key=ranking, reverse=True)[:30]
        for name, ranking in ranking_keys.items()
    }
    output = root / "research/v23_joint_unified_command.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: rows[:10] for name, rows in result.items()}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
