"""Strict audit of a row-local season-clock residual expert over v64.

Only the numeric portion of the current row's row_id is parsed.  No peer-row
or evaluation-distribution statistic is computed.  The feature family is
adapted from a public LG Aimers experiment and evaluated in two chronological
directions before it can be promoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parent
TARGET = "control_success"
SCALES = (0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30)
CAT_COLUMNS = [
    "month", "game_type", "pitcher_team", "batter_team", "pitcher",
    "hand", "count", "clock_team", "clock_pitcher", "month_type",
]


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 1e-6, 1.0 - 1e-6)
    reference = float(target.mean() * (1.0 - target.mean()))
    return float(100_000.0 * (1.0 - np.mean(np.square(prediction - target)) / reference))


def clock_features(rows: pd.DataFrame, season_start: dict[int, int]) -> pd.DataFrame:
    number = rows["row_id"].str.extract(r"(\d+)")[0].astype(float).to_numpy()
    offset = rows["season"].map(season_start).astype(float).to_numpy()
    within = number - offset
    progress = np.clip(within / 255_000.0, 0.0, 1.1)
    progress_bin_10 = np.minimum((progress * 10).astype(int), 10).astype(str)
    progress_bin_5 = np.minimum((progress * 5).astype(int), 5).astype(str)
    month = rows["game_month"].astype(str)
    game_type = rows["game_type"].astype(str)
    pitcher_team = rows["pitcher_team_id"].astype(str)
    pitcher = rows["pitcher_id"].astype(str)
    return pd.DataFrame(
        {
            "season_progress": progress,
            "season_progress2": progress**2,
            "season_progress3": progress**3,
            "progress_sin": np.sin(np.pi * progress),
            "progress_cos": np.cos(np.pi * progress),
            "month_progress": rows["game_month"].to_numpy(dtype=float) + progress,
            "inning": rows["inning"].to_numpy(dtype=float),
            "li": rows["li"].fillna(0.0).to_numpy(dtype=float),
            "score_abs": rows["score_diff_pitcher_team"].abs().to_numpy(dtype=float),
            "runners": rows["num_runners_on"].to_numpy(dtype=float),
            "month": month,
            "game_type": game_type,
            "pitcher_team": pitcher_team,
            "batter_team": rows["batter_team_id"].astype(str),
            "pitcher": pitcher,
            "hand": rows["pitcher_hand"].astype(str) + "|" + rows["batter_hand"].astype(str),
            "count": rows["balls_before"].astype(str) + "-" + rows["strikes_before"].astype(str),
            "clock_team": pitcher_team + "|" + pd.Series(progress_bin_10, index=rows.index),
            "clock_pitcher": pitcher + "|" + pd.Series(progress_bin_5, index=rows.index),
            "month_type": month + "|" + game_type,
        },
        index=rows.index,
    ).reset_index(drop=True)


def fit_predict(
    source_features: pd.DataFrame,
    source_target: np.ndarray,
    destination_features: pd.DataFrame,
    *,
    seed: int,
    task_type: str,
    devices: str,
) -> np.ndarray:
    parameters = dict(
        iterations=450,
        depth=5,
        learning_rate=0.025,
        loss_function="RMSE",
        l2_leaf_reg=180.0,
        random_strength=0.5,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        random_seed=seed,
        task_type=task_type,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )
    if task_type == "GPU":
        parameters.update(devices=devices, gpu_ram_part=0.8, border_count=32)
    model = CatBoostRegressor(**parameters)
    model.fit(source_features, source_target, cat_features=CAT_COLUMNS)
    return np.asarray(model.predict(destination_features), dtype=float)


def centered_residual(target: np.ndarray, base: np.ndarray, game_type: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    residual = np.asarray(target, dtype=float) - np.asarray(base, dtype=float)
    centered = residual.copy()
    centers: dict[str, float] = {}
    for group in ("R", "F"):
        mask = np.asarray(game_type).astype(str) == group
        center = float(residual[mask].mean())
        centered[mask] -= center
        centers[group] = center
    return centered, centers


def evaluate(
    target: np.ndarray,
    base: np.ndarray,
    correction: np.ndarray,
    game_type: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    indices = np.arange(len(target))
    halves = np.array_split(indices, 2)
    quarters = np.array_split(indices, 4)
    for scale in SCALES:
        prediction = np.clip(base + scale * correction, 1e-6, 1.0 - 1e-6)
        group_gains = {}
        for group in ("R", "F"):
            mask = np.asarray(game_type).astype(str) == group
            group_gains[group] = bss(target[mask], prediction[mask]) - bss(target[mask], base[mask])
        rows.append(
            {
                "scale": scale,
                "gain": bss(target, prediction) - bss(target, base),
                "half_gains": [bss(target[i], prediction[i]) - bss(target[i], base[i]) for i in halves],
                "quarter_gains": [bss(target[i], prediction[i]) - bss(target[i], base[i]) for i in quarters],
                "group_gains": group_gains,
                "correction_mean": float((scale * correction).mean()),
                "correction_std": float((scale * correction).std()),
            }
        )
    return rows


def cluster_bootstrap(
    target: np.ndarray,
    base: np.ndarray,
    prediction: np.ndarray,
    pitcher: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    reference = float(target.mean() * (1.0 - target.mean()))
    row_gain = np.square(base - target) - np.square(prediction - target)
    grouped = pd.DataFrame({"pitcher": pitcher.astype(str), "gain": row_gain}).groupby("pitcher")["gain"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(dtype=float)
    sizes = grouped["size"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 64):
        count = min(64, repetitions - start)
        samples = rng.integers(0, len(grouped), size=(count, len(grouped)))
        values[start : start + count] = 100_000.0 * sums[samples].sum(axis=1) / sizes[samples].sum(axis=1) / reference
    return {
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    raw = pd.read_csv(ROOT / "data/train.csv", encoding="utf-8-sig", low_memory=False)
    active = raw["season"].isin([2023, 2024]).to_numpy()
    rows = raw.loc[active].reset_index(drop=True)
    season_start_series = raw.groupby("season", sort=True)["row_id"].first().str.extract(r"(\d+)")[0].astype(int) - 1
    season_start = {int(year): int(value) for year, value in season_start_series.items()}
    features = clock_features(rows, season_start)
    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        target = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    if not np.array_equal(target, rows[TARGET].to_numpy(dtype=float)):
        raise ValueError("v64 OOF and train rows are not aligned")
    game_type = rows["game_type"].astype(str).to_numpy()

    mask23 = season == 2023
    mask24 = season == 2024
    index23 = np.flatnonzero(mask23)
    first23, second23 = np.array_split(index23, 2)
    residual23_first, centers23_first = centered_residual(
        target[first23], base[first23], game_type[first23]
    )
    correction23_second = fit_predict(
        features.iloc[first23],
        residual23_first,
        features.iloc[second23],
        seed=6501,
        task_type=args.task_type,
        devices=args.devices,
    )
    within_2023 = evaluate(
        target[second23], base[second23], correction23_second, game_type[second23]
    )

    residual23, centers23 = centered_residual(target[mask23], base[mask23], game_type[mask23])
    correction24 = fit_predict(
        features.loc[mask23],
        residual23,
        features.loc[mask24],
        seed=6502,
        task_type=args.task_type,
        devices=args.devices,
    )
    forward_2024 = evaluate(target[mask24], base[mask24], correction24, game_type[mask24])

    candidates: list[dict[str, object]] = []
    for scale in SCALES[1:]:
        within = next(row for row in within_2023 if row["scale"] == scale)
        forward = next(row for row in forward_2024 if row["scale"] == scale)
        prediction24 = np.clip(base[mask24] + scale * correction24, 1e-6, 1.0 - 1e-6)
        bootstrap = cluster_bootstrap(
            target[mask24], base[mask24], prediction24,
            rows.loc[mask24, "pitcher_id"].to_numpy(), args.bootstrap, 6500 + int(scale * 1000),
        )
        strict_gate = bool(
            within["gain"] > 0.0
            and forward["gain"] > 0.0
            and min(within["half_gains"]) >= 0.0
            and min(forward["half_gains"]) >= 0.0
            and min(forward["group_gains"].values()) >= 0.0
            and bootstrap["ci_low"] > 0.0
        )
        candidates.append(
            {
                "scale": scale,
                "within_2023_second_half_gain": within["gain"],
                "forward_2024_gain": forward["gain"],
                "within_2023": within,
                "forward_2024": forward,
                "bootstrap_2024": bootstrap,
                "strict_gate": strict_gate,
            }
        )
    candidates.sort(
        key=lambda row: (
            row["strict_gate"],
            min(row["within_2023_second_half_gain"], row["forward_2024_gain"]),
        ),
        reverse=True,
    )
    report = {
        "baseline": "v64_public_method_transfer",
        "candidate_family": "row_local_season_clock_residual",
        "source_residual_centers_removed": {
            "within_2023_first_half": centers23_first,
            "forward_2024_full_2023": centers23,
        },
        "candidates": candidates,
        "selected": candidates[0] if candidates[0]["strict_gate"] else None,
        "rules": {
            "current_row_id_only": True,
            "test_peer_rows_used": False,
            "test_distribution_used": False,
            "forbidden_2025_trackman_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    (ROOT / "research").mkdir(exist_ok=True)
    (ROOT / "research/v65_row_clock.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        ROOT / "outputs/v65_row_clock_oof.npz",
        target_2024=target[mask24],
        base_2024=base[mask24],
        correction_2024=correction24,
        game_type_2024=game_type[mask24],
        pitcher_2024=rows.loc[mask24, "pitcher_id"].to_numpy(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
