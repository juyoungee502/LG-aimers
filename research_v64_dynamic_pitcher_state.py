"""Audit a row-local dynamic pitcher state correction over the v61 baseline.

The idea is independently rebuilt from a publicly documented approach: model
each pitcher's prior-season command level as a league-centred latent state,
estimate its year-to-year persistence from strictly earlier seasons, and let
the official current-row ``asof_pitcher_*`` fields update that prior.  The
validation season, evaluation peers, and 2025 TrackMan history are never used
to build a scored row.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
EPS = 1e-6
CLIP = (0.005, 0.995)
STATE_SMOOTHING = 200.0
TRANSITION_RIDGE = 1.0

# The first candidate exactly follows the public, pre-specified deployment.
# Smaller weights are audited only to measure compatibility with the much
# stronger v61 baseline; they are not selected from a same-fold oracle.
CANDIDATES = {
    "ar_k30_w050_public": ("ar", 30.0, 0.50, "all"),
    "ar_k30_w050_regular": ("ar", 30.0, 0.50, "regular"),
    "ar_k30_w025_regular": ("ar", 30.0, 0.25, "regular"),
    "ar_k30_w0125_regular": ("ar", 30.0, 0.125, "regular"),
    "ar_k100_w025_regular": ("ar", 100.0, 0.25, "regular"),
    "last_k30_w025_regular": ("last", 30.0, 0.25, "regular"),
}


def logit(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def expit(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass(frozen=True)
class CareerState:
    n: pd.Series
    successes: pd.Series


def load_frame() -> pd.DataFrame:
    columns = [
        "season", "pitcher_id", "game_type", "asof_pitcher_n",
        "asof_pitcher_success_rate", "control_success",
    ]
    frame = pd.read_csv(
        ROOT / "data/train.csv", usecols=columns,
        encoding="utf-8-sig", low_memory=False,
    )
    missing = frame["asof_pitcher_success_rate"].isna()
    if not frame.loc[missing, "asof_pitcher_n"].eq(0).all():
        raise ValueError("positive-count pitcher has a missing success rate")
    frame["asof_pitcher_success_rate"] = frame[
        "asof_pitcher_success_rate"
    ].fillna(0.0)
    return frame


def season_states(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    stats = frame.groupby(["season", "pitcher_id"], sort=False)[
        "control_success"
    ].agg(["sum", "count"])
    league = frame.groupby("season", sort=False)["control_success"].mean()
    league_rates = {int(year): float(rate) for year, rate in league.items()}
    years = stats.index.get_level_values("season")
    priors = np.asarray([league_rates[int(year)] for year in years], dtype=float)
    counts = stats["count"].to_numpy(float)
    posterior = (
        stats["sum"].to_numpy(float) + STATE_SMOOTHING * priors
    ) / (counts + STATE_SMOOTHING)
    states = stats.copy()
    states["latent"] = logit(posterior) - logit(priors)
    states["reliability"] = counts / (counts + STATE_SMOOTHING)
    return states, league_rates


def fit_ar1(states: pd.DataFrame, prediction_year: int) -> tuple[float, int]:
    history = states.loc[
        states.index.get_level_values("season") < prediction_year
    ].reset_index()
    previous = history.rename(columns={
        "season": "previous_year", "latent": "previous_latent",
        "reliability": "previous_reliability",
    })[["pitcher_id", "previous_year", "previous_latent", "previous_reliability"]]
    current = history.rename(columns={
        "season": "current_year", "latent": "current_latent",
        "reliability": "current_reliability",
    })[["pitcher_id", "current_year", "current_latent", "current_reliability"]]
    pairs = current.merge(previous, on="pitcher_id", how="inner")
    pairs = pairs.loc[pairs["current_year"].eq(pairs["previous_year"] + 1)]
    if pairs.empty:
        return 0.0, 0
    weight = np.sqrt(
        pairs["current_reliability"].to_numpy(float)
        * pairs["previous_reliability"].to_numpy(float)
    )
    x = pairs["previous_latent"].to_numpy(float)
    y = pairs["current_latent"].to_numpy(float)
    rho = np.sum(weight * x * y) / (
        np.sum(weight * x * x) + TRANSITION_RIDGE
    )
    return float(np.clip(rho, 0.0, 1.0)), int(len(pairs))


def end_state(rows: pd.DataFrame) -> CareerState:
    indices = rows.groupby("pitcher_id", sort=False)["asof_pitcher_n"].idxmax()
    last = rows.loc[indices]
    before_n = last["asof_pitcher_n"].to_numpy(float)
    before_success = np.rint(
        before_n * last["asof_pitcher_success_rate"].to_numpy(float)
    )
    ids = last["pitcher_id"].to_numpy()
    return CareerState(
        n=pd.Series(before_n + 1.0, index=ids),
        successes=pd.Series(
            before_success + last["control_success"].to_numpy(float), index=ids,
        ),
    )


def career_before(frame: pd.DataFrame, prediction_year: int) -> CareerState:
    history = frame.loc[frame["season"].lt(prediction_year)]
    if history.empty:
        return CareerState(pd.Series(dtype=float), pd.Series(dtype=float))
    latest = end_state(history)
    return latest


def latest_state(states: pd.DataFrame, prediction_year: int) -> pd.DataFrame:
    history = states.loc[
        states.index.get_level_values("season") < prediction_year
    ].reset_index()
    indices = history.groupby("pitcher_id", sort=False)["season"].idxmax()
    return history.loc[
        indices, ["pitcher_id", "season", "latent"]
    ].set_index("pitcher_id")


def dynamic_deltas(
    rows: pd.DataFrame,
    prediction_year: int,
    states: pd.DataFrame,
    league_rates: dict[int, float],
    career: CareerState,
) -> tuple[dict[str, np.ndarray], dict]:
    rho, pair_count = fit_ar1(states, prediction_year)
    latest = latest_state(states, prediction_year)
    ids = rows["pitcher_id"]
    prior_n = ids.map(career.n).fillna(0.0).to_numpy(float)
    prior_success = ids.map(career.successes).fillna(0.0).to_numpy(float)
    career_n = rows["asof_pitcher_n"].to_numpy(float)
    career_success = np.rint(
        career_n * rows["asof_pitcher_success_rate"].to_numpy(float)
    )
    current_n = np.maximum(career_n - prior_n, 0.0)
    current_success = np.clip(career_success - prior_success, 0.0, current_n)

    league_prior = float(league_rates[prediction_year - 1])
    last_latent = ids.map(latest["latent"]).fillna(0.0).to_numpy(float)
    last_year = ids.map(latest["season"]).to_numpy(float)
    known = np.isfinite(last_year)
    gap = np.where(known, prediction_year - last_year, 0.0)
    ar_latent = np.where(known, last_latent * np.power(rho, gap), 0.0)
    probabilities = {
        "ar": expit(logit(league_prior) + ar_latent),
        "last": expit(logit(league_prior) + last_latent),
    }
    deltas = {}
    for method, prior_probability in probabilities.items():
        for strength in (30.0, 100.0):
            dynamic = (
                current_success + strength * prior_probability
            ) / (current_n + strength)
            neutral = (
                current_success + strength * league_prior
            ) / (current_n + strength)
            deltas[f"{method}_k{int(strength)}"] = dynamic - neutral
    return deltas, {
        "rho": rho,
        "transition_pairs": pair_count,
        "league_prior": league_prior,
        "known_row_fraction": float(known.mean()),
        "known_pitchers": int(ids.loc[known].nunique()),
        "current_n_mean": float(current_n.mean()),
        "ar_k30_std": float(deltas["ar_k30"].std()),
        "ar_k30_mean": float(deltas["ar_k30"].mean()),
    }


def segment_gain(target: np.ndarray, base: np.ndarray, delta: np.ndarray) -> float:
    return float(bss(target, np.clip(base + delta, *CLIP)) - bss(target, base))


def main() -> None:
    frame = load_frame()
    states, league_rates = season_states(frame)
    with np.load(ROOT / "outputs/v61_oof_predictions.npz") as archive:
        oof = {key: archive[key] for key in archive.files}
    reports = {}
    audits = {}
    corrections = {}
    for year in (2023, 2024):
        rows = frame.loc[frame["season"].eq(year)].reset_index(drop=True)
        mask = oof["season"].astype(int) == year
        target = oof["target"][mask].astype(float)
        base = oof["blended"][mask].astype(float)
        if len(rows) != len(base):
            raise ValueError(f"OOF alignment failed for {year}")
        deltas, audit = dynamic_deltas(
            rows, year, states, league_rates, career_before(frame, year),
        )
        audits[str(year)] = audit
        reports[str(year)] = {}
        corrections[str(year)] = deltas["ar_k30"]
        v61_delta = (
            oof["blended"][mask].astype(float)
            - np.load(ROOT / "outputs/v60_oof_predictions.npz")["blended"][mask]
        )
        for name, (method, strength, weight, gate) in CANDIDATES.items():
            correction = weight * deltas[f"{method}_k{int(strength)}"]
            half = len(rows) // 2
            regular = rows["game_type"].eq("R").to_numpy()
            futures = ~regular
            if gate == "regular":
                correction = correction * regular.astype(float)
            regular_positions = np.flatnonzero(regular)
            regular_quarters = np.array_split(regular_positions, 4)
            reports[str(year)][name] = {
                "gain": segment_gain(target, base, correction),
                "first_half_gain": segment_gain(
                    target[:half], base[:half], correction[:half],
                ),
                "second_half_gain": segment_gain(
                    target[half:], base[half:], correction[half:],
                ),
                "regular_gain": segment_gain(
                    target[regular], base[regular], correction[regular],
                ),
                "futures_gain": segment_gain(
                    target[futures], base[futures], correction[futures],
                ),
                "correction_mean": float(correction.mean()),
                "correction_std": float(correction.std()),
                "regular_quarter_gains": [
                    segment_gain(
                        target[positions], base[positions], correction[positions],
                    )
                    for positions in regular_quarters
                ],
                "v61_delta_correlation_regular": float(np.corrcoef(
                    correction[regular], v61_delta[regular],
                )[0, 1]),
            }

    summary = []
    for name in CANDIDATES:
        values = [reports[str(year)][name] for year in (2023, 2024)]
        summary.append({
            "candidate": name,
            "gain_2023": values[0]["gain"],
            "gain_2024": values[1]["gain"],
            "min_year_gain": min(value["gain"] for value in values),
            "mean_year_gain": float(np.mean([value["gain"] for value in values])),
            "min_half_gain": min(
                value[key] for value in values
                for key in ("first_half_gain", "second_half_gain")
            ),
            "min_regular_quarter_gain": min(
                gain for value in values for gain in value["regular_quarter_gains"]
            ),
        })
    summary.sort(
        key=lambda item: (item["min_year_gain"], item["mean_year_gain"]),
        reverse=True,
    )
    output = {
        "baseline": "v61_public_complete_shape",
        "candidate_family": "dynamic_pitcher_latent_ar1",
        "state_smoothing": STATE_SMOOTHING,
        "transition_ridge": TRANSITION_RIDGE,
        "audits": audits,
        "folds": reports,
        "summary": summary,
        "rules": {
            "strict_prior_seasons_only": True,
            "current_row_asof_only": True,
            "validation_or_test_peer_aggregation": False,
            "forbidden_2025_trackman_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    path = ROOT / "research/v64_dynamic_pitcher_state.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
