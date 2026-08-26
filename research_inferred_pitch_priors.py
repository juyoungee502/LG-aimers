"""Screen pitch-choice command priors reconstructed from cumulative train columns.

The as-of pitch-mix and detailed-outcome counters expose the label of a training
row in the *next* row for the same pitcher/season.  We use those labels only to
build frozen prior-season lookup tables.  Evaluation and submission rows are
never inspected jointly, so the resulting feature is row-independent at
inference and uses no current-pitch information.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "control_success"
PITCH_TYPES = ("fastball", "breaking", "offspeed")
DETAILS = ("reverse", "middle", "wayoff")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--oof", default="outputs/v15_oof_predictions.npz")
    parser.add_argument("--output", default="research/inferred_pitch_priors.json")
    return parser.parse_args()


def bss(target, prediction):
    rate = float(target.mean())
    return 100000. * (
        1. - np.mean((target - np.clip(prediction, .005, .995)) ** 2)
        / (rate * (1. - rate))
    )


def reconstruct_labels(data):
    """Recover current-row categories from the next cumulative as-of state."""
    work = data[[
        "season", "pitcher_id", "asof_pitcher_n",
        "asof_pitcher_pitchmix_n", *[
            f"asof_pitcher_{name}_rate"
            for name in ("reverse", "middle", "ball", "strike")
        ], *[
            f"asof_pitcher_{name}_rate" for name in PITCH_TYPES
        ], TARGET,
    ]].copy()
    work["_row"] = np.arange(len(work))
    order = work.sort_values(
        ["season", "pitcher_id", "asof_pitcher_n", "_row"], kind="stable"
    ).index
    q = work.loc[order]
    group = [q["season"], q["pitcher_id"]]

    result = pd.DataFrame(index=q.index)
    next_n = q.groupby(group, sort=False)["asof_pitcher_n"].shift(-1)
    detail_valid = (next_n - q["asof_pitcher_n"]).between(.999, 1.001)
    for name in ("reverse", "middle", "ball", "strike"):
        count = np.rint(
            q["asof_pitcher_n"].to_numpy(float)
            * q[f"asof_pitcher_{name}_rate"].to_numpy(float)
        )
        count = pd.Series(count, index=q.index)
        increment = count.groupby(group, sort=False).shift(-1) - count
        result[name] = increment.where(detail_valid)
    result["wayoff"] = np.where(
        result[["reverse", "middle"]].notna().all(axis=1),
        ((q[TARGET].eq(0)) & result["reverse"].eq(0)
         & result["middle"].eq(0)).astype(float),
        np.nan,
    )

    next_mix_n = q.groupby(group, sort=False)["asof_pitcher_pitchmix_n"].shift(-1)
    mix_valid = (next_mix_n - q["asof_pitcher_pitchmix_n"]).between(.999, 1.001)
    increments = []
    for name in PITCH_TYPES:
        count = np.rint(
            q["asof_pitcher_pitchmix_n"].to_numpy(float)
            * q[f"asof_pitcher_{name}_rate"].to_numpy(float)
        )
        count = pd.Series(count, index=q.index)
        increments.append((count.groupby(group, sort=False).shift(-1) - count).to_numpy())
    increments = np.column_stack(increments)
    exact = mix_valid.to_numpy() & np.isclose(increments.sum(axis=1), 1.) \
        & np.isclose((increments > .5).sum(axis=1), 1.)
    labels = np.full(len(q), None, dtype=object)
    labels[exact] = np.asarray(PITCH_TYPES, dtype=object)[np.argmax(increments[exact], axis=1)]
    result["pitch_type"] = labels
    return result.sort_index()


def wide_counts(frame, keys):
    table = (frame.groupby([*keys, "pitch_type"], observed=True, sort=False).size()
             .unstack(fill_value=0).reindex(columns=PITCH_TYPES, fill_value=0))
    table.columns = [f"n_{name}" for name in PITCH_TYPES]
    return table.reset_index()


def signal(history, rows, value_col, outcome_k, selection_k, context):
    """Expected pitcher outcome under a context-specific historical pitch mix."""
    h = history.dropna(subset=["pitch_type", value_col]).copy()
    season_mean = h.groupby("season", observed=True)[value_col].transform("mean")
    h["relative"] = h[value_col].to_numpy(float) - season_mean.to_numpy(float)
    global_by_type = h.groupby("pitch_type", observed=True)["relative"].mean()
    global_rel = {name: float(global_by_type.get(name, 0.)) for name in PITCH_TYPES}

    outcomes = h.groupby(
        ["pitcher_id", "pitch_type"], observed=True, sort=False
    )["relative"].agg(["sum", "count"]).reset_index()
    outcomes["value"] = [
        (total + outcome_k * global_rel[name]) / (count + outcome_k)
        for total, count, name in zip(
            outcomes["sum"], outcomes["count"], outcomes["pitch_type"]
        )
    ]
    outcomes = outcomes.pivot(
        index="pitcher_id", columns="pitch_type", values="value"
    ).reindex(columns=PITCH_TYPES).fillna(0.)
    outcomes.columns = [f"v_{name}" for name in PITCH_TYPES]
    outcomes = outcomes.reset_index()

    overall = wide_counts(h, ["pitcher_id"])
    global_mix = h["pitch_type"].value_counts(normalize=True)
    total = overall[[f"n_{name}" for name in PITCH_TYPES]].sum(axis=1).to_numpy(float)
    for name in PITCH_TYPES:
        overall[f"p_{name}"] = (
            overall[f"n_{name}"].to_numpy(float)
            + selection_k * float(global_mix.get(name, 0.))
        ) / (total + selection_k)

    extra_keys = {
        "pitcher": [], "count": ["count_state"],
        "hand_count": ["batter_hand", "count_state"],
    }[context]
    keys = ["pitcher_id", *extra_keys]
    if not extra_keys:
        contextual = overall[["pitcher_id", *[f"p_{name}" for name in PITCH_TYPES]]].copy()
    else:
        contextual = wide_counts(h, keys)
        contextual_total = contextual[
            [f"n_{name}" for name in PITCH_TYPES]
        ].sum(axis=1).to_numpy(float)
        contextual = contextual.merge(
            overall[["pitcher_id", *[f"p_{name}" for name in PITCH_TYPES]]],
            on="pitcher_id", how="left",
        )
        for name in PITCH_TYPES:
            contextual[f"p_{name}"] = (
                contextual[f"n_{name}"].to_numpy(float)
                + selection_k * contextual[f"p_{name}"].fillna(
                    float(global_mix.get(name, 0.))
                ).to_numpy(float)
            ) / (contextual_total + selection_k)
        contextual = contextual[[*keys, *[f"p_{name}" for name in PITCH_TYPES]]]

    base = overall[["pitcher_id", *[f"p_{name}" for name in PITCH_TYPES]]].merge(
        outcomes, on="pitcher_id", how="left"
    )
    base["base_expected"] = sum(
        base[f"p_{name}"].fillna(0.) * base[f"v_{name}"].fillna(global_rel[name])
        for name in PITCH_TYPES
    )
    query = rows.copy()
    query["_order"] = np.arange(len(query))
    query = query.merge(contextual, on=keys, how="left", sort=False)
    query = query.merge(outcomes, on="pitcher_id", how="left", sort=False)
    query = query.merge(
        base[["pitcher_id", "base_expected"]], on="pitcher_id", how="left", sort=False
    )
    for name in PITCH_TYPES:
        query[f"p_{name}"] = query[f"p_{name}"].fillna(float(global_mix.get(name, 0.)))
        query[f"v_{name}"] = query[f"v_{name}"].fillna(global_rel[name])
    expected = sum(
        query[f"p_{name}"] * query[f"v_{name}"] for name in PITCH_TYPES
    ).to_numpy(float)
    baseline = query["base_expected"].fillna(0.).to_numpy(float)
    order = np.argsort(query["_order"].to_numpy())
    return expected[order], (expected - baseline)[order]


def gain_curve(target, base_prediction, raw_signal, regular_mask):
    """Return gains while preserving the base prediction mean."""
    x = np.zeros(len(target), dtype=np.float64)
    centered = raw_signal[regular_mask] - raw_signal[regular_mask].mean()
    x[regular_mask] = centered
    before = bss(target, base_prediction)
    curve = {}
    for weight in np.arange(-2., 2.001, .05):
        prediction = np.clip(base_prediction + weight * x, .005, .995)
        curve[round(float(weight), 2)] = bss(target, prediction) - before
    return curve


def main():
    args = arguments()
    data = pd.read_csv(Path(args.data_dir) / "train.csv", encoding="utf-8-sig", low_memory=False)
    labels = reconstruct_labels(data)
    history = data[[
        "season", "pitcher_id", "batter_hand", "balls_before", "strikes_before", TARGET,
    ]].copy()
    history["count_state"] = (
        history["balls_before"] * 3 + history["strikes_before"]
    ).astype(np.int8)
    history["pitch_type"] = labels["pitch_type"]
    for name in DETAILS:
        history[name] = labels[name]
    print({
        "pitch_type_coverage": float(history["pitch_type"].notna().mean()),
        "detail_coverage": float(history[list(DETAILS)].notna().all(axis=1).mean()),
        "pitch_type_rates": history["pitch_type"].value_counts(normalize=True).to_dict(),
    }, flush=True)

    with np.load(args.oof, allow_pickle=False) as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    rows = pd.concat([
        data.loc[data["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(oof["target"]) or not np.allclose(
        rows[TARGET].to_numpy(), oof["target"]
    ):
        raise ValueError("v15 OOF and train.csv do not align")

    configs = [
        (outcome_k, selection_k, context)
        for outcome_k in (50., 100., 200., 500.)
        for selection_k in (30., 100., 300.)
        for context in ("pitcher", "count", "hand_count")
    ]
    reports = []
    for outcome_k, selection_k, context in configs:
        fold_signals = {}
        for year in (2023, 2024):
            source = history.loc[history["season"].lt(year)]
            target_rows = history.loc[history["season"].eq(year)].reset_index(drop=True)
            success = signal(
                source, target_rows, TARGET, outcome_k, selection_k, context
            )
            components = [
                signal(source, target_rows, name, outcome_k, selection_k, context)[1]
                for name in DETAILS
            ]
            failure = np.column_stack(components) @ np.array([-.75, 1.5, -.5])
            fold_signals[year] = {
                "expected": success[0], "selection": success[1], "failure": failure,
            }
        for mode in ("expected", "selection", "failure"):
            curves = {}
            for year in (2023, 2024):
                mask = oof["season"] == year
                raw_rows = rows.loc[rows["season"].eq(year)].reset_index(drop=True)
                curves[str(year)] = gain_curve(
                    oof["target"][mask].astype(float),
                    oof["blended"][mask].astype(float),
                    fold_signals[year][mode],
                    raw_rows["game_type"].eq("R").to_numpy(),
                )
            candidates = []
            for weight in curves["2023"]:
                gains = [curves[year][weight] for year in ("2023", "2024")]
                candidates.append((min(gains), float(np.mean(gains)), weight, gains))
            robust = max(candidates)
            reports.append({
                "outcome_k": outcome_k, "selection_k": selection_k,
                "context": context, "mode": mode, "weight": robust[2],
                "gain_2023": robust[3][0], "gain_2024": robust[3][1],
                "min_gain": robust[0], "mean_gain": robust[1],
            })
    reports.sort(key=lambda item: (item["min_gain"], item["mean_gain"]), reverse=True)
    print(json.dumps(reports[:30], indent=2), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "pitch_type_coverage": float(history["pitch_type"].notna().mean()),
        "detail_coverage": float(history[list(DETAILS)].notna().all(axis=1).mean()),
        "top": reports,
    }, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
