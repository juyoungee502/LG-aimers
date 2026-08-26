"""Screen fine pitch-type failure priors from aligned official Trackman logs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_inferred_pitch_priors import bss, reconstruct_labels


TARGET = "control_success"
FINE_TYPES = (
    "Fastball", "Slider", "Curveball", "ChangeUp", "Splitter", "Sinker",
    "Cutter", "Other",
)
FAILURES = ("reverse", "middle", "wayoff")
FAILURE_WEIGHTS = np.array([-.75, 1.50, -.50], dtype=np.float64)


def normalized_pitch_type(values):
    values = values.replace({
        "Changeup": "ChangeUp", "Four-Seam": "Fastball", "SInker": "Sinker",
    })
    return values.where(values.isin(FINE_TYPES[:-1]), "Other")


def aligned_history(root: Path, data: pd.DataFrame, labels: pd.DataFrame):
    with np.load(root / "outputs" / "trackman_pitch_alignment.npz") as aligned:
        links = pd.DataFrame({
            "row_id": aligned["row_id"].astype(str),
            "trackman_id": aligned["trackman_id"].astype(str),
        })
    trackman = pd.read_csv(
        root / "data" / "trackman_history.csv",
        usecols=["trackman_id", "tagged_pitch_type"],
        encoding="utf-8-sig", low_memory=False,
    )
    trackman["trackman_id"] = trackman["trackman_id"].astype(str)
    trackman["pitch_type_fine"] = normalized_pitch_type(trackman["tagged_pitch_type"])
    links = links.merge(
        trackman[["trackman_id", "pitch_type_fine"]],
        on="trackman_id", how="left", validate="one_to_one",
    )
    columns = [
        "row_id", "season", "pitcher_id", "batter_hand", "balls_before",
        "strikes_before", "game_type", TARGET,
    ]
    history = data[columns].copy()
    history["row_id"] = history["row_id"].astype(str)
    for failure in FAILURES:
        history[failure] = labels[failure].to_numpy(np.float32)
    history = history.merge(
        links[["row_id", "pitch_type_fine"]], on="row_id", how="inner",
        validate="one_to_one",
    )
    history["count_state"] = (
        history["balls_before"] * 3 + history["strikes_before"]
    ).astype(np.int8)
    return history.dropna(subset=["pitch_type_fine", *FAILURES])


def wide_counts(frame, keys):
    table = (
        frame.groupby([*keys, "pitch_type_fine"], observed=True, sort=False)
        .size().unstack(fill_value=0).reindex(columns=FINE_TYPES, fill_value=0)
    )
    table.columns = [f"n_{name}" for name in FINE_TYPES]
    return table.reset_index()


def selection_delta(history, rows, value_column, outcome_k, selection_k):
    """Expected relative outcome under context mix minus overall pitcher mix."""
    source = history.dropna(subset=["pitch_type_fine", value_column]).copy()
    source["relative"] = (
        source[value_column]
        - source.groupby("season", observed=True)[value_column].transform("mean")
    )
    global_table = source.groupby("pitch_type_fine", observed=True)["relative"].agg(
        ["sum", "count"]
    )
    global_relative = {
        name: float(global_table.loc[name, "sum"] / global_table.loc[name, "count"])
        if name in global_table.index else 0.0 for name in FINE_TYPES
    }
    outcomes = source.groupby(
        ["pitcher_id", "pitch_type_fine"], observed=True, sort=False,
    )["relative"].agg(["sum", "count"]).reset_index()
    outcomes["value"] = [
        (total + outcome_k * global_relative[name]) / (count + outcome_k)
        for total, count, name in zip(
            outcomes["sum"], outcomes["count"], outcomes["pitch_type_fine"],
        )
    ]
    outcomes = outcomes.pivot(
        index="pitcher_id", columns="pitch_type_fine", values="value",
    ).reindex(columns=FINE_TYPES).fillna(0.0)
    outcomes.columns = [f"v_{name}" for name in FINE_TYPES]
    outcomes = outcomes.reset_index()

    overall = wide_counts(source, ["pitcher_id"])
    global_mix = source["pitch_type_fine"].value_counts(normalize=True)
    count_columns = [f"n_{name}" for name in FINE_TYPES]
    total = overall[count_columns].sum(axis=1).to_numpy(float)
    for name in FINE_TYPES:
        overall[f"p_{name}"] = (
            overall[f"n_{name}"].to_numpy(float)
            + selection_k * float(global_mix.get(name, 0.0))
        ) / (total + selection_k)

    keys = ["pitcher_id", "batter_hand", "count_state"]
    contextual = wide_counts(source, keys)
    contextual_total = contextual[count_columns].sum(axis=1).to_numpy(float)
    probability_columns = [f"p_{name}" for name in FINE_TYPES]
    contextual = contextual.merge(
        overall[["pitcher_id", *probability_columns]],
        on="pitcher_id", how="left",
    )
    for name in FINE_TYPES:
        contextual[f"p_{name}"] = (
            contextual[f"n_{name}"].to_numpy(float)
            + selection_k * contextual[f"p_{name}"].fillna(
                float(global_mix.get(name, 0.0))
            ).to_numpy(float)
        ) / (contextual_total + selection_k)
    contextual = contextual[[*keys, *probability_columns]]

    base = overall[["pitcher_id", *probability_columns]].merge(
        outcomes, on="pitcher_id", how="left",
    )
    base["base_expected"] = sum(
        base[f"p_{name}"].fillna(0.0)
        * base[f"v_{name}"].fillna(global_relative[name])
        for name in FINE_TYPES
    )
    query = rows.copy()
    if "count_state" not in query:
        query["count_state"] = (
            query["balls_before"] * 3 + query["strikes_before"]
        ).astype(np.int8)
    query["_order"] = np.arange(len(query))
    query = query.merge(contextual, on=keys, how="left", sort=False)
    query = query.merge(outcomes, on="pitcher_id", how="left", sort=False)
    query = query.merge(
        base[["pitcher_id", "base_expected"]], on="pitcher_id", how="left",
        sort=False,
    )
    for name in FINE_TYPES:
        query[f"p_{name}"] = query[f"p_{name}"].fillna(
            float(global_mix.get(name, 0.0))
        )
        query[f"v_{name}"] = query[f"v_{name}"].fillna(global_relative[name])
    expected = sum(
        query[f"p_{name}"] * query[f"v_{name}"] for name in FINE_TYPES
    ).to_numpy(float)
    baseline = query["base_expected"].fillna(0.0).to_numpy(float)
    return (expected - baseline)[np.argsort(query["_order"].to_numpy())]


def failure_signal(history, rows, outcome_k, selection_k):
    return np.column_stack([
        selection_delta(history, rows, failure, outcome_k, selection_k)
        for failure in FAILURES
    ]) @ FAILURE_WEIGHTS


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    labels = reconstruct_labels(data)
    history = aligned_history(root, data, labels)
    with np.load(root / "outputs" / "v16_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    reports = []
    configurations = [
        (outcome_k, selection_k)
        for outcome_k in (5.0, 10.0, 20.0, 50.0)
        for selection_k in (100.0, 300.0, 600.0)
    ]
    signals = {}
    for year in (2023, 2024):
        source = history.loc[history["season"].lt(year)]
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        proxy = data.loc[data["season"].eq(year - 1)].reset_index(drop=True)
        for outcome_k, selection_k in configurations:
            raw = failure_signal(source, rows, outcome_k, selection_k)
            proxy_raw = failure_signal(source, proxy, outcome_k, selection_k)
            center = float(proxy_raw[proxy["game_type"].eq("R")].mean())
            signal = np.zeros(len(rows), dtype=np.float64)
            regular = rows["game_type"].eq("R").to_numpy()
            signal[regular] = raw[regular] - center
            signals[(year, outcome_k, selection_k)] = (signal, center)
        print(f"Prepared fine signals for {year}: history={len(source)}", flush=True)
    for outcome_k, selection_k in configurations:
        for outer_weight in np.arange(0.25, 2.001, 0.25):
            gains, halves = {}, []
            centers = {}
            for year in (2023, 2024):
                mask = oof["season"] == year
                y = oof["target"][mask].astype(float)
                base = oof["blended"][mask].astype(float)
                signal, center = signals[(year, outcome_k, selection_k)]
                prediction = np.clip(base + outer_weight * signal, .005, .995)
                half = len(y) // 2
                values = [
                    bss(y, prediction) - bss(y, base),
                    bss(y[:half], prediction[:half]) - bss(y[:half], base[:half]),
                    bss(y[half:], prediction[half:]) - bss(y[half:], base[half:]),
                ]
                gains[str(year)] = values
                halves.extend(values[1:])
                centers[str(year)] = center
            reports.append({
                "outcome_k": outcome_k, "selection_k": selection_k,
                "outer_weight": float(outer_weight),
                "gain_2023": gains["2023"][0], "gain_2024": gains["2024"][0],
                "min_year": min(gains["2023"][0], gains["2024"][0]),
                "min_half": min(halves), "centers": centers,
            })
    reports.sort(
        key=lambda row: (row["min_year"], row["min_half"], row["gain_2024"]),
        reverse=True,
    )
    output = root / "research" / "trackman_failure_prior.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "aligned_rows": len(history),
        "aligned_regular_coverage": len(history) / data["game_type"].eq("R").sum(),
        "fine_pitch_rates": history["pitch_type_fine"].value_counts(
            normalize=True
        ).to_dict(),
        "failure_weights": dict(zip(FAILURES, FAILURE_WEIGHTS.tolist())),
        "reports": reports,
    }, indent=2), encoding="utf-8")
    print(json.dumps(reports[:50], indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
