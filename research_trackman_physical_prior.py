"""Screen physical pitch-archetype failure priors over the v16 baseline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler

from research_inferred_pitch_priors import bss, reconstruct_labels
from research_trackman_failure_prior import FAILURES, selection_delta


TARGET = "control_success"
PHYSICAL_FEATURES = (
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side", "zone_speed",
)
WEIGHT_VECTORS = {
    "safe": np.array([-.50, 1.00, -.40]),
    "strong": np.array([-.75, 1.50, -.50]),
    "ridge": np.array([-.60, .90, -.60]),
}


def physical_history(root, data, labels):
    with np.load(root / "outputs" / "trackman_pitch_alignment.npz") as aligned:
        links = pd.DataFrame({
            "row_id": aligned["row_id"].astype(str),
            "trackman_id": aligned["trackman_id"].astype(str),
        })
    trackman = pd.read_csv(
        root / "data" / "trackman_history.csv",
        usecols=["trackman_id", "pitcher_hand", *PHYSICAL_FEATURES],
        encoding="utf-8-sig", low_memory=False,
    )
    trackman["trackman_id"] = trackman["trackman_id"].astype(str)
    direction = np.where(trackman["pitcher_hand"].eq("Left"), -1.0, 1.0)
    trackman["horz_break"] *= direction
    trackman["rel_side"] *= direction
    links = links.merge(
        trackman.drop(columns="pitcher_hand"), on="trackman_id", how="left",
        validate="one_to_one",
    )
    history = data[[
        "row_id", "season", "pitcher_id", "batter_hand", "balls_before",
        "strikes_before", "game_type", TARGET,
    ]].copy()
    history["row_id"] = history["row_id"].astype(str)
    for failure in FAILURES:
        history[failure] = labels[failure].to_numpy(np.float32)
    history = history.merge(
        links.drop(columns="trackman_id"), on="row_id", how="inner",
        validate="one_to_one",
    )
    history["count_state"] = (
        history["balls_before"] * 3 + history["strikes_before"]
    ).astype(np.int8)
    return history.dropna(subset=[*PHYSICAL_FEATURES, *FAILURES])


def assign_archetypes(history, clusters, seed=20260814):
    values = history[list(PHYSICAL_FEATURES)].to_numpy(np.float32)
    scaler = RobustScaler(quantile_range=(10.0, 90.0)).fit(values)
    scaled = scaler.transform(values)
    model = MiniBatchKMeans(
        n_clusters=clusters, random_state=seed, batch_size=8192, n_init=5,
        max_iter=150, reassignment_ratio=.005,
    ).fit(scaled)
    output = history.copy()
    output["physical_archetype"] = np.char.add("a", model.labels_.astype(str))
    return output, {
        "centers": scaler.inverse_transform(model.cluster_centers_).tolist(),
        "counts": np.bincount(model.labels_, minlength=clusters).tolist(),
    }


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    labels = reconstruct_labels(data)
    history = physical_history(root, data, labels)
    with np.load(root / "outputs" / "v16_oof_predictions.npz") as loaded:
        oof = {key: loaded[key] for key in loaded.files}
    signals, cluster_info = {}, {}
    for year in (2023, 2024):
        source = history.loc[history["season"].lt(year)].copy()
        rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
        proxy = data.loc[data["season"].eq(year - 1)].reset_index(drop=True)
        for clusters in (4, 6, 8, 10):
            archetypes, info = assign_archetypes(source, clusters)
            types = tuple(f"a{index}" for index in range(clusters))
            row_components, proxy_components = [], []
            for failure in FAILURES:
                row_components.append(selection_delta(
                    archetypes, rows, failure, 50.0, 300.0,
                    types=types, type_column="physical_archetype",
                ))
                proxy_components.append(selection_delta(
                    archetypes, proxy, failure, 50.0, 300.0,
                    types=types, type_column="physical_archetype",
                ))
            signals[(year, clusters)] = (
                np.column_stack(row_components), np.column_stack(proxy_components),
            )
            cluster_info[f"{year}_{clusters}"] = info
            print(
                f"Prepared physical signals: year={year} clusters={clusters} "
                f"history={len(archetypes)}", flush=True,
            )
    reports = []
    for clusters in (4, 6, 8, 10):
        for weight_name, failure_weights in WEIGHT_VECTORS.items():
            for outer_weight in np.arange(.25, 1.501, .25):
                gains, halves, centers = {}, [], {}
                for year in (2023, 2024):
                    rows = data.loc[data["season"].eq(year)].reset_index(drop=True)
                    proxy = data.loc[data["season"].eq(year - 1)].reset_index(drop=True)
                    components, proxy_components = signals[(year, clusters)]
                    raw = components @ failure_weights
                    proxy_raw = proxy_components @ failure_weights
                    center = float(proxy_raw[proxy["game_type"].eq("R")].mean())
                    correction = np.zeros(len(rows), dtype=np.float64)
                    regular = rows["game_type"].eq("R").to_numpy()
                    correction[regular] = raw[regular] - center
                    mask = oof["season"] == year
                    y = oof["target"][mask].astype(float)
                    base = oof["blended"][mask].astype(float)
                    prediction = np.clip(
                        base + outer_weight * correction, .005, .995,
                    )
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
                    "clusters": clusters, "failure_weights": weight_name,
                    "outer_weight": float(outer_weight),
                    "gain_2023": gains["2023"][0],
                    "gain_2024": gains["2024"][0],
                    "min_year": min(gains["2023"][0], gains["2024"][0]),
                    "min_half": min(halves), "centers": centers,
                })
    reports.sort(
        key=lambda row: (row["min_year"], row["min_half"], row["gain_2024"]),
        reverse=True,
    )
    output = root / "research" / "trackman_physical_prior.json"
    output.write_text(json.dumps({
        "aligned_physical_rows": len(history),
        "physical_features": PHYSICAL_FEATURES,
        "weight_vectors": {k: v.tolist() for k, v in WEIGHT_VECTORS.items()},
        "reports": reports, "clusters": cluster_info,
    }, indent=2), encoding="utf-8")
    print(json.dumps(reports[:50], indent=2), flush=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
