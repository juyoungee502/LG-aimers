"""Audit the reproducible public hierarchical-stack OOF against v64.

The public repository reports a later 1146.3952 leaderboard result but keeps
the newest correction parameters private.  Its preceding hierarchical stack,
including strict 2023/2024 OOF predictions and training code, is public.  This
script uses those OOF predictions only as a research comparator.  No external
prediction or model is copied into the submission bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_v25_temporal_portfolio import bss


ROOT = Path(__file__).resolve().parent
SCALES = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125,
          0.15, 0.175, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75)
CLIP = (0.005, 0.995)


def gain(y: np.ndarray, base: np.ndarray, candidate: np.ndarray) -> float:
    return float(bss(y, candidate) - bss(y, base))


def cluster_bootstrap(
    y: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    pitcher: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    reference = float(y.mean() * (1.0 - y.mean()))
    row_gain = np.square(base - y) - np.square(candidate - y)
    grouped = pd.DataFrame({
        "pitcher": pitcher.astype(str), "gain": row_gain,
    }).groupby("pitcher", sort=False)["gain"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 64):
        count = min(64, repetitions - start)
        sampled = rng.integers(0, len(grouped), size=(count, len(grouped)))
        values[start:start + count] = (
            100_000.0 * sums[sampled].sum(axis=1)
            / sizes[sampled].sum(axis=1) / reference
        )
    return {
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "positive_probability": float(np.mean(values > 0.0)),
    }


def load_public(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    anchors = path / "evaluation" / "anchors"
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("hierarchical_stack", "adaptive_gate", "psych_latent", "psych_regime_film"):
        source = anchors / f"{name}.npz"
        if not source.is_file():
            continue
        with np.load(source, allow_pickle=True) as archive:
            output[name] = (
                archive["p23"].astype(float), archive["p24"].astype(float),
            )
    if not output:
        raise FileNotFoundError(f"No public anchor OOF files under {anchors}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-repo", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    with np.load(ROOT / "outputs/v64_oof_predictions.npz", allow_pickle=True) as archive:
        y = archive["target"].astype(float)
        base = archive["blended"].astype(float)
        season = archive["season"].astype(int)
    raw = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=["season", "game_type", "pitcher_id"],
        encoding="utf-8-sig", low_memory=False,
    )
    rows = pd.concat([
        raw.loc[raw["season"].eq(year)] for year in (2023, 2024)
    ], ignore_index=True)
    if len(rows) != len(y) or not np.array_equal(rows["season"].to_numpy(int), season):
        raise ValueError("v64 OOF and official rows are not aligned")

    public = load_public(args.public_repo.resolve())
    masks = {year: season == year for year in (2023, 2024)}
    base23, base24 = base[masks[2023]], base[masks[2024]]
    # Isolate the public residual channels as well as testing the complete
    # public anchor.  A channel is represented as v64 plus the public delta so
    # the same scale audit below remains valid.
    if {"adaptive_gate", "psych_latent"}.issubset(public):
        adaptive23, adaptive24 = public["adaptive_gate"]
        latent23, latent24 = public["psych_latent"]
        public["psych_latent_channel_only"] = (
            base23 + latent23 - adaptive23,
            base24 + latent24 - adaptive24,
        )
    if {"adaptive_gate", "psych_regime_film"}.issubset(public):
        adaptive23, adaptive24 = public["adaptive_gate"]
        film23, film24 = public["psych_regime_film"]
        public["psych_regime_channel_only"] = (
            base23 + film23 - adaptive23,
            base24 + film24 - adaptive24,
        )
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    pitcher = rows["pitcher_id"].to_numpy()
    reports: list[dict[str, object]] = []

    alignment_source = (
        args.public_repo / "evaluation" / "anchors" / "hierarchical_stack.npz"
    )
    with np.load(alignment_source, allow_pickle=True) as archive:
        public_y = np.concatenate([
            archive["y23"].astype(float), archive["y24"].astype(float),
        ])
    if not np.array_equal(public_y, y):
        raise ValueError("public OOF target order is not aligned")

    for name, (p23, p24) in public.items():
        public_prediction = np.concatenate([p23, p24])
        if len(public_prediction) != len(y):
            raise ValueError(f"{name} OOF length differs from official rows")
        # The public archives also contain their targets.  Verify exact row
        # alignment before using the predictions as a research comparator.
        candidates: list[dict[str, object]] = []
        for r_scale in SCALES:
            for f_scale in SCALES:
                row_scale = np.where(regular, r_scale, f_scale)
                prediction = np.clip(
                    base + row_scale * (public_prediction - base), *CLIP,
                )
                years: dict[str, object] = {}
                preliminary = True
                for year in (2023, 2024):
                    active = masks[year]
                    positions = np.flatnonzero(active)
                    halves = np.array_split(positions, 2)
                    group_gain = {
                        label: gain(y[active & group], base[active & group], prediction[active & group])
                        for label, group in (("R", regular), ("F", ~regular))
                    }
                    row = {
                        "gain": gain(y[active], base[active], prediction[active]),
                        "half_gains": [gain(y[i], base[i], prediction[i]) for i in halves],
                        "group_gains": group_gain,
                    }
                    years[str(year)] = row
                    preliminary &= bool(
                        row["gain"] > 0.0
                        and min(row["half_gains"]) >= 0.0
                        and min(group_gain.values()) >= 0.0
                    )
                candidates.append({
                    "r_scale": r_scale,
                    "f_scale": f_scale,
                    "years": years,
                    "preliminary_gate": preliminary,
                    "mean_gain": float(np.mean([
                        years["2023"]["gain"], years["2024"]["gain"],
                    ])),
                    "min_gain": float(min(
                        years["2023"]["gain"], years["2024"]["gain"],
                    )),
                })
        candidates.sort(
            key=lambda row: (
                row["preliminary_gate"], row["min_gain"], row["mean_gain"],
            ), reverse=True,
        )
        best = candidates[0]
        row_scale = np.where(regular, best["r_scale"], best["f_scale"])
        prediction = np.clip(
            base + row_scale * (public_prediction - base), *CLIP,
        )
        boot = {
            str(year): cluster_bootstrap(
                y[masks[year]], base[masks[year]], prediction[masks[year]],
                pitcher[masks[year]], args.bootstrap, 651146 + year,
            ) for year in (2023, 2024)
        }
        strict = bool(
            best["preliminary_gate"]
            and min(item["ci_low"] for item in boot.values()) > 0.0
        )
        reports.append({
            "public_anchor": name,
            "public_anchor_scores": {
                str(year): float(bss(y[masks[year]], public_prediction[masks[year]]))
                for year in (2023, 2024)
            },
            "best": best,
            "bootstrap": boot,
            "strict_gate": strict,
        })

    reports.sort(
        key=lambda row: (
            row["strict_gate"], row["best"]["min_gain"], row["best"]["mean_gain"],
        ), reverse=True,
    )
    report = {
        "baseline": "v64_public_method_transfer",
        "public_source": "calico-cat17/LG-Aimers-9th",
        "reported_later_score": 1146.3952,
        "latest_parameters_public": False,
        "anchors": reports,
        "selected": reports[0] if reports[0]["strict_gate"] else None,
        "rules": {
            "external_oof_used_for_research_only": True,
            "external_prediction_or_model_for_submission": False,
            "official_target_alignment_verified": True,
            "forbidden_2025_trackman_used": False,
            "test_row_aggregation_used": False,
            "v62_or_v63_component_used": False,
        },
    }
    (ROOT / "research").mkdir(exist_ok=True)
    path = ROOT / "research/v65_public_1146_structure.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
