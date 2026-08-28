"""Validate trained artifacts and build the code-submission ZIP."""
from __future__ import annotations
import argparse
import json
import zipfile
from pathlib import Path

REQUIRED_MODELS = (
    "metadata.json", "lgb_0.txt", "lgb_1.txt",
    "catboost_0.cbm", "catboost_1.cbm", "catboost_2.cbm",
    "catboost_other_0.cbm", "catboost_other_1.cbm", "catboost_other_2.cbm",
    "catboost_two_strike_0.cbm", "catboost_two_strike_1.cbm", "catboost_two_strike_2.cbm",
    "catboost_categorical_0.cbm", "catboost_categorical_1.cbm", "catboost_categorical_2.cbm",
    "catboost_categorical_other_0.cbm", "catboost_categorical_other_1.cbm", "catboost_categorical_other_2.cbm",
    "catboost_categorical_two_strike_0.cbm", "catboost_categorical_two_strike_1.cbm", "catboost_categorical_two_strike_2.cbm",
    "catboost_brier_0.cbm", "catboost_brier_1.cbm", "catboost_brier_2.cbm",
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submit-dir", default="submit")
    p.add_argument("--output", default="submission_v17.zip")
    p.add_argument("--expected-version", default="v17_trackman_context")
    args = p.parse_args(); root = Path(args.submit_dir)
    base_version = (
        "v54_roster_robust_command"
        if args.expected_version in (
            "v55_v54_regime_scaling", "v56_v54_regime_scaling",
        )
        else args.expected_version
    )
    model_names = list(REQUIRED_MODELS)
    if base_version in (
        "v14_weighted_catboost", "v15_weighted_categorical_specialist",
        "v16_pitch_failure_prior",
        "v17_trackman_context",
        "v18_f_regime",
        "v19_failure_specialist",
        "v20_residual_portfolio",
        "v21_robust_residual_portfolio",
        "v22_component_residual_portfolio",
        "v23_probability_residual_portfolio",
        "v24_robust_command_resolution",
        "v25_strict_temporal_portfolio",
        "v26_pareto_temporal_portfolio",
        "v38_lowcard_ensemble",
        "v54_roster_robust_command",
    ):
        model_names.extend(f"catboost_weighted_{index}.cbm" for index in range(3))
    if base_version in (
        "v15_weighted_categorical_specialist", "v16_pitch_failure_prior",
        "v17_trackman_context",
        "v18_f_regime",
        "v19_failure_specialist",
        "v20_residual_portfolio",
        "v21_robust_residual_portfolio",
        "v22_component_residual_portfolio",
        "v23_probability_residual_portfolio",
        "v24_robust_command_resolution",
        "v25_strict_temporal_portfolio",
        "v26_pareto_temporal_portfolio",
        "v38_lowcard_ensemble",
        "v54_roster_robust_command",
    ):
        model_names.extend(
            f"catboost_weighted_categorical_{label}_{index}.cbm"
            for label in ("other", "two_strike") for index in range(3)
        )
    if base_version in ("v17_trackman_context", "v18_f_regime", "v19_failure_specialist", "v20_residual_portfolio", "v21_robust_residual_portfolio", "v22_component_residual_portfolio", "v23_probability_residual_portfolio", "v24_robust_command_resolution", "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble", "v54_roster_robust_command"):
        model_names.extend(
            f"catboost_trackman_context_{index}.cbm" for index in range(3)
        )
    if base_version in ("v18_f_regime", "v19_failure_specialist", "v20_residual_portfolio", "v21_robust_residual_portfolio", "v22_component_residual_portfolio", "v23_probability_residual_portfolio", "v24_robust_command_resolution", "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble", "v54_roster_robust_command"):
        model_names.extend(f"catboost_f_regime_{index}.cbm" for index in range(3))
    if base_version in ("v19_failure_specialist", "v20_residual_portfolio", "v21_robust_residual_portfolio", "v22_component_residual_portfolio", "v23_probability_residual_portfolio", "v24_robust_command_resolution", "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble", "v54_roster_robust_command"):
        model_names.extend(
            f"catboost_failure_{label}.cbm"
            for label in ("reverse", "middle", "wayoff")
        )
    if base_version in ("v24_robust_command_resolution", "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble", "v54_roster_robust_command"):
        model_names.extend(
            f"catboost_v24_command_{label}_{index}.cbm"
            for label in ("no_month", "full", "recent") for index in range(3)
        )
        model_names.extend(
            f"catboost_v24_resolution_{mode}_{index}.cbm"
            for mode in (
                "regime_count", "regime_count_hands", "regime_count_runners",
            ) for index in range(3)
        )
    if base_version in ("v38_lowcard_ensemble", "v54_roster_robust_command"):
        model_names.extend(
            f"catboost_v38_failure_{label}.cbm"
            for label in ("reverse", "middle", "wayoff")
        )
        model_names.extend(
            f"catboost_v38_direct_{index}.cbm" for index in range(3)
        )
    if base_version == "v54_roster_robust_command":
        model_names.extend((
            "catboost_v54_command.cbm", "catboost_v54_overlap.cbm",
        ))
        model_names.extend(
            f"catboost_v54_recent_{index}.cbm" for index in range(6)
        )
        model_names.extend(
            f"catboost_v54_joint_{index}.cbm" for index in range(3)
        )
    required = [
        root / "script.py", root / "requirements.txt", Path("feature_engineering.py"),
        Path("residual_effects.py"),
    ] + [root / "model" / n for n in model_names]
    if base_version in ("v17_trackman_context", "v18_f_regime", "v19_failure_specialist", "v20_residual_portfolio", "v21_robust_residual_portfolio", "v22_component_residual_portfolio", "v23_probability_residual_portfolio", "v24_robust_command_resolution", "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble", "v54_roster_robust_command"):
        required.append(Path("trackman_context.py"))
    if base_version in ("v19_failure_specialist", "v20_residual_portfolio", "v21_robust_residual_portfolio", "v22_component_residual_portfolio", "v23_probability_residual_portfolio", "v24_robust_command_resolution", "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble", "v54_roster_robust_command"):
        required.append(Path("failure_context.py"))
    if base_version in ("v20_residual_portfolio", "v21_robust_residual_portfolio", "v22_component_residual_portfolio", "v23_probability_residual_portfolio", "v24_robust_command_resolution", "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble", "v54_roster_robust_command"):
        required.append(Path("residual_portfolio.py"))
    if base_version in ("v22_component_residual_portfolio", "v23_probability_residual_portfolio", "v24_robust_command_resolution", "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble", "v54_roster_robust_command"):
        required.append(Path("component_residual_portfolio.py"))
    if base_version in ("v23_probability_residual_portfolio", "v24_robust_command_resolution", "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble", "v54_roster_robust_command"):
        required.append(Path("probability_residual_portfolio.py"))
    if base_version in ("v24_robust_command_resolution", "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio", "v38_lowcard_ensemble", "v54_roster_robust_command"):
        required.append(Path("v24_robust_candidate.py"))
    if base_version in (
        "v25_strict_temporal_portfolio", "v26_pareto_temporal_portfolio",
        "v38_lowcard_ensemble",
        "v54_roster_robust_command",
    ):
        required.append(Path("v25_temporal_portfolio.py"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(f"Missing submission files: {missing}")
    metadata = json.loads((root / "model" / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("version") != args.expected_version:
        raise ValueError(f"Unexpected model version: {metadata.get('version')}")
    output = Path(args.output)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in required:
            arcname = path.name if path in (
                Path("feature_engineering.py"), Path("residual_effects.py"),
                Path("trackman_context.py"), Path("failure_context.py"),
                Path("residual_portfolio.py"),
                Path("component_residual_portfolio.py"),
                Path("probability_residual_portfolio.py"),
                Path("v24_robust_candidate.py"),
                Path("v25_temporal_portfolio.py"),
            ) else path.relative_to(root).as_posix()
            archive.write(path, arcname)
    print(f"Created {output.resolve()} ({output.stat().st_size / 1024**2:.2f} MiB)")
    with zipfile.ZipFile(output) as archive:
        print("Contents:", archive.namelist())

if __name__ == "__main__": main()
