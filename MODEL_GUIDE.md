# Model guide

## Run

```powershell
pip install -r requirements.txt
python train.py --preset full --task-type CPU
python script.py
powershell -ExecutionPolicy Bypass -File .\make_submission.ps1
```

Use `--preset fast` only for a pipeline smoke test. If CatBoost CUDA support is
available, `--task-type GPU` can reduce training time. Training creates
`model/catboost.cbm` and `model/metadata.json`; local inference creates
`output/submission.csv`; the PowerShell script creates `submission.zip`.

On a single-GPU server use:

```bash
python train.py --preset full --task-type GPU --devices 0
```

For multiple GPUs, CatBoost accepts values such as `--devices 0:1` or
`--devices 0-3`.

## Recommended BSS ensemble (v14)

The competition model is trained with rolling 2023/2024 validation and directly
selects the blend by normalized Brier score:

```bash
bash run_v14.sh
```

The one-command runner trains one weighted component, upgrades v13 artifacts, builds
`submission_v14.zip`, saves the training log and OOF diagnostics, and bundles the artifacts as
`outputs/results_v14.zip`. V11 must have been trained once and upgraded through v13 on the
server; v14 trains only three additional CatBoost models for final inference.

The trainer writes native LightGBM/CatBoost models and JSON metadata under
`submit/model/`. V14 adds a CatBoost component with a three-year sample-weight
half-life; it improved five-block 2024 blend validation by 5.49 BSS. V13's
empirical-Bayes main/context effects build final
tables from the most recent 2024 OOF residuals; this beat the two-season source in
the three latest rolling transfers. Deterministic table geometry reshapes the batter
effect and adds player-exposure directions. The final inference remains a frozen,
row-independent lookup using only the current row's keys. Diagnostics are saved to
`outputs/v14_oof_predictions.npz`.
The builder checks the model version and expected file layout before producing
the ZIP.

## Why this design

- The target rate falls from 0.5647 in 2019 to 0.4861 in 2024. Validation is
  therefore chronological (2024 only), and older seasons receive exponentially
  smaller training weights.
- CatBoost natively processes the high-cardinality anonymous pitcher and batter
  IDs. Its ordered categorical statistics were designed to limit target leakage.
- Fixed-prior Bayesian shrinkage stabilizes `asof_*` rates when their sample count
  is small. Missingness and cold-start flags let the model learn fallback rules.
- Baseball context is represented by count, handedness matchup, scoring position,
  leverage, score state, recent-vs-career deltas, and pitch-mix concentration.
- The Trackman player IDs do not map to the main anonymous player IDs. The model
  intentionally avoids an invalid join; the supplied historical pitch-mix features
  already provide safe pre-pitch summaries.
- A Platt sigmoid is fitted only on the future 2024 holdout. Its strength is chosen
  from a conservative grid and disabled automatically if it does not improve
  holdout log loss.

## Validation output

`train.py` prints chronological holdout log loss, AUC, Brier score, target rate,
prediction mean, chosen tree count, and calibration parameters. These metrics are
also stored in `model/metadata.json`.

## Research basis

- Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*,
  NeurIPS 2018: https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- Sidle and Tran, *Using multi-class classification methods to predict baseball
  pitch types*, Journal of Sports Analytics 2018: count-dependent pitching behavior
  motivates explicit count features.
