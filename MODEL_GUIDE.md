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

## Recommended BSS ensemble (v17)

The competition model is trained with rolling 2023/2024 validation and directly
selects the blend by normalized Brier score:

```bash
bash run_v17.sh
```

The one-command runner trains the three-seed Trackman context specialist on top of
v16, builds `submission_v17.zip`, saves the
training log and OOF diagnostics, and bundles the artifacts as
`outputs/results_v17.zip`. V16 must have been built once on the server, and the
official `data/trackman_history.csv` plus `outputs/trackman_pitch_alignment.npz`
must be available.

The trainer writes native LightGBM/CatBoost models and JSON metadata under
`submit/model/`. V14 adds a CatBoost component with a three-year sample-weight
half-life; it improved five-block 2024 blend validation by 5.49 BSS. V15 applies
the same weighting to separate native-categorical count specialists, improving
the updated five-block score by another 3.18 BSS. V16 recovers historical coarse
pitch type and detailed failure labels from the next as-of counter increment for
99.793% of training rows, then freezes a pitcher/hand/count pitch-choice failure
prior. Its centering constant comes from a training proxy, never from evaluation
rows. V17 reliably aligns 88.30% of regular-season pitches to the supplied
Trackman history. It freezes pitcher-by-count and pitcher-by-batter-hand repertoire
deviations from 2019-2024, and blends a numeric CatBoost specialist only on regular
season rows. Single-seed rolling gains were +75.85 in 2023 and +18.43 in 2024;
all four half-season slices improved. V13's
empirical-Bayes main/context effects build final
tables from the most recent 2024 OOF residuals; this beat the two-season source in
the three latest rolling transfers. Deterministic table geometry reshapes the batter
effect and adds player-exposure directions. The final inference remains a frozen,
row-independent lookup using only the current row's keys. Diagnostics are saved to
`outputs/v17_oof_predictions.npz`.
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
- Anonymous and Trackman IDs are linked only through high-confidence historical
  game/pitch sequence alignment. The submission stores frozen aggregate tables;
  it never joins the current evaluation pitch to Trackman or aggregates test rows.
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
