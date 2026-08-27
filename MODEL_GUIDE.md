# Model guide

## Current candidate: v25

V25 starts from the v24 command/resolution model and adds a frozen temporal
residual portfolio. Its R/F policies were selected under 61 chronological,
half-season, quarter, and monthly constraints rather than a single 2024 score.
They combine strongly shrunk one-dimensional tables, state-context tables, and
one conservative F inning-calibration table.

Exact forward validation against v24:

| Metric | v24 | v25 | Gain |
|---|---:|---:|---:|
| 2024 OOF BSS | 1007.70 | 1030.70 | +23.00 |
| R regime BSS | - | - | +14.97 |
| F regime BSS | - | - | +83.24 |

The weakest of all audited slices is still positive: +7.15 BSS for R and
+20.66 BSS for F. Inference uses only the current evaluation row and tables
frozen from 2024 training data. It uses no 2025 Trackman data, no leaderboard-
derived calibration, and no aggregation across evaluation rows.

## GPU server: train, validate, and package

The v24 artifacts must already exist. If `outputs/v24_oof_predictions.npz` is
missing, the runner invokes `run_v24.sh` first. Otherwise v25 only freezes and
validates its residual tables, so the v25 step itself does not require a GPU.

Run this one command sequence on the server:

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v25.sh
```

This creates:

- `submission_v25.zip`: code-submission ZIP
- `outputs/v25_oof_predictions.npz`: OOF diagnostics
- `training_v25.log`: complete validation/build log
- `outputs/results_v25.zip`: all three files bundled together

Copy the result bundle to the PC from PowerShell:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v25.zip" .
```

Submit `submission_v25.zip` from inside the result bundle.

## Base training

For the older standalone pipeline:

```powershell
pip install -r requirements.txt
python train.py --preset full --task-type CPU
python script.py
powershell -ExecutionPolicy Bypass -File .\make_submission.ps1
```

Use `--preset fast` only for a pipeline smoke test. On a single-GPU Linux
server, CatBoost training can use:

```bash
python train.py --preset full --task-type GPU --devices 0
```

CatBoost accepts device ranges such as `--devices 0:1` or `--devices 0-3` for
multiple GPUs.

## Design constraints

- Validation is chronological because the target rate shifts materially across
  seasons. The production residual source is the latest allowed season, 2024.
- CatBoost handles the anonymous high-cardinality pitcher and batter IDs with
  ordered categorical statistics.
- Bayesian shrinkage stabilizes season and context rates with small counts.
- State features cover count, handedness, runners, leverage, score, recent
  trends, pitch mix, and season-vs-prior deviations.
- Anonymous and Trackman IDs are linked only through historical 2019-2024 pitch
  sequence alignment. No current evaluation pitch is joined to Trackman.
- Every v25 lookup is frozen in `metadata.json`; test-row frequency, grouping,
  rolling statistics, and target encoding over the test set are prohibited.

## Validation artifacts

`train_v25_temporal_portfolio.py` reproduces all four forward-transfer audits,
checks the strict minimum-gain gates, freezes 2024 tables for deployment, and
writes the complete report into `submit/model/metadata.json`. It fails before
packaging if R falls below +7.0 or F below +20.0 on any audited slice.

Earlier checkpoints remain useful references: v17 scored 1076 and v23 scored
1105 on the public leaderboard. V24 was not submitted when v25 was developed.

## Research basis

- Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*,
  NeurIPS 2018:
  https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- Sidle and Tran, *Using multi-class classification methods to predict baseball
  pitch types*, Journal of Sports Analytics 2018. Count-dependent pitching
  behavior motivates explicit count and pitcher-by-state effects.
