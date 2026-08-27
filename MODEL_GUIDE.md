# Model guide

## Current candidate: v26

V26 starts from the v24 command/resolution model and replaces the v25 residual
policy with a higher-gain Pareto-robust temporal portfolio. Its R/F policies
were selected under chronological, half-season, quarter, and monthly
constraints rather than a single 2024 score. V25 remains available as a more
conservative fallback; the two residual policies are never stacked.

Exact forward validation against v24:

| Metric | v24 | v26 | Gain |
|---|---:|---:|---:|
| 2024 OOF BSS | 1007.70 | 1043.74 | +36.04 |
| R regime BSS | - | - | +24.12 |
| F regime BSS | - | - | +125.49 |

The weakest of all audited slices is still positive: +5.00 BSS for R and
+10.83 BSS for F. Inference uses only the current evaluation row and tables
frozen from 2024 training data. It uses no 2025 Trackman data, no leaderboard-
derived calibration, and no aggregation across evaluation rows.

## GPU server: train, validate, and package

The v24 artifacts must already exist. If `outputs/v24_oof_predictions.npz` is
missing, the runner invokes `run_v24.sh` first. Otherwise v26 only freezes and
validates its residual tables, so the v26 step itself does not require a GPU.

Run this one command sequence on the server:

```bash
cd ~/바탕화면/LG-aimers
git pull --ff-only origin experiment/junseo-catboost-gpu
source .venv/bin/activate
bash run_v26.sh
```

This creates:

- `submission_v26.zip`: code-submission ZIP
- `outputs/v26_oof_predictions.npz`: OOF diagnostics
- `training_v26.log`: complete validation/build log
- `outputs/results_v26.zip`: all three files bundled together

Copy the result bundle to the PC from PowerShell:

```powershell
scp "JunseoPark@sia-com3:~/바탕화면/LG-aimers/outputs/results_v26.zip" .
```

Submit `submission_v26.zip` from inside the result bundle.

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
- Every v26 lookup is frozen in `metadata.json`; test-row frequency, grouping,
  rolling statistics, and target encoding over the test set are prohibited.

## Validation artifacts

`train_v26_pareto_portfolio.py` reproduces all four forward-transfer audits,
checks the strict minimum-gain gates, freezes 2024 tables for deployment, and
writes the complete report into `submit/model/metadata.json`. It fails before
packaging if R falls below +4.9 or F below +10.0 on any audited slice.

Earlier checkpoints remain useful references: v17 scored 1076 and v23 scored
1105 on the public leaderboard. V24 was not submitted when v25 was developed.

## Research basis

- Prokhorenkova et al., *CatBoost: unbiased boosting with categorical features*,
  NeurIPS 2018:
  https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- Sidle and Tran, *Using multi-class classification methods to predict baseball
  pitch types*, Journal of Sports Analytics 2018. Count-dependent pitching
  behavior motivates explicit count and pitcher-by-state effects.
