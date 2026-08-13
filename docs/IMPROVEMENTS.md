# How to make this baseline stack stronger

## Protocol (highest ROI)

1. **Subject / patient-wise splits** — replace plain `train_test_split` with `LeaveOneGroupOut` / `StratifiedGroupKFold`. Current Klinik numbers are optimistic.
2. **Frozen eval recipe** — one YAML per dataset: window length, filter band, sfreq, feature set, seed, metrics (acc / macro-F1 / AUROC).
3. **Same recipe as SSL repo** — identical windows + labels so embedding transfer is comparable.

## Features & models

4. CSP + band-power stack on BCI IV-2a (standard MI baseline).
5. Hyperparameter search with nested CV (outer subject CV, inner model select).
6. Calibration + threshold sweep for seizure / Klinik (not only argmax accuracy).

## Engineering

7. One CLI entry (`scripts/run_experiment.py`) → CSV/JSON under `results/runs/`.
8. Drop notebook-duplicated class defs; notebook only calls package.
9. Unit tests on transforms (PSD shape, WTE dim, no NaN).
10. Remove optional wandb hard-dep; make logging sink pluggable.

## Ambition upgrades

11. Public leaderboard table: Klinik / BCI2a / LEE / CHB-MIT with LOSO.
12. Leakage audit script (trial overlap, patient ID in both splits).
13. Tiny paper-style card: task, split, metric, seed, commit hash.
