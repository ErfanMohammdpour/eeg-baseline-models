# Roadmap

## Protocol

1. Move to subject / patient-wise splits with `LeaveOneGroupOut` or `StratifiedGroupKFold`.
2. Freeze one eval recipe per dataset: window length, filter band, sampling rate, feature set, seed, metrics.
3. Match windowing and labels with the SSL companion repo for fair comparison.

## Features and models

4. Add CSP + band-power baselines on BCI IV-2a.
5. Nested CV for model selection under subject-wise outer folds.
6. Calibration and threshold sweeps for seizure / Klinik tasks.

## Engineering

7. One CLI entry (`scripts/run_experiment.py`) writing CSV/JSON under `results/runs/`.
8. Keep notebooks thin: call package APIs only.
9. Unit tests for transforms (PSD shape, WTE dim, no NaN).
10. Optional logging sinks; no hard dependency on a single tracker.

## Ambition

11. Public tables for Klinik / BCI IV-2a / LEE / CHB-MIT with LOSO.
12. Leakage audit script for trial and patient overlap.
13. Short result card: task, split, metric, seed, commit hash.
