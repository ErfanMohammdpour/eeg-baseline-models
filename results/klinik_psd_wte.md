# Klinik · PSD + WTE baselines

| Field | Value |
|-------|-------|
| Dataset | Klinik (binary clinical EEG) |
| Features | PSD + WTE → flatten |
| Feature dim | 2814 |
| Split | stratified train/test 80/20 (~636 / ~159) |
| Source | `notebooks/klinik_psd_wte.ipynb` |

| Model | Test acc | Test F1 macro | Train acc | Train F1 macro |
|-------|----------|---------------|-----------|----------------|
| XGBoost | 0.98742 | 0.98742 | 1.00000 | 1.00000 |
| KNeighborsClassifier | 0.96226 | 0.96214 | 0.96693 | 0.96686 |
| BalancedRandomForestClassifier | 0.93082 | 0.93081 | 0.97642 | 0.97640 |
| SVC (ovo) | 0.91000 | 0.91000 | 0.94000 | 0.94000 |
| GaussianNB | 0.84906 | 0.84833 | 0.81761 | 0.81586 |

Notes:
- SVM row from classification report in notebook (rounded to 2 decimals in print).
- Not subject-wise / LOSO. Re-run with `GroupKFold` / `LeaveOneGroupOut` before claiming clinical SOTA.
