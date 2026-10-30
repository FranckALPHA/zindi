# Experiment Log — Liquidity / Financial Stress Prediction Challenge

## Evaluation Strategy
- **Metric:** ROC-AUC on predicted probabilities.
- **Validation Protocol:** 5-fold Stratified K-Fold cross-validation (preserving 15% class ratio across all folds, fixed seed 42).
- **Leakage Prevention:** All scalers, encodings, and feature stats computed inside each fold.

---

## Log Entries

| Exp ID | Model Architecture | Validation ROC-AUC | Public LB Score | Notes / Key Changes |
| :--- | :--- | :--- | :--- | :--- |
| **EXP-001** | Logistic Regression Starter Baseline (5-Fold Stratified) | **0.75536** (PR: 0.376) | **0.313125683** | Official starter benchmark: One-hot + StandardScaler + LogisticRegression(balanced) -> `submissions/submission_001_starter_logreg.csv` |
| **EXP-002** | 5-Fold Stratified LightGBM (Raw Features) | **0.88461** | **0.683977072** | GBDT baseline using native categorical encoding; massive jump (+0.3708 over baseline on LB) -> `submissions/submission_002_lgbm_raw.csv` |
| **EXP-003** | 5-Fold Stratified LightGBM + Feature Engineering | **0.89919** (PR: 0.6815) | **0.708074315** | 🏆 **TOP 1 PUBLIC LB !** Top features: `bal_slope_6m`, `inflow_slope_6m`, `net_cf_slope_6m` -> `submissions/submission_003_lgbm_fe.csv` |
| **EXP-004a**| CatBoost 5-Fold Stratified + Feature Engineering | **0.89731** (PR: 0.6817) | **0.701918328** | CatBoost native categorical trees + longitudinal features -> `submissions/submission_cb_fe.csv` |
| **EXP-004b**| XGBoost 5-Fold Stratified + Feature Engineering | **0.89425** (PR: 0.6667) | **0.698949148** | XGBoost hist tree method + longitudinal features -> `submissions/submission_xgb_fe.csv` |
| **EXP-004c**| Super-Ensemble 3 Modèles (LGBM 56% + CB 38% + XGB 6%) | **0.90099** | **0.707249530** | Stabilité maximale pour le Private LB -> `submissions/submission_004_ensemble.csv` |
