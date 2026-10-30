# AI4EAC Liquidity Stress / Financial Stress Prediction Challenge

## 1. Challenge Overview
- **Goal:** Predict whether a mobile money customer is likely to experience liquidity stress in the next 30 days (`liquidity_stress_next_30d`), using 6 months of historical mobile money behavioral and demographic data.
- **Platform:** Zindi
- **Evaluation Metric:** ROC-AUC (Probabilities between 0 and 1)
- **Target:** `liquidity_stress_next_30d` (binary: 0 = No stress, 1 = Experiencing liquidity stress)
- **Class Distribution:** Highly imbalanced: 85.0% class 0 (34,000), 15.0% class 1 (6,000).
- **Public / Private Leaderboard Split:** 30% Public / 70% Private (or 20% / 80%).
- **Daily Submission Limit:** 10 submissions/day, 200 overall limit.

## 2. Official Rules & Constraints
1. **Raw Probabilities Required:** Rule states: *"If the error metric requires probabilities to be submitted, do not set thresholds (or round your probabilities) to improve your place on the leaderboard."*
2. **Open-Source Tools Only:** No AutoML tools (auto-sklearn, FLAML, etc.). Standard packages (LightGBM, XGBoost, CatBoost, Scikit-Learn) allowed.
3. **Reproducibility:** Top 10 code review within 48h. Fixed random seeds and deterministic pipelines mandatory.

## 3. Data Structure
- `train.csv`: 40,000 rows, 184 columns (ID + 182 features + target).
- `test.csv`: 30,000 rows, 183 columns (ID + 182 features).
- `SampleSubmission.csv`: 30,000 rows, 2 columns (`ID`, `Target`).
- Missing values: 0 across all columns in train and test!
- Features:
  - 5 Categoricals: `gender`, `region`, `smartphone`, `segment`, `earning_pattern`
  - 2 Demographics/Profile: `arpu`, `age`
  - 1 Activity: `x_90_d_activity_rate`
  - 174 Transactional & Balance features: 29 metrics recorded across 6 months (M1 = most recent, M6 = oldest).
