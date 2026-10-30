"""
LightGBM 5-Fold Stratified Pipeline with Domain Feature Engineering (EXP-003)
Evaluates out-of-fold ROC-AUC and feature importances.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_engineering import create_features

DATA_DIR = "/home/franck-alpha/Bureau/research/zindi/financial-stress-prediction-challenge/data"
SUB_DIR = "/home/franck-alpha/Bureau/research/zindi/financial-stress-prediction-challenge/submissions"
os.makedirs(SUB_DIR, exist_ok=True)

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
RANDOM_STATE = 42

print("Loading raw data...")
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))

print("Generating domain financial features...")
train = create_features(train_raw)
test = create_features(test_raw)

feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]
cat_cols = [c for c in feature_cols if train[c].dtype in ['object', 'str', 'string', 'category']]

print(f"Total features with engineering: {len(feature_cols)}, Categoricals: {cat_cols}")

# Convert categoricals to 'category' dtype
for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

X = train[feature_cols]
y = train[TARGET].values
X_test = test[feature_cols]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_aucs = []
feature_importances = np.zeros(len(feature_cols))

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

print("\nStarting 5-Fold Stratified Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_val = X.iloc[val_idx], y[val_idx]
    
    trn_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_va, label=y_val, reference=trn_data)
    
    model = lgb.train(
        lgb_params,
        trn_data,
        num_boost_round=1500,
        valid_sets=[trn_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    val_probs = model.predict(X_va, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_probs
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    feature_importances += model.feature_importance(importance_type='gain') / skf.n_splits
    
    fold_auc = roc_auc_score(y_val, val_probs)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold+1} ROC-AUC: {fold_auc:.5f} (Best Iter: {model.best_iteration})")

mean_auc = np.mean(fold_aucs)
oof_auc = roc_auc_score(y, oof_preds)
print(f"\n==========================================")
print(f"EXP-003 Mean Fold ROC-AUC: {mean_auc:.5f}")
print(f"EXP-003 Overall OOF ROC-AUC: {oof_auc:.5f}")
print(f"==========================================")

# Top 20 Most Important Features
fi_df = pd.DataFrame({'feature': feature_cols, 'importance': feature_importances}).sort_values('importance', ascending=False)
print("\nTop 20 Features by Gain:")
print(fi_df.head(20).to_string(index=False))

# Save submission
submission = sample_sub.copy()
submission["Target"] = test_preds
sub_path = os.path.join(SUB_DIR, "submission_003_lgbm_fe.csv")
submission.to_csv(sub_path, index=False)
print(f"\nSaved submission to: {sub_path}")
print("Submission stats:")
print(submission["Target"].describe())

# Save OOF predictions for future ensembling
oof_path = os.path.join(SUB_DIR, "oof_003_lgbm_fe.npy")
np.save(oof_path, oof_preds)
print(f"Saved OOF predictions to: {oof_path}")
