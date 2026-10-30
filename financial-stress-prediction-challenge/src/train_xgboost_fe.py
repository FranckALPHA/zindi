"""
XGBoost 5-Fold Stratified Pipeline with Domain Feature Engineering
Evaluates out-of-fold ROC-AUC using native categorical support and histogram tree method.
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from feature_engineering import create_features

DATA_DIR = "/home/franck-alpha/Bureau/research/zindi/financial-stress-prediction-challenge/data"
SUB_DIR = "/home/franck-alpha/Bureau/research/zindi/financial-stress-prediction-challenge/submissions"
os.makedirs(SUB_DIR, exist_ok=True)

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
RANDOM_STATE = 42

print("Loading raw data for XGBoost...")
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))

print("Generating domain financial features...")
train = create_features(train_raw)
test = create_features(test_raw)

cat_cols = ['gender', 'region', 'smartphone', 'segment', 'earning_pattern']
for col in cat_cols:
    train[col] = train[col].astype('category')
    test[col] = test[col].astype('category')

feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]

X = train[feature_cols]
y = train[TARGET].values
X_test = test[feature_cols]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))
fold_aucs = []

xgb_params = {
    'n_estimators': 1200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'eval_metric': 'auc',
    'enable_categorical': True,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

print("\nStarting XGBoost 5-Fold Stratified Cross-Validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    X_va, y_val = X.iloc[val_idx], y[val_idx]
    
    model = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        verbose=100
    )
    
    val_probs = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = val_probs
    test_preds += model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    fold_auc = roc_auc_score(y_val, val_probs)
    fold_aucs.append(fold_auc)
    print(f"XGBoost Fold {fold+1} ROC-AUC: {fold_auc:.5f} (Best Iter: {model.best_iteration})")

mean_auc = np.mean(fold_aucs)
oof_auc = roc_auc_score(y, oof_preds)
print(f"\n==========================================")
print(f"XGBoost Mean Fold ROC-AUC: {mean_auc:.5f}")
print(f"XGBoost Overall OOF ROC-AUC: {oof_auc:.5f}")
print(f"==========================================")

# Save submission
submission = sample_sub.copy()
submission["Target"] = test_preds
sub_path = os.path.join(SUB_DIR, "submission_xgb_fe.csv")
submission.to_csv(sub_path, index=False)
print(f"\nSaved XGBoost submission to: {sub_path}")

# Save OOF predictions
oof_path = os.path.join(SUB_DIR, "oof_xgb_fe.npy")
np.save(oof_path, oof_preds)
print(f"Saved OOF predictions to: {oof_path}")
