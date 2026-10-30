"""
Official Starter Notebook Pipeline: Logistic Regression on Scaled Features
Evaluated via 5-Fold Stratified K-Fold Cross Validation.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/home/franck-alpha/Bureau/research/zindi/financial-stress-prediction-challenge/data"
SUB_DIR = "/home/franck-alpha/Bureau/research/zindi/financial-stress-prediction-challenge/submissions"
os.makedirs(SUB_DIR, exist_ok=True)

TARGET = "liquidity_stress_next_30d"
ID_COL = "ID"
RANDOM_STATE = 42

print("Loading data...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))

feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]
cat_cols = [c for c in feature_cols if train[c].dtype in ['object', 'str', 'string', 'category']]

print(f"Features: {len(feature_cols)} ({len(cat_cols)} categoricals)")

# One-hot encoding as in starter notebook
X_raw = train[feature_cols].copy()
test_raw = test[feature_cols].copy()

X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
test_encoded = pd.get_dummies(test_raw, columns=cat_cols, drop_first=True)
X_encoded, test_encoded = X_encoded.align(test_encoded, join="left", axis=1, fill_value=0)

y = train[TARGET].values
X = X_encoded.values
X_test = test_encoded.values

print(f"Encoded feature count: {X.shape[1]}")

# 5-Fold Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

fold_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_val = X[val_idx], y[val_idx]
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_va_sc = scaler.transform(X_va)
    X_te_sc = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=200, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
    model.fit(X_tr_sc, y_tr)
    
    val_probs = model.predict_proba(X_va_sc)[:, 1]
    oof_preds[val_idx] = val_probs
    test_preds += model.predict_proba(X_te_sc)[:, 1] / skf.n_splits
    
    fold_auc = roc_auc_score(y_val, val_probs)
    fold_aucs.append(fold_auc)
    print(f"Fold {fold+1} ROC-AUC: {fold_auc:.5f}")

mean_auc = np.mean(fold_aucs)
oof_auc = roc_auc_score(y, oof_preds)
print(f"\nMean Fold ROC-AUC: {mean_auc:.5f}")
print(f"Overall OOF ROC-AUC: {oof_auc:.5f}")

# Save baseline submission
submission = sample_sub.copy()
submission["Target"] = test_preds
sub_path = os.path.join(SUB_DIR, "submission_001_starter_logreg.csv")
submission.to_csv(sub_path, index=False)
print(f"Saved starter submission to: {sub_path}")
print("Submission stats:")
print(submission["Target"].describe())
