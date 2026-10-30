"""
Ensemble Blender for Liquidity Stress Prediction Challenge (EXP-004)
Optimizes blending weights via Nelder-Mead on Out-Of-Fold ROC-AUC.
Supports probability blending and rank averaging.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

DATA_DIR = "/home/franck-alpha/Bureau/research/zindi/financial-stress-prediction-challenge/data"
SUB_DIR = "/home/franck-alpha/Bureau/research/zindi/financial-stress-prediction-challenge/submissions"

TARGET = "liquidity_stress_next_30d"

print("Loading ground truth target...")
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), usecols=[TARGET])
y_true = train[TARGET].values

models = {}

# Check available OOF files and corresponding test predictions
model_configs = [
    ("LightGBM", "oof_003_lgbm_fe.npy", "submission_003_lgbm_fe.csv"),
    ("CatBoost", "oof_cb_fe.npy", "submission_cb_fe.csv"),
    ("XGBoost", "oof_xgb_fe.npy", "submission_xgb_fe.csv")
]

for name, oof_file, sub_file in model_configs:
    oof_p = os.path.join(SUB_DIR, oof_file)
    sub_p = os.path.join(SUB_DIR, sub_file)
    if os.path.exists(oof_p) and os.path.exists(sub_p):
        oof = np.load(oof_p)
        sub = pd.read_csv(sub_p)["Target"].values
        auc = roc_auc_score(y_true, oof)
        models[name] = {"oof": oof, "test": sub, "auc": auc}
        print(f"Loaded {name}: OOF ROC-AUC = {auc:.5f}")

if len(models) < 2:
    print(f"Not enough models loaded yet (found {len(models)}). Need at least 2.")
    exit(0)

names = list(models.keys())
n_models = len(names)
print(f"\nEnsembling {n_models} models: {names}")

# Matrix of OOFs: shape (N, n_models)
OOF_matrix = np.column_stack([models[m]["oof"] for m in names])
TEST_matrix = np.column_stack([models[m]["test"] for m in names])

# Correlation between predictions
corr_df = pd.DataFrame(OOF_matrix, columns=names).corr()
print("\nPrediction Correlation Matrix:")
print(corr_df.to_string())

# 1. Simple Equal Weight Average
eq_oof = np.mean(OOF_matrix, axis=1)
eq_auc = roc_auc_score(y_true, eq_oof)
print(f"\nEqual Weights OOF ROC-AUC: {eq_auc:.5f}")

# 2. Optimal Probability Weighting via Nelder-Mead
def loss_func(weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    blend = np.dot(OOF_matrix, weights)
    # Return negative ROC-AUC to minimize
    return -roc_auc_score(y_true, blend)

init_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 500})

opt_weights = res.x / np.sum(res.x)
opt_blend_oof = np.dot(OOF_matrix, opt_weights)
opt_auc = roc_auc_score(y_true, opt_blend_oof)

print(f"\nOptimal Weighted OOF ROC-AUC: {opt_auc:.5f}")
for m, w in zip(names, opt_weights):
    print(f"  Weight {m}: {w:.4f}")

# 3. Rank Averaging
RANK_OOF_matrix = np.column_stack([rankdata(models[m]["oof"]) / len(y_true) for m in names])
rank_oof = np.mean(RANK_OOF_matrix, axis=1)
rank_auc = roc_auc_score(y_true, rank_oof)
print(f"Rank Averaging OOF ROC-AUC: {rank_auc:.5f}")

# Select best blend (probability blend preserves raw probability distribution better for Zindi)
best_test_preds = np.dot(TEST_matrix, opt_weights)

# Save ensemble submission
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "SampleSubmission.csv"))
sample_sub["Target"] = best_test_preds
out_sub = os.path.join(SUB_DIR, "submission_004_ensemble.csv")
sample_sub.to_csv(out_sub, index=False)
print(f"\nSaved optimal ensemble submission to: {out_sub}")
print("Ensemble submission stats:")
print(sample_sub["Target"].describe())
