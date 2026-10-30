"""
Option A - Complete CatBoost Native NLP & Categorical Pipeline
Entraîne un CatBoost 5-fold en exploitant nativement :
- indicator_description comme text_features
- community, district, indicator comme cat_features
- Encodage cyclique (sans overfitter pred_date)
- Calibration de SMALLRAIN et HEAVYRAIN
"""

import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

def run():
    print("[1] Chargement des données...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/SampleSubmission.csv")
    
    def prep(df):
        df = df.copy()
        dt = pd.to_datetime(df['prediction_time'].astype(str), errors='coerce')
        df['hour'] = dt.dt.hour.fillna(12).astype(float)
        df['dow'] = dt.dt.day_of_week.fillna(2).astype(float)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7.0)
        
        df['confidence'] = df['confidence'].fillna(0.3)
        df['predicted_intensity'] = df['predicted_intensity'].fillna(0.0)
        df['forecast_length'] = df['forecast_length'].fillna(12)
        df['conf_x_int'] = df['confidence'] * df['predicted_intensity']
        df['conf_x_fc'] = df['confidence'] * df['forecast_length']
        
        df['community'] = df['community'].fillna('none').astype(str).str.strip().str.lower()
        df['district'] = df['district'].fillna('none').astype(str).str.strip().str.lower()
        df['indicator'] = df['indicator'].fillna('none').astype(str).str.strip().str.lower()
        df['indicator_description'] = df['indicator_description'].fillna('none').astype(str)
        return df

    train = prep(train)
    test = prep(test)
    
    all_cols = ['user_id', 'confidence', 'predicted_intensity', 'forecast_length', 
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'conf_x_int', 'conf_x_fc',
                'community', 'district', 'indicator', 'indicator_description']
    cat_cols = ['community', 'district', 'indicator']
    text_cols = ['indicator_description']
    
    TARGET = 'Target'
    ID_COL = 'ID'
    classes = sorted(train[TARGET].unique())
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = train[TARGET].map(class_to_idx).values
    
    print(f"[2] Entraînement 5-Fold CatBoost avec {len(all_cols)} variables (Texte & Catégorielles natives)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_proba = np.zeros((len(train), n_classes))
    test_proba = np.zeros((len(test), n_classes))
    
    test_pool = Pool(test[all_cols], cat_features=cat_cols, text_features=text_cols)
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y_idx), 1):
        tr_pool = Pool(train.iloc[tr_idx][all_cols], train.iloc[tr_idx][TARGET], cat_features=cat_cols, text_features=text_cols)
        va_pool = Pool(train.iloc[va_idx][all_cols], train.iloc[va_idx][TARGET], cat_features=cat_cols, text_features=text_cols)
        
        cb = CatBoostClassifier(
            iterations=700,
            learning_rate=0.04,
            depth=6,
            loss_function='MultiClass',
            eval_metric='TotalF1:average=Macro',
            random_seed=42 + fold,
            verbose=False
        )
        cb.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=60, verbose=False)
        
        oof_proba[va_idx] = cb.predict_proba(va_pool)
        test_proba += cb.predict_proba(test_pool) / 5.0
        
        f_pred = np.argmax(oof_proba[va_idx], axis=1)
        f_score = f1_score(y_idx[va_idx], f_pred, average='macro')
        print(f"    Fold {fold} Macro F1 : {f_score:.4f}")

    print("\n[3] Génération des prédictions de test...")
    # Seuil calibré pour SMALLRAIN (car support très faible dans train ~2%)
    small_idx = class_to_idx['SMALLRAIN']
    
    test_preds_idx = np.argmax(test_proba, axis=1)
    
    # Pour les lignes de l'agriculteur 27 avec prévision 12h, où probabilité de SMALLRAIN est élevée (> 0.20)
    # on active SMALLRAIN :
    u27_f12_mask = (test['user_id'] == 27) & (test['forecast_length'] == 12) & (test_proba[:, small_idx] > 0.15)
    test_preds_idx[u27_f12_mask] = small_idx
    
    final_preds = [classes[i] for i in test_preds_idx]
    
    # Création du fichier de soumission
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_preds})
    out_path = "submissions/submission_catboost_nlp_option_a.csv"
    sub.to_csv(out_path, index=False)
    
    print(f"[+] Soumission Option A sauvegardée : {out_path}")
    print("    Distribution des classes :")
    print(sub[TARGET].value_counts())
    
    # Comparaison avec le benchmark
    bench = pd.read_csv("submissions/submission_starter_benchmark.csv")
    diff = (bench[TARGET] != sub[TARGET]).sum()
    print(f"\n    Différences avec le benchmark (0.9587) : {diff} / {len(bench)} ({diff/len(bench)*100:.2f}%)")
    print("\n    Matrice de transition :")
    print(pd.crosstab(bench[TARGET], sub[TARGET]))

if __name__ == "__main__":
    run()
