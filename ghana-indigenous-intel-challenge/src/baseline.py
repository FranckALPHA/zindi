"""
Baseline Pipeline for Ghana's Indigenous Intel Challenge
Métrique : Macro F1-Score
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

def load_data(data_dir):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    sample_sub_path = os.path.join(data_dir, "SampleSubmission.csv")

    assert os.path.exists(train_path), f"Fichier manquant : {train_path}"
    assert os.path.exists(test_path), f"Fichier manquant : {test_path}"
    assert os.path.exists(sample_sub_path), f"Fichier manquant : {sample_sub_path}"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_sub_path)
    return train, test, sample_sub

def feature_engineering(df):
    df = df.copy()
    
    # Nettoyage des colonnes catégorielles
    for col in ['community', 'district', 'indicator']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df.loc[df[col].isin(['nan', 'none', '']), col] = np.nan
            
    # Extraction temporelle
    if 'prediction_time' in df.columns:
        dt = pd.to_datetime(df['prediction_time'].astype(str), errors='coerce')
        df['pred_hour'] = dt.dt.hour
        df['pred_dow'] = dt.dt.dayofweek
        df['pred_day'] = dt.dt.day
        df['pred_month'] = dt.dt.month
        
        # Encodage cyclique de l'heure
        df['hour_sin'] = np.sin(2 * np.pi * df['pred_hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['pred_hour'] / 24.0)

    # Indicateur présent ou manquant
    df['has_indicator'] = df['indicator'].notna().astype(int)
    
    return df

def run_pipeline():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_dir, "data")
    submissions_dir = os.path.join(project_dir, "submissions")
    os.makedirs(submissions_dir, exist_ok=True)

    print("[1] Chargement des données...")
    train, test, sample_sub = load_data(data_dir)
    print(f"    Train shape : {train.shape}, Test shape : {test.shape}")

    print("\n[2] Feature Engineering...")
    train_feat = feature_engineering(train)
    test_feat = feature_engineering(test)

    TARGET = "Target"
    ID_COL = "ID"
    drop_cols = [ID_COL, TARGET, "prediction_time", "time_observed", "indicator_description"]
    
    feature_cols = [c for c in train_feat.columns if c not in drop_cols]
    
    numeric_cols = []
    cat_cols = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(train_feat[c]):
            numeric_cols.append(c)
        else:
            cat_cols.append(c)

    print(f"    Features numériques ({len(numeric_cols)}) : {numeric_cols}")
    print(f"    Features catégorielles ({len(cat_cols)}) : {cat_cols}")

    X = train_feat[feature_cols]
    y = train_feat[TARGET]
    X_test = test_feat[feature_cols]

    # Preprocessing
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    # Modèle avec gestion du déséquilibre de classe
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )

    full_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    print("\n[3] Évaluation Stratified 5-Fold CV (Métrique : Macro F1)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = pd.Series(index=y.index, dtype=object)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        full_pipe.fit(X_tr, y_tr)
        preds = full_pipe.predict(X_va)
        oof_preds.iloc[val_idx] = preds
        
        fold_f1 = f1_score(y_va, preds, average='macro')
        print(f"    Fold {fold} - Macro F1: {fold_f1:.4f}")

    total_f1 = f1_score(y, oof_preds, average='macro')
    print(f"\n---> OOF Global Macro F1 Score : {total_f1:.4f} <---")
    print("\nRapport détaillé de classification :")
    print(classification_report(y, oof_preds))

    print("\n[4] Entraînement sur l'ensemble complet et prédiction Test...")
    full_pipe.fit(X, y)
    test_preds = full_pipe.predict(X_test)

    # Création du fichier de soumission conforme
    sub = pd.DataFrame({
        ID_COL: test[ID_COL],
        TARGET: test_preds
    })

    # Alignement rigoureux sur SampleSubmission
    submission = sample_sub[[ID_COL]].merge(sub, on=ID_COL, how="left")
    sub_path = os.path.join(submissions_dir, "submission_rf_balanced.csv")
    submission.to_csv(sub_path, index=False)
    print(f"[+] Soumission sauvegardée avec succès : {sub_path}")
    print(submission.head(10))

    # Vérifications de cohérence
    assert submission.shape[0] == sample_sub.shape[0], "Erreur de dimension !"
    assert list(submission.columns) == list(sample_sub.columns), "Erreur de colonnes !"
    assert not submission[TARGET].isnull().any(), "Valeurs manquantes détectées dans la cible !"
    print("\n[✓] Tous les tests de validation de la soumission sont validés.")

if __name__ == "__main__":
    run_pipeline()
