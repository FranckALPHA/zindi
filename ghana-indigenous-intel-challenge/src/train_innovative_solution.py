"""
Innovative Solution Pipeline - Breaking out of the 0.9587 Plateau
1. TF-IDF sur le savoir indigène textuel (indicator_description)
2. Encodage temporel cyclique (généralisable sans sur-apprentissage des dates)
3. Profils de fiabilité par observateur (agrégations bayésiennes)
4. Ensemble diversifié : Random Forest + CatBoost + LightGBM
5. Optimisation directe des seuils de décision pour la métrique Macro-F1
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 42

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_dir, "data")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(data_dir, "SampleSubmission.csv"))
    return train, test, sample_sub, project_dir

def build_features(train, test):
    print("[1] Ingénierie des caractéristiques innovante...")
    train = train.copy()
    test = test.copy()
    
    # 1. Traitement temporel cyclique
    for df in [train, test]:
        dt = pd.to_datetime(df['prediction_time'].astype(str), errors='coerce')
        df['hour'] = dt.dt.hour.fillna(12)
        df['dow'] = dt.dt.day_of_week.fillna(2)
        df['day'] = dt.dt.day.fillna(15)
        df['month'] = dt.dt.month.fillna(7)
        
        # Encodages cycliques
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7.0)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7.0)
        
        # Périodes de la journée
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
        
        # Indicateurs d'absence / présence
        df['has_indicator'] = df['indicator'].notna().astype(int)
        df['has_desc'] = df['indicator_description'].notna().astype(int)
        df['has_time_obs'] = df['time_observed'].notna().astype(int)
        
        # Remplissage par défaut
        df['confidence'] = df['confidence'].fillna(0.3)
        df['predicted_intensity'] = df['predicted_intensity'].fillna(0.0)
        df['forecast_length'] = df['forecast_length'].fillna(12)
        
        # Interactions physiques
        df['conf_x_intensity'] = df['confidence'] * df['predicted_intensity']
        df['intensity_x_forecast'] = df['predicted_intensity'] * df['forecast_length']
        df['conf_x_forecast'] = df['confidence'] * df['forecast_length']

    # 2. TF-IDF sur les descriptions textuelles des savoirs indigènes (50 composantes)
    print("    Extraction sémantique TF-IDF des indicateurs traditionnels...")
    train_text = train['indicator_description'].fillna('missing').astype(str)
    test_text = test['indicator_description'].fillna('missing').astype(str)
    
    tfidf = TfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=2, stop_words='english')
    train_tfidf = tfidf.fit_transform(train_text).toarray()
    test_tfidf = tfidf.transform(test_text).toarray()
    
    tfidf_cols = [f'tfidf_{i}' for i in range(50)]
    train_tfidf_df = pd.DataFrame(train_tfidf, columns=tfidf_cols, index=train.index)
    test_tfidf_df = pd.DataFrame(test_tfidf, columns=tfidf_cols, index=test.index)
    
    train = pd.concat([train, train_tfidf_df], axis=1)
    test = pd.concat([test, test_tfidf_df], axis=1)

    # 3. Profils statistiques par observateur (user_id)
    print("    Calcul des profils de confiance bayésienne par agriculteur...")
    user_stats = train.groupby('user_id').agg({
        'confidence': ['mean', 'std'],
        'predicted_intensity': ['mean', 'max'],
        'has_indicator': 'mean'
    })
    user_stats.columns = [f'user_{c[0]}_{c[1]}' for c in user_stats.columns]
    user_stats = user_stats.reset_index().fillna(0)
    
    train = train.merge(user_stats, on='user_id', how='left').fillna(0)
    test = test.merge(user_stats, on='user_id', how='left').fillna(0)

    # 4. Encodage des catégorielles
    for col in ['community', 'district', 'indicator', 'time_observed']:
        le = LabelEncoder()
        train[col] = train[col].fillna('missing').astype(str)
        test[col] = test[col].fillna('missing').astype(str)
        all_vals = pd.concat([train[col], test[col]]).unique()
        le.fit(all_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    return train, test

def optimize_thresholds(oof_proba, y_true, classes):
    print("\n[3] Optimisation mathématique des seuils de décision pour le Macro F1...")
    
    def loss_func(weights):
        w_proba = oof_proba * weights
        preds = np.argmax(w_proba, axis=1)
        pred_labels = [classes[i] for i in preds]
        return -f1_score(y_true, pred_labels, average='macro')

    # Initialisation neutre : poids = 1.0 pour chaque classe
    init_weights = np.ones(len(classes))
    res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 300})
    best_weights = res.x
    print(f"    Poids optimaux trouvés : {best_weights.round(4)}")
    
    baseline_preds = [classes[i] for i in np.argmax(oof_proba, axis=1)]
    base_f1 = f1_score(y_true, baseline_preds, average='macro')
    
    opt_preds = [classes[i] for i in np.argmax(oof_proba * best_weights, axis=1)]
    opt_f1 = f1_score(y_true, opt_preds, average='macro')
    
    print(f"    Macro F1 avant optimisation (Argmax brut) : {base_f1:.5f}")
    print(f"    Macro F1 après optimisation des seuils    : {opt_f1:.5f} (Gain: +{opt_f1 - base_f1:.5f})")
    return best_weights

def run():
    train, test, sample_sub, project_dir = load_data()
    train_feat, test_feat = build_features(train, test)
    
    TARGET = 'Target'
    ID_COL = 'ID'
    drop_cols = [ID_COL, TARGET, 'prediction_time', 'indicator_description']
    feature_cols = [c for c in train_feat.columns if c not in drop_cols]
    
    X = train_feat[feature_cols]
    y = train_feat[TARGET]
    X_test = test_feat[feature_cols]
    
    classes = sorted(y.unique())
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = y.map(class_to_idx).values
    
    print(f"\n[2] Entraînement Stratified 5-Fold avec Modèles Diversifiés...")
    print(f"    Nombre de features : {len(feature_cols)}")
    print(f"    Classes : {classes}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_rf = np.zeros((len(X), n_classes))
    oof_cb = np.zeros((len(X), n_classes))
    oof_lgb = np.zeros((len(X), n_classes))
    
    test_rf = np.zeros((len(X_test), n_classes))
    test_cb = np.zeros((len(X_test), n_classes))
    test_lgb = np.zeros((len(X_test), n_classes))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_idx), 1):
        X_tr, y_tr = X.iloc[tr_idx], y_idx[tr_idx]
        X_va, y_va = X.iloc[val_idx], y_idx[val_idx]
        
        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=300, random_state=SEED + fold, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        oof_rf[val_idx] = rf.predict_proba(X_va)
        test_rf += rf.predict_proba(X_test) / 5.0
        
        # 2. CatBoost
        cb = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            random_seed=SEED + fold,
            loss_function='MultiClass',
            auto_class_weights='Balanced',
            verbose=False
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=80, verbose=False)
        oof_cb[val_idx] = cb.predict_proba(X_va)
        test_cb += cb.predict_proba(X_test) / 5.0
        
        # 3. LightGBM
        lgbm = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            random_state=SEED + fold,
            class_weight='balanced',
            n_jobs=-1,
            verbosity=-1
        )
        lgbm.fit(X_tr, y_tr)
        oof_lgb[val_idx] = lgbm.predict_proba(X_va)
        test_lgb += lgbm.predict_proba(X_test) / 5.0
        
        fold_blend = (oof_rf[val_idx] * 0.4 + oof_cb[val_idx] * 0.35 + oof_lgb[val_idx] * 0.25)
        fold_f1 = f1_score(y_va, np.argmax(fold_blend, axis=1), average='macro')
        print(f"    Fold {fold} Macro F1 (Blend brut) : {fold_f1:.4f}")
        
    oof_ensemble = (oof_rf * 0.40 + oof_cb * 0.35 + oof_lgb * 0.25)
    test_ensemble = (test_rf * 0.40 + test_cb * 0.35 + test_lgb * 0.25)
    
    # Optimisation des seuils
    best_weights = optimize_thresholds(oof_ensemble, y, classes)
    
    # Application des seuils optimaux sur les probabilités de test
    opt_test_proba = test_ensemble * best_weights
    final_test_preds = [classes[i] for i in np.argmax(opt_test_proba, axis=1)]
    
    # Création de la soumission
    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_test_preds})
    sub_path = os.path.join(project_dir, "submissions", "submission_innovative_nlp_thresholded.csv")
    sub.to_csv(sub_path, index=False)
    
    print(f"\n[4] Soumission sauvegardée : {sub_path}")
    print("    Distribution des classes prédites :")
    print(sub[TARGET].value_counts())
    
    # Comparaison avec le benchmark
    bench_path = os.path.join(project_dir, "submissions", "submission_starter_benchmark.csv")
    if os.path.exists(bench_path):
        bench = pd.read_csv(bench_path)
        diff = (bench[TARGET] != sub[TARGET]).sum()
        print(f"\n    Différences avec le benchmark (0.9587) : {diff} / {len(bench)} ({diff/len(bench)*100:.2f}%)")
        print("\n    Matrice de transition :")
        print(pd.crosstab(bench[TARGET], sub[TARGET]))

if __name__ == "__main__":
    run()
