"""
Top Solution Pipeline for Ghana's Indigenous Intel Challenge
Architecture inspirée de la solution 2ème place (LightGBM + Random Forest + HistGB + TF-IDF)
Optimisé pour maximiser le Macro F1 et prédire toutes les classes avec précision.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

SEED = 42

def _strip_norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.strip()
            df[c] = s.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "none": np.nan, "": np.nan})
    return df

def advanced_features(df, is_train=True, text_vectorizer=None, fit_text=False):
    df = df.copy()

    # Time features
    df['prediction_time'] = pd.to_datetime(df['prediction_time'], errors='coerce')
    df['hour'] = df['prediction_time'].dt.hour.fillna(12)
    df['day'] = df['prediction_time'].dt.day.fillna(15)
    df['month'] = df['prediction_time'].dt.month.fillna(6)
    df['dayofweek'] = df['prediction_time'].dt.day_of_week.fillna(2)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['day_of_year'] = df['prediction_time'].dt.dayofyear.fillna(180)

    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    # Time of day
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)

    # Ghana Seasonal patterns
    df['rainy_main'] = ((df['month'] >= 4) & (df['month'] <= 6)).astype(int)
    df['rainy_secondary'] = ((df['month'] >= 9) & (df['month'] <= 11)).astype(int)
    df['dry_season'] = ((df['rainy_main'] == 0) & (df['rainy_secondary'] == 0)).astype(int)

    # Interactions
    df['confidence'] = df['confidence'].fillna(0.3)
    df['predicted_intensity'] = df['predicted_intensity'].fillna(0.0)
    df['forecast_length'] = df['forecast_length'].fillna(12)

    df['conf_x_intensity'] = df['confidence'] * df['predicted_intensity']
    df['conf_squared'] = df['confidence'] ** 2
    df['conf_cubed'] = df['confidence'] ** 3
    df['intensity_x_forecast'] = df['predicted_intensity'] * df['forecast_length']
    df['conf_x_forecast'] = df['confidence'] * df['forecast_length']
    df['conf_x_hour'] = df['confidence'] * df['hour']
    df['conf_x_rainy'] = df['confidence'] * (df['rainy_main'] + df['rainy_secondary'])

    # Missing indicators
    df['has_indicator'] = df['indicator'].notna().astype(int)
    df['has_indicator_desc'] = df['indicator_description'].notna().astype(int)
    df['has_time_observed'] = df['time_observed'].notna().astype(int)
    df['missing_count'] = 3 - df['has_indicator'] - df['has_indicator_desc'] - df['has_time_observed']

    # Include pred_date and time_observed as categorical
    df['pred_date'] = df['prediction_time'].dt.date.astype(str)
    df['time_observed'] = df['time_observed'].fillna('missing').astype(str)

    # Text features TF-IDF
    if 'indicator_description' in df.columns:
        text_data = df['indicator_description'].fillna('missing').astype(str)
        if fit_text:
            text_vectorizer = TfidfVectorizer(max_features=50, ngram_range=(1, 2), min_df=2, stop_words='english')
            text_matrix = text_vectorizer.fit_transform(text_data)
        elif text_vectorizer is not None:
            text_matrix = text_vectorizer.transform(text_data)
        else:
            text_matrix = None

        if text_matrix is not None:
            text_df = pd.DataFrame(text_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(text_matrix.shape[1])])
            df = pd.concat([df.reset_index(drop=True), text_df], axis=1)

    return df, text_vectorizer

def build_aggregations(train, test):
    train_orig = train.copy()
    test_orig = test.copy()

    # User stats
    user_stats = train.groupby('user_id').agg({
        'confidence': ['mean', 'std', 'count'],
        'predicted_intensity': ['mean', 'sum'],
        'forecast_length': 'mean'
    }).reset_index()
    user_stats.columns = ['user_id'] + [f'user_{col[0]}_{col[1]}' for col in user_stats.columns[1:]]
    user_stats['user_confidence_std'] = user_stats['user_confidence_std'].fillna(0)

    # Community stats
    comm_stats = train.groupby('community').agg({
        'confidence': ['mean', 'std'],
        'predicted_intensity': 'mean',
        'Target': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'NORAIN'
    }).reset_index()
    comm_stats.columns = ['community', 'comm_conf_mean', 'comm_conf_std', 'comm_intensity_mean', 'comm_target_mode']
    comm_stats['comm_conf_std'] = comm_stats['comm_conf_std'].fillna(0)

    # District stats
    dist_stats = train.groupby('district').agg({
        'confidence': 'mean',
        'predicted_intensity': 'mean',
        'Target': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'NORAIN'
    }).reset_index()
    dist_stats.columns = ['district', 'dist_conf_mean', 'dist_intensity_mean', 'dist_target_mode']

    # Merge
    train = train.merge(user_stats, on='user_id', how='left')
    train = train.merge(comm_stats, on='community', how='left')
    train = train.merge(dist_stats, on='district', how='left')

    test = test.merge(user_stats, on='user_id', how='left')
    test = test.merge(comm_stats, on='community', how='left')
    test = test.merge(dist_stats, on='district', how='left')

    # Fill
    for col in train.columns:
        if train[col].dtype in ['float64', 'int64']:
            train[col] = train[col].fillna(0)
            test[col] = test[col].fillna(0)

    for col in ['comm_target_mode', 'dist_target_mode']:
        if col in train.columns:
            train[col] = train[col].fillna('NORAIN')
            test[col] = test[col].fillna('NORAIN')

    return train, test

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.abspath(os.path.join(base_dir, ".."))
    data_dir = os.path.join(proj_dir, "data")
    sub_dir = os.path.join(proj_dir, "submissions")
    os.makedirs(sub_dir, exist_ok=True)

    print("[+] Chargement des fichiers bruts...")
    raw_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    raw_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(data_dir, "SampleSubmission.csv"))

    train_clean = _strip_norm(raw_train)
    test_clean = _strip_norm(raw_test)

    print("[+] Feature Engineering avancé...")
    train_feat, text_vec = advanced_features(train_clean, is_train=True, fit_text=True)
    test_feat, _ = advanced_features(test_clean, is_train=False, text_vectorizer=text_vec)

    print("[+] Création des agrégations multi-niveaux...")
    train_agg, test_agg = build_aggregations(train_feat, test_feat)

    # Encodage des catégories
    cat_cols = ['community', 'district', 'indicator', 'comm_target_mode', 'dist_target_mode', 'time_observed', 'pred_date']
    for c in cat_cols:
        le = LabelEncoder()
        train_agg[c] = train_agg[c].fillna('missing').astype(str)
        test_agg[c] = test_agg[c].fillna('missing').astype(str)
        all_vals = pd.concat([train_agg[c], test_agg[c]]).unique()
        le.fit(all_vals)
        train_agg[c] = le.transform(train_agg[c])
        test_agg[c] = le.transform(test_agg[c])

    # Target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(train_agg['Target'])
    classes = target_encoder.classes_
    n_classes = len(classes)
    print(f"[+] Classes cibles : {classes}")

    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"[+] Poids des classes : {class_weight_dict}")

    drop_cols = ['ID', 'prediction_time', 'indicator_description', 'Target']
    feature_cols = [c for c in train_agg.columns if c not in drop_cols]
    print(f"[+] Nombre de variables d'entrée : {len(feature_cols)}")

    X = train_agg[feature_cols].copy()
    X_test = test_agg[feature_cols].copy()

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_preds_lgb = np.zeros((len(X), n_classes))
    oof_preds_rf = np.zeros((len(X), n_classes))
    oof_preds_hgb = np.zeros((len(X), n_classes))

    test_preds_lgb = np.zeros((len(X_test), n_classes))
    test_preds_rf = np.zeros((len(X_test), n_classes))
    test_preds_hgb = np.zeros((len(X_test), n_classes))

    print(f"\n[+] Entraînement de l'ensemble sur Stratified {n_splits}-Fold CV...")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        class_weight='balanced_subsample',
        random_state=SEED,
        n_jobs=-1
    )

    hgb_model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_depth=10,
        class_weight='balanced',
        random_state=SEED
    )

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # 1. Random Forest
        rf_model.fit(X_tr, y_tr)
        oof_preds_rf[val_idx] = rf_model.predict_proba(X_val)
        test_preds_rf += rf_model.predict_proba(X_test) / n_splits

        # 2. HistGradientBoosting
        hgb_model.fit(X_tr, y_tr)
        oof_preds_hgb[val_idx] = hgb_model.predict_proba(X_val)
        test_preds_hgb += hgb_model.predict_proba(X_test) / n_splits

        # 3. LightGBM si présent
        if HAS_LGB:
            sample_weights_tr = np.array([class_weights[cls_id] for cls_id in y_tr])
            sample_weights_val = np.array([class_weights[cls_id] for cls_id in y_val])
            lgb_train = lgb.Dataset(X_tr, label=y_tr, weight=sample_weights_tr)
            lgb_val = lgb.Dataset(X_val, label=y_val, weight=sample_weights_val, reference=lgb_train)

            lgb_params = {
                'objective': 'multiclass',
                'num_class': n_classes,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': 0.02,
                'num_leaves': 63,
                'max_depth': 8,
                'min_child_samples': 15,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': SEED,
                'verbose': -1
            }
            lgb_trained = lgb.train(
                lgb_params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_train, lgb_val],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            oof_preds_lgb[val_idx] = lgb_trained.predict(X_val, num_iteration=lgb_trained.best_iteration)
            test_preds_lgb += lgb_trained.predict(X_test, num_iteration=lgb_trained.best_iteration) / n_splits

        fold_ensemble_oof = (0.5 * oof_preds_rf[val_idx] + 0.5 * oof_preds_hgb[val_idx]) if not HAS_LGB else (0.35 * oof_preds_lgb[val_idx] + 0.35 * oof_preds_hgb[val_idx] + 0.30 * oof_preds_rf[val_idx])
        fold_f1 = f1_score(y_val, np.argmax(fold_ensemble_oof, axis=1), average='macro')
        print(f"    Fold {fold:2d}/{n_splits} - Macro F1: {fold_f1:.4f}")

    if HAS_LGB:
        oof_final = 0.40 * oof_preds_lgb + 0.30 * oof_preds_hgb + 0.30 * oof_preds_rf
        test_final = 0.40 * test_preds_lgb + 0.30 * test_preds_hgb + 0.30 * test_preds_rf
    else:
        oof_final = 0.50 * oof_preds_rf + 0.50 * oof_preds_hgb
        test_final = 0.50 * test_preds_rf + 0.50 * test_preds_hgb

    oof_classes = np.argmax(oof_final, axis=1)
    total_f1 = f1_score(y, oof_classes, average='macro')
    print(f"\n=======================================================")
    print(f"===> OOF Global Ensemble Macro F1 Score : {total_f1:.4f} <===")
    print(f"=======================================================")
    print("\nRapport détaillé de classification :")
    print(classification_report(y, oof_classes, target_names=classes))

    # Prédictions sur le test
    test_classes = np.argmax(test_final, axis=1)
    test_labels = target_encoder.inverse_transform(test_classes)

    sub = pd.DataFrame({
        'ID': raw_test['ID'],
        'Target': test_labels
    })

    # Alignement SampleSubmission
    submission = sample_sub[['ID']].merge(sub, on='ID', how='left')
    out_file = os.path.join(sub_dir, "submission_top1_ensemble.csv")
    submission.to_csv(out_file, index=False)
    print(f"\n[✓] Soumission enregistrée avec succès : {out_file}")
    print("\nDistribution des prédictions Test :")
    print(submission['Target'].value_counts())

if __name__ == "__main__":
    main()
