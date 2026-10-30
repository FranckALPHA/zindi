"""
Scientific Physics-Grounded Pipeline for Ghana's Indigenous Intel Challenge
Intègre :
1. Orographie physique (Plateau d'Atiwa West vs Basses Terres d'Obuasi East)
2. Indice de fiabilité bayésienne des observateurs (Bayesian Observer Reliability)
3. Règle physique d'inversion thermique (Rosée / Brouillard => 100% NORAIN)
4. Détecteur expert de pluie fine (Farmer 27 - SMALLRAIN)
5. Consensus spatial villageois
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.abspath(os.path.join(base_dir, ".."))
    data_dir = os.path.join(proj_dir, "data")
    sub_dir = os.path.join(proj_dir, "submissions")

    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    sample_sub = pd.read_csv(os.path.join(data_dir, "SampleSubmission.csv"))

    # 1. Nettoyage de base
    for col in ['community', 'district', 'indicator']:
        train[col] = train[col].astype(str).str.strip().str.lower().replace({"nan": "missing", "none": "missing", "": "missing"})
        test[col] = test[col].astype(str).str.strip().str.lower().replace({"nan": "missing", "none": "missing", "": "missing"})

    # 2. Indice de fiabilité bayésienne de l'observateur
    farmer_rain_preds = train.groupby('user_id')['predicted_intensity'].apply(lambda s: (s > 0).sum())
    farmer_tp = train[(train['predicted_intensity'] > 0) & (train['Target'] != 'NORAIN')].groupby('user_id').size()
    farmer_precision = (farmer_tp / farmer_rain_preds.replace(0, np.nan)).fillna(0.0).to_dict()

    train['farmer_reliability'] = train['user_id'].map(farmer_precision).fillna(0.15)
    test['farmer_reliability'] = test['user_id'].map(farmer_precision).fillna(0.15)

    # 3. Features temporelles et cycliques
    for df in [train, test]:
        dt = pd.to_datetime(df['prediction_time'], errors='coerce')
        df['hour'] = dt.dt.hour.fillna(12)
        df['day'] = dt.dt.day.fillna(15)
        df['month'] = dt.dt.month.fillna(6)
        df['dayofweek'] = dt.dt.day_of_week.fillna(2)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Encodage Orographique : Atiwa West est le seul massif montagneux
        df['is_atiwa_highland'] = (df['district'] == 'atiwa_west').astype(int)
        df['is_obuasi_dry_lowland'] = (df['district'] == 'obuasi_east').astype(int)

        # Règle d'inversion thermique : dew / fog
        df['is_thermal_inversion'] = df['indicator'].isin(['dew', 'fog']).astype(int)

        # Interactions
        df['conf_x_intensity'] = df['confidence'].fillna(0.3) * df['predicted_intensity'].fillna(0.0)
        df['conf_x_reliability'] = df['confidence'].fillna(0.3) * df['farmer_reliability']
        df['intensity_x_atiwa'] = df['predicted_intensity'].fillna(0.0) * df['is_atiwa_highland']

    # 4. TF-IDF sur les descriptions
    tfidf = TfidfVectorizer(max_features=40, ngram_range=(1, 2), stop_words='english')
    train_text = tfidf.fit_transform(train['indicator_description'].fillna('missing'))
    test_text = tfidf.transform(test['indicator_description'].fillna('missing'))

    train_text_df = pd.DataFrame(train_text.toarray(), columns=[f'tfidf_{i}' for i in range(train_text.shape[1])])
    test_text_df = pd.DataFrame(test_text.toarray(), columns=[f'tfidf_{i}' for i in range(test_text.shape[1])])

    train = pd.concat([train.reset_index(drop=True), train_text_df], axis=1)
    test = pd.concat([test.reset_index(drop=True), test_text_df], axis=1)

    # Encodage catégoriel
    for c in ['community', 'district', 'indicator']:
        le = LabelEncoder()
        all_vals = pd.concat([train[c], test[c]]).unique()
        le.fit(all_vals)
        train[c] = le.transform(train[c])
        test[c] = le.transform(test[c])

    drop_cols = ['ID', 'prediction_time', 'indicator_description', 'time_observed', 'Target']
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X = train[feature_cols].fillna(0)
    X_test = test[feature_cols].fillna(0)

    target_enc = LabelEncoder()
    y = target_enc.fit_transform(train['Target'])
    classes = target_enc.classes_
    n_classes = len(classes)

    print(f"[+] Entraînement Modèle Scientifique Orographique & Bayésien ({len(feature_cols)} features)...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), n_classes))
    test_preds = np.zeros((len(X_test), n_classes))

    rf = RandomForestClassifier(n_estimators=300, max_depth=16, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=10, class_weight='balanced', random_state=42)

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        rf.fit(X_tr, y_tr)
        hgb.fit(X_tr, y_tr)

        val_prob = 0.5 * rf.predict_proba(X_va) + 0.5 * hgb.predict_proba(X_va)
        oof_preds[val_idx] = val_prob

        test_preds += (0.5 * rf.predict_proba(X_test) + 0.5 * hgb.predict_proba(X_test)) / 10

    total_f1 = f1_score(y, np.argmax(oof_preds, axis=1), average='macro')
    print(f"\n=======================================================")
    print(f"===> OOF Global Scientific Macro F1 : {total_f1:.4f} <===")
    print(f"=======================================================")
    print(classification_report(y, np.argmax(oof_preds, axis=1), target_names=classes))

    # Application des règles physiques déduites de la recherche
    test_pred_classes = np.argmax(test_preds, axis=1)
    final_labels = target_enc.inverse_transform(test_pred_classes)

    raw_test_clean = test.copy()
    raw_test_clean['Predicted_Target'] = final_labels
    raw_test_clean['district_clean'] = raw_test_clean['district'].astype(str).str.strip().str.lower()
    raw_test_clean['indicator_clean'] = raw_test_clean['indicator'].astype(str).str.strip().str.lower()

    # Règle 1 : Inversion thermique (fog / dew) => 100% NORAIN
    fog_dew_mask = raw_test_clean['indicator_clean'].isin(['fog', 'dew'])
    raw_test_clean.loc[fog_dew_mask, 'Predicted_Target'] = 'NORAIN'

    # Règle 2 : Obuasi East => Basses terres sans pluie
    obuasi_mask = (raw_test_clean['district_clean'] == 'obuasi_east')
    raw_test_clean.loc[obuasi_mask, 'Predicted_Target'] = 'NORAIN'

    # Règle 3 : Détecteur expert Farmer 27 => SMALLRAIN
    farmer27_mask = (raw_test_clean['user_id'] == 27) & (raw_test_clean['forecast_length'] == 12)
    raw_test_clean.loc[farmer27_mask, 'Predicted_Target'] = 'SMALLRAIN'

    # Sauvegarde
    sub = sample_sub[['ID']].merge(raw_test_clean[['ID', 'Predicted_Target']].rename(columns={'Predicted_Target': 'Target'}), on='ID', how='left')
    out_path = os.path.join(sub_dir, "submission_scientific_research.csv")
    sub.to_csv(out_path, index=False)
    print(f"\n[✓] Soumission scientifique générée : {out_path}")
    print("\nDistribution des prédictions Test :")
    print(sub['Target'].value_counts())

if __name__ == "__main__":
    main()
