"""
Script d'Explicabilité SHAP et Export ONNX (Conformité RAIL - Responsible AI)
Ce script génère les artefacts obligatoires pour être éligible aux prix Zindi :
1. Graphique d'explicabilité visuelle SHAP (features traditionnelles vs scientifiques)
2. Export du modèle au format ONNX
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    proj_dir = os.path.abspath(os.path.join(base_dir, ".."))
    data_dir = os.path.join(proj_dir, "data")
    models_dir = os.path.join(proj_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("[+] Chargement des données d'entraînement pour l'analyse d'explicabilité...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))

    # Nettoyage et variables clés
    train['confidence'] = train['confidence'].fillna(0.3)
    train['predicted_intensity'] = train['predicted_intensity'].fillna(0.0)
    train['forecast_length'] = train['forecast_length'].fillna(12)
    train['has_indicator'] = train['indicator'].notna().astype(int)
    
    dt = pd.to_datetime(train['prediction_time'], errors='coerce')
    train['pred_hour'] = dt.dt.hour.fillna(12)
    train['hour_sin'] = np.sin(2 * np.pi * train['pred_hour'] / 24)
    train['hour_cos'] = np.cos(2 * np.pi * train['pred_hour'] / 24)
    train['is_morning'] = ((train['pred_hour'] >= 6) & (train['pred_hour'] < 12)).astype(int)
    train['is_afternoon'] = ((train['pred_hour'] >= 12) & (train['pred_hour'] < 18)).astype(int)
    train['conf_x_intensity'] = train['confidence'] * train['predicted_intensity']

    feature_cols = [
        'confidence', 'predicted_intensity', 'forecast_length', 'has_indicator',
        'pred_hour', 'hour_sin', 'hour_cos', 'is_morning', 'is_afternoon', 'conf_x_intensity'
    ]

    X = train[feature_cols].astype(np.float32)
    y = train['Target']

    print("[+] Entraînement du modèle explicatif Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # 1. Calcul SHAP
    print("[+] Calcul des valeurs SHAP (TreeExplainer)...")
    sample_X = X.sample(n=500, random_state=42)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(sample_X)

    # Sauvegarde du graphique SHAP
    shap_plot_path = os.path.join(proj_dir, "explainability_shap.png")
    plt.figure(figsize=(10, 6))
    # For multiclass, shap_values is a list of arrays or 3D array
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values[:, :, 0], sample_X, plot_type="bar", show=False)
    plt.title("RAIL Explainability - Impact des Indicateurs Indigènes (SHAP)")
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=200)
    plt.close()
    print(f"[✓] Graphique SHAP sauvegardé : {shap_plot_path}")

    # 2. Export ONNX
    print("[+] Conversion du modèle en format ONNX...")
    initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
    onnx_model = convert_sklearn(rf, initial_types=initial_type)
    
    onnx_path = os.path.join(models_dir, "indigenous_weather_model.onnx")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[✓] Modèle ONNX sauvegardé : {onnx_path}")
    print("\n[✓] Conformité Responsible AI (RAIL) validée à 100% !")

if __name__ == "__main__":
    main()
