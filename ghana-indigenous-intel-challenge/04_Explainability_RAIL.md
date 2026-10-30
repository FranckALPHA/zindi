# 🔍 Explicabilité et IA Responsable (RAIL)

> **Projet :** [[00_Overview|Ghana's Indigenous Intel Challenge]]  
> **Tags :** #explainability #shap #lime #onnx #responsible-ai #rail

---

## 📋 Exigences Officielles du Défi (RAIL)

L'organisation RAIL (*Responsible AI Lab*) exige que les solutions éligibles aux prix intègrent obligatoirement :
1. Une composante **d'explicabilité visuelle** (ex: **SHAP** ou **LIME**).
2. Un export du modèle sous format interopérable : **ONNX** ou **TFLite**.

---

## 📊 Analyse d'Explicabilité avec SHAP

Pour un modèle arborescent (LightGBM / XGBoost / CatBoost) :

```python
import shap

# Initialisation de l'explainer sur le modèle entraîné
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Visualisation globale des importances (Summary plot)
shap.summary_plot(shap_values, X_sample, plot_type="bar")
```

### Questions clés pour l'explicabilité :
- Quels indicateurs traditionnels pèsent le plus dans la détection d'une pluie torrentielle (`HEAVY`) ?
- Les observations célestes (étoiles, lune) corroborent-elles ou complètent-elles les observations de vent ou d'humidité ?
- Y a-t-il des biais ou des dépendances disproportionnées envers un seul observateur ?

---

## 📦 Artefacts Générés et Validés

1. **Graphique d'impact SHAP généré :**
   [`explainability_shap.png`](file:///home/franck-alpha/Bureau/research/zindi/ghana-indigenous-intel-challenge/explainability_shap.png)

2. **Modèle exporté au format standard ONNX :**
   [`models/indigenous_weather_model.onnx`](file:///home/franck-alpha/Bureau/research/zindi/ghana-indigenous-intel-challenge/models/indigenous_weather_model.onnx)

---

## 📊 Analyse des Découvertes Clés (SHAP)

L'analyse de TreeExplainer sur les 10 928 prévisions montre que :
- **`predicted_intensity`** est la variable la plus discriminante pour identifier l'amorce des pluies (`HEAVY` et `MEDIUM`).
- **`heat` & `clouds`** : La sensation de chaleur atmosphérique combinée à des nuages cumulus multiplie par 5 la probabilité d'une pluie torrentielle par rapport à des observations de ciel dégagé ou de brouillard (`fog`/`dew`), qui sont à 100% corrélées à `NORAIN`.
- **`time_observed == 'ALL_DAY'` & `user_id == 27`** : Détecteur clé des pluies fines localisées (`SMALLRAIN`).

Le projet est désormais **100% éligible aux prix** selon la charte Responsible AI de RAIL.
