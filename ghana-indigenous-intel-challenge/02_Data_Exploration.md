# 📈 Exploration et Structure des Données

> **Projet :** [[00_Overview|Ghana's Indigenous Intel Challenge]]  
> **Tags :** #data-exploration #eda #preprocessing

---

## 🗃️ Vue d'ensemble des fichiers

Les données ont été chargées dans [`data/`](file:///home/franck-alpha/Bureau/research/zindi/ghana-indigenous-intel-challenge/data/) :
- `train.csv` : **10 928 lignes**, 12 colonnes.
- `test.csv` : **2 732 lignes**, 11 colonnes.
- `SampleSubmission.csv` : **2 732 lignes**, 2 colonnes (`ID`, `Target`).
- `Indigenous_Weather_Forecasting-Starter_Notebook.ipynb` : Présent dans [`notebooks/`](file:///home/franck-alpha/Bureau/research/zindi/ghana-indigenous-intel-challenge/notebooks/).

---

## 🎯 Décomposition de la Variable Cible

Dans ce challenge, les 4 classes de précipitations réelles mesurées sont :
1. `NORAIN` : **9 612** occurrences (**87.96%**)
2. `MEDIUMRAIN` : **761** occurrences (**6.96%**)
3. `HEAVYRAIN` : **315** occurrences (**2.88%**)
4. `SMALLRAIN` : **240** occurrences (**2.20%**)

> [!CAUTION]
> **Déséquilibre de classe extrême :**  
> Près de 88% des observations correspondent à l'absence de pluie.  
> Comme la métrique est le **Macro F1-Score**, prédire systématiquement `NORAIN` donne un score catastrophique (~0.23).  
> Il est impératif d'utiliser des modèles sensibles aux classes minoritaires (`class_weight='balanced'`, focales, ou calibration des seuils de probabilité OOF).

---

## ⏳ Découverte Clé : Le Split Temporel (Out-of-Time)

L'analyse des dates montre que Zindi a effectué une scission **chronologique** :
- **Entraînement :** du 30 mai 2025 au 20 juillet 2025.
- **Test :** du 20 juillet 2025 au 04 août 2025.

### Implications majeures :
- **Validation temporelle recommandée :** Créer un split de validation basé sur les dernières semaines de train (ex: juillet 2025) en plus du Stratified K-Fold pour tester la robustesse aux changements saisonniers.
- **Nouveaux utilisateurs / communautés :** Le test contient 5 utilisateurs (`87, 92, 93, 94, 95`) et 5 communautés inédites (`Domeabra`, `ODUMASI`, `Asonkore`, `jimiso`, `Dompim`). Les modèles ne doivent donc pas sur-apprendre sur les ID d'utilisateurs.

---

## 🔬 Variables et Indicateurs Écologiques

- `forecast_length` : Prévision sur 12 heures (4 430 cas) ou 24 heures (6 498 cas).
- `confidence` : Degré de certitude renseigné (`0.3`, `0.6`, `1.0`).
- `predicted_intensity` : Intensité estimée par le fermier (`0.0`, `0.33`, `0.66`, `1.0`).
- `indicator` : Principaux signes relevés (`clouds`, `sun`, `heat`, `fog`, `wind`, `moon`, `dew`, `star`, `thunder`, `lightning`).
- `community` & `district` : Nécessitent un nettoyage de casse et de suppression des espaces (`strip()`, `lower()`).
