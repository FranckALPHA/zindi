# 🌧️ Ghana's Indigenous Intel Challenge (Cameroon Community Hack)

> **Événement :** Monthly Event — Cellule IA ENSPY (5GI)  
> **Lien officiel du challenge :** [Zindi Competition Page](https://zindi.world/competitions/ghana-indigenous-intel-challenge-cameroon-community-hack)  
> **Mot de passe du hackathon :** `Gi27`  
> **Prix :** 50 000 XAF  
> **Clôture stricte :** **Vendredi 30 octobre 2026 à 12h15** *(Urgence : ~3h30 restantes)*  
> **IA autorisée :** Oui, mention explicite de l'organisation.  
> **Tags :** #enspy #zindi #meteorology #indigenous-knowledge #hackathon

---

## 🎯 Objectif du problème

Les agriculteurs locaux du bassin du fleuve **Pra** au **Ghana** utilisent depuis des générations des indicateurs écologiques et célestes (direction du vent, comportements des arbres et des oiseaux, cycle lunaire, étoiles, chaleur) pour anticiper les précipitations. Grâce à l'application mobile **Smart Indigenous Weather (SIW)** conçue par RAIL, ces observations traditionnelles ont été collectées de manière structurée et corrélées avec les relevés réels de pluviomètres.

**Votre mission :**  
Construire un modèle de classification prédisant le type de pluie attendu dans les **12 à 24 heures** à venir, en s'appuyant sur les indicateurs écologiques indigènes saisis par les agriculteurs.

---

## 📊 Format des prédictions et Métrique

- **Métrique d'évaluation :** **F1-Score** (Macro F1 pour les classes multi-catégorielles).
- **Format du fichier de soumission :** Deux colonnes exactement :
  - `ID` : identifiant du test (ex: `ID_ekOZy_12hr_rain_type`)
  - `Target` : catégorie prédite (ex: `NORAIN`, `SMALL`, `MODERATE`, `HEAVY`)
  
Exemple de soumission :
```csv
ID,Target
ID_ekOZy_12hr_rain_type,NORAIN
ID_3RVHW_12hr_rain_type,SMALL
```

---

## ⚖️ Règles essentielles et Contraintes

1. **Outils Open Source uniquement :** Pas de services tiers propriétaires ou AutoML automatisé.
2. **Explicabilité obligatoire (RAIL - Responsible AI) :**
   - Le modèle doit être accompagné d'une méthode d'explicabilité (ex: **SHAP**, **LIME** ou **Grad-CAM**).
   - Les modèles finaux doivent pouvoir être convertis au format **ONNX** ou **TFLite**.
3. **Limites de soumissions :** 2 soumissions par jour, maximum 30 au total.
4. **Split Public / Privé :** ~30% du jeu de test sur le Public Leaderboard, 70% sur le Private Leaderboard (révélé à la clôture).

---

## 🔗 Liens vers les autres notes du dossier

- 🌿 [[01_Domain_Knowledge|01. Savoirs indigènes et météorologie au Ghana]]
- 📈 [[02_Data_Exploration|02. Exploration et structure des données]]
- 🧠 [[03_Modeling_Strategy|03. Stratégie de modélisation et Feature Engineering]]
- 🔍 [[04_Explainability_RAIL|04. Explicabilité Responsible AI (SHAP & ONNX)]]
- 📝 [[05_Experiment_Log|05. Journal des expériences et scores Leaderboard]]
- 🔬 [[06_Personal_Research_Scientific_Insights|06. Recherche Approfondie (Orographie & Fiabilité Bayésienne)]]
- 📐 [[07_Score_Calculator_&_Metrics|07. Calculateur de Score Officiel et Comparateur]]
