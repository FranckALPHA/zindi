# 🏆 Zindi Machine Learning & Data Science Research Hub

Ce dépôt centralise l'ensemble des travaux de recherche, architectures d'ingénierie des caractéristiques (*feature engineering*), pipelines de modélisation et journaux d'expérimentation pour les compétitions sur la plateforme **Zindi**.

---

## 📂 Compétitions Documentées

| Compétition | Domaine & Type | Métrique | Statut | Dossier |
| :--- | :--- | :---: | :---: | :--- |
| **Financial / Liquidity Stress Prediction** | Fintech / Longitudinal Panel (Tabular) | PR-AUC / ROC-AUC | 🟢 Actif (Score: **0.70807**) | [`financial-stress-prediction-challenge/`](financial-stress-prediction-challenge/) |
| **Ghana's Indigenous Intel Challenge** | Agro-météorologie / NLP & Cyclique | Macro F1-score | 🏁 Terminé (Score: **0.95874**) | [`ghana-indigenous-intel-challenge/`](ghana-indigenous-intel-challenge/) |

---

## 🏗️ Structure Standard d'un Projet

Pour chaque compétition, un dossier dédié structure l'intégralité du cycle de vie du projet :

```text
├── 00_Overview.md            # Définition du problème, métrique officielle, règles et calendrier
├── 01_Domain_Knowledge.md    # État de l'art métier, recherche bibliographique et contexte sectoriel
├── 02_Data_Exploration.md    # Dictionnaire des variables, analyse exploratoire (EDA), intégrité
├── 03_Modeling_Strategy.md   # Stratégie de validation croisée sans fuite, architectures modèles
├── 04_Explainability_RAIL.md # Explicabilité des décisions (SHAP) et conformité Responsible AI
├── 05_Experiment_Log.md      # Journal chronologique des expérimentations (CV vs Leaderboard)
├── data/                     # Ignoré par Git (données propriétaires Zindi sous licence)
├── src/                      # Code source modulaire et reproductible (scripts d'entraînement, FE)
└── submissions/              # Historique des soumissions générées et prédictions OOF
```

---

## 🛡️ Règles de Conformité & Données

- Conformément aux règles officielles de la plateforme Zindi et à la licence CC-BY-SA 4.0, les fichiers de données brutes (`train.csv`, `test.csv`) ainsi que les soumissions volumineuses sont strictement ignorés via `.gitignore` pour préserver l'intégrité de la compétition.
- Les scripts utilisent uniquement des bibliothèques open-source et des graines aléatoires (*seeds*) fixes garantissant une stricte reproductibilité des résultats lors de l'audit de code (Code Review).
