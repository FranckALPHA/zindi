# 🧠 Stratégie de Modélisation et Validation

> **Projet :** [[00_Overview|Ghana's Indigenous Intel Challenge]]  
> **Tags :** #machine-learning #cross-validation #feature-engineering #ensembling

---

## 📐 Schéma de Validation Croisée

La métrique officielle étant le **Macro F1-Score**, la validation locale doit reproduire fidèlement cette évaluation :
- **Stratified K-Fold (5 ou 10 Folds)** basé sur la classe cible (`Target`).
- Si l'identifiant agriculteur/communauté est présent : tester également un **GroupKFold** pour vérifier la capacité de généralisation à de nouveaux observateurs non vus lors de l'entraînement.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 🛠️ Feature Engineering Prioritaire

1. **Encodage Temporel Cyclique :**
   ```python
   df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
   df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
   df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
   df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
   ```
2. **Features Textuelles sur les Notes de Terrain :**
   - Application d'un **TF-IDF Vectorizer** (sur les 50 à 100 n-grammes les plus pertinents).
   - Réduction dimensionnelle par TruncatedSVD si nécessaire.
3. **Comptages et Ratios d'Indicateurs :**
   - Nombre total d'indicateurs écologiques cochés/rapportés par observation.
   - Combinaisons fréquentes (ex: vent fort + nuage bas).

---

## 🤖 Architectures de Modèles

Nous adoptons une approche incrémentale :
1. **Baseline 0 :** Dummy / Logistic Regression sur indicateurs bruts.
2. **Baseline 1 :** **LightGBM** multiclasse avec gestion des poids de classes (`class_weight='balanced'`).
3. **Baseline 2 :** **CatBoost** (performant sur les variables catégorielles indigènes) et **XGBoost**.
4. **Ensemble (Stacking) :** Méta-modèle Logistic Regression combinant les probabilités prédites out-of-fold.

---

## 🎯 Post-traitement et Optimisation du Macro F1

Le simple `argmax` sur les probabilités prédites pénalise souvent les classes minoritaires (`HEAVY`).  
- **Optimisation de seuils (Nelder-Mead ou recherche par grille) :** Ajuster les seuils de décision pour maximiser directement le Macro F1 sur les prédictions hors-pli (OOF).
