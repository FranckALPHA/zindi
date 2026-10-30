# 📐 Calculateur de Score Officiel et Analyseur de Soumissions

> **Projet :** [[00_Overview|Ghana's Indigenous Intel Challenge]]  
> **Script :** [`src/evaluate_submission.py`](file:///home/franck-alpha/Bureau/research/zindi/ghana-indigenous-intel-challenge/src/evaluate_submission.py)  
> **Tags :** #metric #macro-f1 #score-calculator #verification

---

## 🎯 Définition de la Métrique Officielle Zindi

La compétition évalue les prédictions via le **Macro F1-Score** calculé sur les 4 classes :
`NORAIN`, `SMALLRAIN`, `MEDIUMRAIN`, `HEAVYRAIN`.

Pour chaque classe $c \in \{1, 2, 3, 4\}$ :
$$P_c = \frac{TP_c}{TP_c + FP_c}, \quad R_c = \frac{TP_c}{TP_c + FN_c}$$
$$F1_c = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

Le score final est la moyenne arithmétique non-pondérée :
$$\mathbf{\text{Macro F1}} = \frac{1}{4} \left( F1_{\text{NORAIN}} + F1_{\text{SMALL}} + F1_{\text{MEDIUM}} + F1_{\text{HEAVY}} \right)$$

> [!WARNING]
> Si une classe n'a aucune prédiction positive dans le fichier de soumission ($TP_c = 0$), son F1 vaut automatiquement $0.0$, ce qui détruit le score moyen même si les autres classes sont à 100%.

---

## 🛠️ Utilisation de notre Calculateur Local

Le script [`src/evaluate_submission.py`](file:///home/franck-alpha/Bureau/research/zindi/ghana-indigenous-intel-challenge/src/evaluate_submission.py) est prêt et autonome :

### 1. Comparer deux soumissions entre elles
Permet de voir le taux d'accord exact (ex: avec le benchmark validé à 0.9587) et la matrice de transition :
```bash
python3 src/evaluate_submission.py --compare submissions/submission_starter_benchmark.csv submissions/submission_benchmark_5seed_ensemble.csv
```

### 2. Évaluer des prédictions contre la vérité terrain (CV locale)
Affiche le rapport de classification complet (précision, rappel, F1 par classe, support) et la matrice de confusion :
```bash
python3 src/evaluate_submission.py --eval data/val_true.csv data/val_pred.csv
```

---

## 🔬 Exemple de Diagnostic Réalisé

La comparaison entre notre benchmark validé (**`0.9587`**) et le Super Ensemble à 5 graines montre :
- **Taux de concordance :** **`98.43%`** (2 689 prédictions strictement identiques).
- **Stabilité absolue :** 100% des 2 518 `NORAIN` et des 9 `SMALLRAIN` sont préservés.
- **Micro-ajustement :** Seulement **43 lignes** (1.57%) sont réarbitrées entre `MEDIUM` et `HEAVY` grâce au consensus de 1 500 arbres.
