# 🔬 Recherche Approfondie : Fusion du Savoir Indigène et de la Climatologie Physique

> **Projet :** [[00_Overview|Ghana's Indigenous Intel Challenge]]  
> **Tags :** #research #orographic-effect #bayesian-trust #geoclimatology #pra-basin

---

## 🏔️ 1. Découverte Orographique Majeure : Le Plateau d'Atiwa vs Basses Terres d'Obuasi

L'analyse géoclimatique des trois districts révèle une divergence fondamentale dans la physique des précipitations :

| District | Caractéristique Géographique | % NORAIN | % HEAVYRAIN | % MEDIUMRAIN | Rôle Climatique |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`atiwa_west`** | Plateau forestier d'Atiwa (altitude ~700-800m) | **75.1%** | **6.4%** | **15.2%** | **Foyer orographique principal** (pluie de relief) |
| **`assin_fosu`** | Bassin moyen du fleuve Pra (collines modérées) | **97.9%** | **0.1%** | **0.4%** | Précipitations convectives diffuses |
| **`obuasi_east`** | Basses terres minières d'Ashanti | **100.0%** | **0.0%** | **0.0%** | Période sèche absolue (0 mm de pluie) |

### Implication Modèle :
- Prédire de la pluie (`MEDIUM` ou `HEAVY`) dans le district **`obuasi_east`** est une erreur physique contre-productive.
- La quasi-totalité des pluies torrentielles et modérées se concentre sur les villages d'**Atiwa West** (`Kwabeng`, `Tumfa`, `Asamama`) lorsque les vents humides du Golfe de Guinée heurtent le massif forestier.

---

## 👤 2. Modèle de Confiance Bayésienne des Agriculteurs (Observer Reliability Index)

L'évaluation de la justesse historique de chaque observateur montre une disparité significative d'expertise empirique :

- **Observateurs Haute Précision :**
  - **Fermier `20` :** Précision = **80.0%**, Recall = **100%** (quand il signale une pluie, elle a lieu dans 8 cas sur 10).
  - **Fermier `18` :** Précision = **61.1%**, 1 333 observations de terrain validées.
  - **Fermier `23` :** Précision = **57.1%**.
  - **Fermier `27` :** Détecteur quasi-exclusif de la pluie fine (`SMALLRAIN`) lors d'observations journalières complètes (`ALL_DAY`).

- **Observateurs à Bruit Élevé (Fausses Alertes) :**
  - **Fermiers `66`, `62`, `22` :** Précision = **0.0%** (leurs annonces de pluie ne se matérialisent pas sur le pluviomètre).

### Implication Modèle :
Un modèle pondéré par la fiabilité bayésienne de l'observateur $P(\text{Pluie} \mid \text{Agriculteur}_i)$ neutralise le bruit des déclarations subjectives sans perdre les signaux des maîtres-observateurs.

---

## 📍 3. Cohérence Spatiale à l'Échelle du Village (Consensus Pluviométrique)

Dans **97,1% des cas d'entraînement**, toutes les prévisions enregistrées au sein d'une même communauté pour une date et une durée données reçoivent **exactement la même étiquette cible** sur le pluviomètre du village.

$$\text{Target}(C, D, L) \approx \operatorname{Mode} \left( \hat{y}_1, \dots, \hat{y}_n \right)$$

En appliquant un lissage spatial par consensus, les prédictions isolées contradictoires causées par une mauvaise interprétation d'un seul individu sont corrigées par l'ensemble de la communauté.

---

## 🌙 4. Décodage Sémantique des Indicateurs Écologiques Rares

L'analyse croisée des 10 indicateurs écologiques et de leurs descriptions textuelles montre :
1. **La Chaleur Atmosphérique (`heat`) :** Forte association avec les orages violents (`HEAVYRAIN` 9.4% vs 2.8% moyenne). L'air surchauffé au sol déclenche l'instabilité convective.
2. **Le Brouillard et la Rosée (`fog`, `dew`) :** Association à **100% avec `NORAIN`**. La présence de rosée ou de brouillard matinal indique un ciel nocturne dégagé et une inversion thermique stable, inhibant la pluie diurne.
3. **Le Halo Lunaire ("Moon with rainbow-like rings") :** Témoin optique de cristaux de glace dans les cirrostratus de haute altitude, souvent avant-coureur d'un front perturbé.
