# 🌿 Savoirs Indigènes et Contexte Météorologique au Ghana

> **Projet :** [[00_Overview|Ghana's Indigenous Intel Challenge]]  
> **Tags :** #domain-knowledge #ghana #climatology #indigenous-indicators

---

## 🌍 Contexte Géographique et Climatologique

### Bassin du fleuve Pra (Pra River Basin)
- Situé au sud du Ghana (régions Centrale, Orientale et Ashanti).
- Région à climat tropical de transition avec un régime pluviométrique **bimodal** :
  - **Grande saison des pluies :** d'avril à juin / début juillet.
  - **Petite saison des pluies :** de septembre à novembre.
  - **Saisons sèches :** décembre à février (soumise à l'Harmattan, vent sec du Sahara) et un répit relatif en août.

---

## 🦅 Les Indicateurs Écologiques Indigènes (IEI)

Les agriculteurs s'appuient sur 4 grandes familles d'observations empiriques :

### 1. Phénomènes Célestes et Astronomiques
- **Lune :** Position, orientation du croissant, présence d'un halo lumineux ("lune couronnée" souvent annonciatrice de pluie imminente).
- **Étoiles :** Constellations visibles, éclat et scintillement dans la nuit.

### 2. Facteurs Atmosphériques et Thermiques
- **Vents :** Direction et fraîcheur matinale des vents (ex: vents maritimes humides du sud-ouest vs vents secs du nord-est).
- **Chaleur / Humidité perçue :** Sensation de lourdeur ou de fraîcheur soudaine au crépuscule ou à l'aube.
- **Formations nuageuses :** Couleur des nuages à l'horizon (sombre, blanc cotonneux, rougeoiement au coucher du soleil).

### 3. Flore et Comportement Végétal
- **Feuillage :** Enroulement des feuilles, orientation tombante avant l'orage.
- **Floraison :** Floraison précoce de certaines essences autochtones indiquant l'amorce des précipitations.

### 4. Faune et Comportement Animal
- **Oiseaux :** Chants spécifiques d'espèces d'oiseaux locales, vols bas avant une baisse de pression atmosphérique.
- **Insectes :** Déplacement en colonnes des fourmis, vol intense d'insectes à proximité du sol.

---

## 💡 Idées pour le Feature Engineering

1. **Variables cycliques temporelles :**
   - Heure de la journée (sin/cos de l'heure pour capturer la continuité entre 23h et 01h).
   - Mois / saison (marquage des deux pics de saison des pluies au Ghana).
2. **Traitement textuel (TF-IDF / N-grammes) :**
   - Extraire les mots-clés prédominants dans les descriptions des agriculteurs (`wind`, `cloud`, `dark`, `morning`, `birds`, `moon`, etc.).
3. **Agrégations par agriculteur / communauté :**
   - Certains agriculteurs ont des habitudes spécifiques d'observation (le "Star-Gazer", le "Tree Whisperer"). Mesurer le taux de succès historique par observateur peut fournir un signal fort.
