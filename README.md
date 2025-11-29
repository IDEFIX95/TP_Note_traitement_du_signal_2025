# 🎧 Projet — Séparation Voix / Instrumental 

Projet académique **CY Tech — Traitement du signal**  
Développé en **Python (NumPy, SciPy, Librosa, Matplotlib, mir_eval)**

Réalisé par : Rayane Manseur Rayan Hussein Emine Ould Agatt Florian Vo Romain Bowé Clément Rimbeuf Anthusan Srikaran

L’**objectif** du projet est de séparer un morceau audio en deux composantes :
la **voix** et l’**instrumental**, en utilisant des méthodes classiques du traitement du signal :
**STFT, masques temps–fréquence, filtrage fréquentiel, HPSS, variation temporelle, reconstruction, et évaluation quantitative (SDR, SIR, SAR).**

Ce pipeline complet permet d’aller depuis les données brutes, jusqu’à la reconstruction et l’analyse comparative des méthodes.

---

## 🎯 Objectifs du projet

Ce projet illustre plusieurs opérations de **traitement du signal audio** :

1. Comprendre et appliquer les bases du **traitement du signal audio**.
2. Manipuler la **DFT / FFT, STFT**, **masques fréquences / temps**.
3. Implémenter plusieurs méthodes de **séparation de sources**.
4. Générer un dataset contrôlé (mélanges voix + instrumental).
5. Reconstruire des signaux via **ISTFT**.
6. Évaluer les méthodes via les métriques standard (**SDR, SIR, SAR**).
7. Visualiser les masques et les résultats


---
## 📁 Structure du projet

```text
.
├── src/
│   ├── mix.py                # Génération automatique des mixes
│   ├── separation.py         # Implémentation des 4 méthodes de séparation
│   ├── stft_utils.py         # Fonctions STFT / ISTFT / normalisation
│   ├── metrics.py            # Calcul des métriques SDR / SIR / SAR
│   ├── plots.py              # Affichage des masques et signaux
│   └── main.py               # Pipeline principal (lecture → séparation → save → metrics)
│
├── data/
│   ├── Vocals/               # Pistes vocales de référence
│   ├── Instrumentals/        # Pistes instrumentales de référence
│   └── Mixes/                # Mixes générés automatiquement
│
└── results/
    ├── <nom_du_mix>/
    │   ├── vocals_est.wav
    │   ├── instru_est.wav
    │   ├── mask_bande.png
    │   ├── mask_hpss.png
    │   ├── mask_variability.png
    │   └── mask_hybride.png
    └── metrics_globales.csv

```

---

## 🧰 Outils requis sur chaque machine

- Python 3.x
- Bibliothèques Python :
    - `numpy`
    - `scipy`
    - `matplotlib`
    - `librosa`
    - `mir_eval`

Installation des dépendances (par exemple) :
```bash
pip install numpy scipy matplotlib librosa mir_eval
```
---

## 🚀 Pipeline complet
###1️⃣ Génération automatique des mixes (`mix.py`)

Ce script :

- charge la piste vocale et instrumentale
- convertit en mono si nécessaire
- normalise chaque signal
- applique un mix linéaire :
      `mix = α·voix + β·instrumental`
- sauvegarde le mix dans `data/mix`

Pour lancer la génération de tous les mixes :
```bash
python src/mix.py
```

---

###2️⃣ Analyse temps–fréquence via STFT

Nous utilisons :

- `librosa.stft` pour obtenir le spectrogramme complexe
- module
  `S(f,t) = |S(f,t)| e^{iϕ(f,t)}`

Toutes les méthodes de séparation travaillent sur le spectrogramme, jamais sur le signal temps direct.

---

###3️⃣ Méthodes de séparation (`separation.py`)

Nous avons implémenté 4 méthodes :

- Filtre en bande (80–4000 Hz) : simple filtre fréquentiel basé sur la gamme vocale.

- HPSS (Harmonic / Percussive Source Separation) : séparation par filtres médians :
        - composante harmonique → voix
        - composante percussive → instrumental

- Masque par variabilité temporelle : analyse des variations rapides du module du spectre.

- Masque hybride (méthode finale) : combinaison pondérée des 3 précédentes. **Meilleure méthode selon notre étude.**

Chaque méthode génère :
- `vocals_est.wav`
- `instru_est.wav`

Ainsi que les masques (png), sauvegardés via `Vizualitation.py`.

---

###4️⃣ Reconstruction temporelle (`src/other/IFFT.py`)

- ISTFT via librosa.istft
- Tests de cohérence sur signaux simples

Dans le pipeline réel, la reconstruction est déclenchée depuis `separation.py`.


---

###5️⃣ Visualisation (`src/other/Vizualitation.py`)

Génère automatiquement :

- spectrogrammes
- masques de séparation (voix/instru)
- courbes temporelles

---

###6️⃣ Évaluation SDR / SIR / SAR

Dans `separation.py` :

- compare chaque source estimée aux sources réelles
- utilise `mir_eval.separation.bss_eval_sources`
- génère un CSV global de résultats :

```bash
results_metrics.csv
```
---

###📊 Résultats (résumé du rapport)

- **Hybride** = meilleure méthode (SDR ≈ 9 dB)

- **HPSS** = bon compromis

- **Filtre bande** = simple mais limité

- **Variabilité** = meilleur SIR mais détruit le signal → mauvais SDR/SAR

---

##▶️ Exécution du pipeline complet

```bash
python src/separation.py
```

Ce script :

- charge les mixes

- applique les 4 méthodes

- reconstruit les sources

- génère les masques + figures

- calcule les métriques
