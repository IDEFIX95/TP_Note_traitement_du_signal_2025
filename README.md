# TP_Note_traitement_du_signal_202
# 🎧 Projet – Traitement du signal audio (Mixage, FFT, Filtrage, IFFT)

Projet académique **CY Tech — Traitement du signal**  
Développé en **Python**

L’objectif du projet est d’illustrer les opérations classiques du traitement du signal audio : séparation d’un morceau en deux pistes **(voix et instrumental)**, analyse fréquentielle **(FFT)**, **filtrage** de certaines fréquences, puis **reconstruction** finale via l’**IFFT** et un **mixage contrôlé**.

On part d’une musique, on l’analyse, on la transforme et on tente de reconstruire un signal audio cohérent à partir des composantes modifiées.

---

## 🎯 Objectifs du projet

Ce projet illustre plusieurs opérations de **traitement du signal audio** en Python :

1. Lecture et préparation de fichiers audio (`.wav`)
2. Création de **mixes** à partir de pistes vocales et instrumentales
3. Analyse fréquentielle (**FFT**)
4. Filtrage passe-bas
5. Reconstruction du signal dans le domaine temporel (**IFFT**)


---
## 📁 Structure du projet

```text
.
├── Mix.py        # Lecture des WAV, normalisation, génération de mixes
├── FFT.py        # Calcul et affichage du spectre (FFT)
├── Filter.py     # Filtre passe-bas + affichage signal filtré
├── IFFT.py       # Reconstruction du signal par IFFT
└── Dataset/
    ├── Vocals/          # Pistes vocales (.wav)
    ├── Instrumentals/   # Pistes instrumentales (.wav)
    └── Mixes/           # Dossier de sortie pour les mixes générés
```

---

## 🧰 Outils requis sur chaque machine

- Python 3.x
- Bibliothèques Python :
    - `numpy`
    - `scipy`
    - `matplotlib`

Installation des dépendances (par exemple) :
```bash
pip install numpy scipy matplotlib
```
---

## 🚀 Utilisation 
###1️⃣ Générer des mixes audio (`Mix.py`)

Ce module :

- lit les fichiers `.wav` dans `Dataset/Vocals` et `Dataset/Instrumentals`

- convertit les signaux en **mono** et les **normalise** entre -1 et 1

- crée des mixes pondérés (par défaut : 0.7 pour la voix, 0.3 pour l’instrumental)

- renormalise le mix final

- sauvegarde les fichiers dans `Dataset/Mixes` sous la forme :
`mix_<nom_vocal>_<nom_instru>.wav`

Pour lancer la génération de tous les mixes :
```bash
python Mix.py
```
🟢 **Résultats** :
Les fichiers `.wav` générés se trouvent dans :
```bash
Dataset/Mixes/
```

---

## 2️⃣ Analyse fréquentielle – FFT (`FFT.py`)

Ce module propose :

- une fonction `compute_fft(signal, sampling_rate)` qui :

    - calcule la FFT du signal

    - retourne les **fréquences** et les **magnitudes** normalisées

- une fonction `plot_signal_and_spectrum(t, signal, fft_frequencies, fft_magnitudes)` qui :

    - affiche le signal dans le domaine temporel

    - affiche le spectre de magnitude dans le domaine fréquentiel

Exemple (mode script, si tu complètes la génération du signal dans le `main`) :

```bash
python FFT.py
```

🟢 **Résultats** :
Les graphiques s’affichent dans une fenêtre `matplotlib` (non sauvegardés par défaut).


---

## 3️⃣ Filtrage passe-bas (`Filter.py`)

Ce module permet :

- de définir un filtre passe-bas de Butterworth avec `butter_lowpass(cutoff, fs, order)`

- d’appliquer ce filtre à un signal avec `lowpass_filter(data, cutoff, fs, order)`

- de tracer le signal original et le signal filtré avec `plot_signals(original_signal, filtered_signal, t)`

En mode script (une fois l’indentation du `if __name__ == "__main__":` corrigée si besoin), le fichier :

- crée un signal de test composé de plusieurs sinusoïdes (5, 50, 120 Hz)

- applique un filtre passe-bas (par ex. coupure à 50 Hz)

- affiche les signaux avant / après filtrage

Pour lancer l’exemple :

```bash
python Filter.py
```
🟢 **Résultats** :
Deux graphiques `matplotlib` s’affichent :

1. Signal original
2. Signal filtré (basses fréquences conservées)

---

## 4️⃣ Reconstruction temporelle – IFFT (`IFFT.py`)

Ce module contient :

- `compute_ifft(fft_values)` : reconstruit un signal temporel à partir de ses valeurs FFT (et renvoie la partie réelle)

- `plot_time_signal(t, time_signal)` : affiche le signal reconstruit dans le domaine temporel

En mode script, le fichier :

- génère un signal test (somme de sinusoïdes)

- calcule sa FFT

- applique l’IFFT

- affiche le signal reconstruit

Pour lancer l’exemple :
```bash
python IFFT.py
```

🟢 **Résultats** :
Un graphique `matplotlib` affiche le signal temporel reconstruit.

---

## 📦 Sorties du projet

| Module | Résultat produit |
|--------|------------------|
| `Mix.py` | Fichiers audio recomposés (`Dataset/Reconstructed/`) |
| `FFT.py` | Graphiques temporel + spectre (matplotlib) |
| `Filter.py` | Signals filtrés + visualisation |
| `IFFT.py` | Signal reconstruit en domaine temporel |
