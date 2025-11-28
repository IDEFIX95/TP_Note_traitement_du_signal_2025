import os
import inspect
import glob

import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

# imports métriques
from museval.metrics import bss_eval_sources

# === Imports Demucs (commentés pour l’instant) ===
# import torch
# from demucs.pretrained import get_model
# from demucs.apply import apply_model


# =========================
#  PARAMETRES GLOBAUX
# =========================

SR = 22050      # fréquence d'échantillonnage
N_FFT = 2048    # taille de la FFT
HOP = 512       # pas de la STFT
WINDOW = "hann" # fenêtre de Hann

# =========================
#  FONCTIONS ÉLÉMENTAIRES
# =========================

def load_audio(path, sr=SR):
    """
    Charge un fichier audio WAV.
    - Si le fichier est stéréo, le convertit en mono (moyenne des canaux).
    - NE NORMALISE PAS (on le fait à la fin des séparations).
    """
    y, sr_loaded = librosa.load(path, sr=sr, mono=False)

    # Si stéréo → downmix manuel
    if y.ndim == 2:
        print(f"  -> fichier stéréo détecté ({y.shape[0]} canaux), conversion en mono...")
        y = np.mean(y, axis=0)

    return y, sr_loaded


def stft(y):
    """STFT : temps -> fréquences."""
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    mag, phase = librosa.magphase(S)
    return S, mag, phase


def istft(S):
    """iSTFT : fréquences -> temps."""
    return librosa.istft(S, hop_length=HOP, window=WINDOW)


def normalize(y):
    """Normalise un signal temporel entre -1 et 1."""
    m = np.max(np.abs(y)) + 1e-8
    return y / m

# ===============================
#  MÉTHODES DE SÉPARATION
# ===============================

# 1) Masque "porte" bande de fréquences
def band_masks(sr, n_fft, f_low=80, f_high=4000):
    """Masque fréquentiel binaire basé sur la bande 80 Hz – 4 kHz (voix)."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    voice_band = (freqs >= f_low) & (freqs <= f_high)

    M_voice = voice_band[:, None].astype(float)  # (freq, 1) → broadcast sur le temps
    M_instr = 1.0 - M_voice
    return M_voice, M_instr


def separate_band(mix, sr):
    S_mix, mag_mix, phase_mix = stft(mix)
    M_voice, M_instr = band_masks(sr, N_FFT)

    S_voice = M_voice * S_mix
    S_instr = M_instr * S_mix

    y_voice = normalize(istft(S_voice))
    y_instr = normalize(istft(S_instr))
    return y_voice, y_instr


# 2) HPSS (Harmonique / Percussif)
def separate_hpss(mix):
    """
    HPSS : décompose le spectre en composante harmonique (H) et percussive (P)
    via des filtres médians (stable dans le temps vs stable en fréquence).
    """
    S_mix, mag_mix, phase_mix = stft(mix)
    H, P = librosa.decompose.hpss(S_mix)

    y_h = normalize(istft(H))  # partie harmonique
    y_p = normalize(istft(P))  # partie percussive
    return y_h, y_p


# 3) Masque par variabilité temporelle
def variability_mask(mag_mix, alpha=2.0):
    """
    Masque temps-fréquence basé sur la variabilité :
    M_voice(f,t) ∝ |M(f,t+1) - M(f,t)|^alpha normalisé.
    Une augmentation de alpha rend le masque plus agressif.
    """
    # dérivée temporelle locale
    diff_t = np.abs(np.diff(mag_mix, axis=1))  # (freq, time-1)

    # on remet à la même taille en temps
    var_local = np.pad(diff_t, ((0, 0), (0, 1)), mode='edge')

    # normalisation dans [0,1]
    var_norm = var_local / (np.max(var_local) + 1e-8)

    # masque soft - alpha > 1 accentue les zones très variables
    M_voice = var_norm ** alpha
    M_instr = 1.0 - M_voice
    return M_voice, M_instr


def separate_variability(mix):
    S_mix, mag_mix, phase_mix = stft(mix)

    M_voice, M_instr = variability_mask(mag_mix, alpha=2.0)

    S_voice = M_voice * S_mix
    S_instr = M_instr * S_mix

    y_voice = normalize(istft(S_voice))
    y_instr = normalize(istft(S_instr))
    return y_voice, y_instr


# 4) Masque hybride : bande douce + HPSS + variabilité
def soft_band_mask(sr, n_fft, f_low=80, f_high=4000, width=200):
    """Version soft du masque de bande (pentes lissées)."""
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    low_slope  = np.clip((freqs - (f_low - width)) / width, 0, 1)
    high_slope = np.clip(((f_high + width) - freqs) / width, 0, 1)

    band = low_slope * high_slope  # 0..1
    M_voice = band[:, None]        # (freq, 1)
    M_instr = 1.0 - M_voice
    return M_voice, M_instr


def hybrid_mask(S_mix, mag_mix, sr):
    # 1) prior bande douce
    M_band_voice, _ = soft_band_mask(sr, N_FFT)

    # 2) HPSS → mag harmonique / (harmonique + percussif)
    H, P = librosa.decompose.hpss(S_mix)
    mag_H = np.abs(H)
    mag_P = np.abs(P)
    M_hpss = mag_H / (mag_H + mag_P + 1e-8)

    # 3) Variabilité temps-fréquence
    M_var_voice, _ = variability_mask(mag_mix, alpha=2.0)

    # M_band_voice : (freq,1) -> (freq,time)
    M_band_voice_full = np.repeat(M_band_voice, mag_mix.shape[1], axis=1)

    # Combinaison multiplicative + normalisation
    M_comb = M_band_voice_full * M_hpss * (0.5 + 0.5 * M_var_voice)
    M_comb = M_comb / (np.max(M_comb) + 1e-8)

    M_voice = M_comb
    M_instr = 1.0 - M_voice
    return M_voice, M_instr


def separate_hybrid(mix, sr):
    S_mix, mag_mix, phase_mix = stft(mix)
    M_voice, M_instr = hybrid_mask(S_mix, mag_mix, sr)

    S_voice = M_voice * S_mix
    S_instr = M_instr * S_mix

    y_voice = normalize(istft(S_voice))
    y_instr = normalize(istft(S_instr))
    return y_voice, y_instr, M_voice


def save_mask_frequency_plot(M_voice, sr, title, out_path):
    """
    Affiche (et sauvegarde) un scatter 2D : fréquences → voix vs instru.
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # moyenne sur le temps
    voice_strength = M_voice.mean(axis=1)  # (n_freqs,)
    voice_bins = voice_strength > 0.5

    plt.figure(figsize=(7, 3))
    # Instru (0)
    plt.scatter(freqs[~voice_bins],
                np.zeros(np.sum(~voice_bins)),
                c="b", s=10, label="Instru")
    # Voix (1)
    plt.scatter(freqs[voice_bins],
                np.ones(np.sum(voice_bins)),
                c="r", s=10, label="Voix")

    plt.yticks([0, 1], ["Instru", "Voix"])
    plt.xlabel("Fréquence (Hz)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ===================================
#  DEMUCS (COMMENTÉ POUR LE MOMENT)
# ===================================

# def separate_demucs(wav_path, out_subdir, sr_target=44100):
#     """
#     Sépare un mix en (voice_demucs, instr_demucs) avec Demucs
#     en utilisant librosa pour le chargement (pas torchaudio).
#     """
#     print("  -> Demucs en cours...")
#
#     # 1) Charger le fichier audio en stéréo
#     y, sr = librosa.load(wav_path, sr=sr_target, mono=False)
#
#     # Convertir en stéréo si mono
#     if y.ndim == 1:
#         y = np.stack([y, y], axis=0)   # (2, T)
#
#     # (channels, time) -> Tensor (1, channels, time)
#     mix = torch.from_numpy(y).float().unsqueeze(0)  # (1, 2, T)
#
#     # 2) Charger le modèle Demucs
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = get_model("htdemucs").to(device)
#     model.eval()
#
#     mix = mix.to(device)
#
#     # 3) Appliquer le modèle
#     with torch.no_grad():
#         estimates = apply_model(model, mix, split=True, overlap=0.25)[0]
#
#     # Ordre : [drums, bass, other, vocals]
#     drums, bass, other, vocals = estimates
#
#     # 4) Accompagnement = drums + bass + other
#     accomp = drums + bass + other
#
#     # 5) Convertir en numpy (T, C)
#     vocals_np = vocals.cpu().numpy().transpose(1, 0)
#     accomp_np = accomp.cpu().numpy().transpose(1, 0)
#
#     # 6) Sauvegarde
#     sf.write(os.path.join(out_subdir, "voice_demucs.wav"), vocals_np, sr_target)
#     sf.write(os.path.join(out_subdir, "instr_demucs.wav"), accomp_np, sr_target)
#
#     print("  -> Fichiers Demucs écrits.")


# ==============
# VISUALISATION
# ==============

def plot_spectrogram(S, sr, title):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(S), ref=np.max),
                             sr=sr, hop_length=HOP,
                             y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_example(file_path):
    mix, sr = load_audio(file_path)
    S_mix, mag_mix, phase_mix = stft(mix)

    y_voice_band, y_instr_band = separate_band(mix, sr)
    S_voice_band, _, _ = stft(y_voice_band)

    plot_spectrogram(S_mix, sr, "Spectrogramme mélange")
    plot_spectrogram(S_voice_band, sr, "Spectrogramme voix (bande)")


# ==============================
#  CHEMINS (robustes au cwd)
# ==============================

# chemin du fichier courant
script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
script_dir  = os.path.dirname(script_path)

# projet_separation = parent de src
BASE_DIR = os.path.abspath(os.path.join(script_dir, ".."))
MIX_DIR  = os.path.join(BASE_DIR, "data", "Mixes")
OUT_DIR  = os.path.join(BASE_DIR, "results")

# répertoires des ground truth
VOCALS_DIR = os.path.join(BASE_DIR, "data", "Vocals")
INSTR_DIR  = os.path.join(BASE_DIR, "data", "Instrumentals")

print("script_dir :", script_dir)
print("BASE_DIR   :", BASE_DIR)
print("MIX_DIR    :", MIX_DIR)
print("OUT_DIR    :", OUT_DIR)

os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
#  BOUCLE PRINCIPALE : SEPARATION
# ==============================

wav_paths = glob.glob(os.path.join(MIX_DIR, "*.wav"))
print("Fichiers .wav trouvés :", wav_paths)

for wav_path in wav_paths:
    print("\n=== Traitement de :", wav_path, "===")
    base = os.path.splitext(os.path.basename(wav_path))[0]
    out_subdir = os.path.join(OUT_DIR, base)
    os.makedirs(out_subdir, exist_ok=True)

    mix, sr = load_audio(wav_path)
    print("  - sr =", sr, " | longueur =", len(mix))

    # STFT unique pour certains masques
    S_mix, mag_mix, phase_mix = stft(mix)

    # 1) FILTRE PAR BANDE
    y_v_band, y_i_band = separate_band(mix, sr)
    sf.write(os.path.join(out_subdir, "voice_band.wav"),  y_v_band, sr)
    sf.write(os.path.join(out_subdir, "instr_band.wav"), y_i_band, sr)
    print("  -> fichiers bande écrits")

    M_voice_band, M_instr_band = band_masks(sr, N_FFT)
    save_mask_frequency_plot(
        M_voice_band,
        sr,
        f"Masque bande (80–4000 Hz) - {base}",
        os.path.join(out_subdir, "mask_band_freqs.png")
    )

    # 2) HPSS
    y_h, y_p = separate_hpss(mix)
    sf.write(os.path.join(out_subdir, "harmonic_hpss.wav"),   y_h, sr)
    sf.write(os.path.join(out_subdir, "percussive_hpss.wav"), y_p, sr)
    print("  -> fichiers HPSS écrits")

    H, P = librosa.decompose.hpss(S_mix)
    mag_H = np.abs(H)
    mag_P = np.abs(P)
    M_voice_hpss = mag_H / (mag_H + mag_P + 1e-8)

    save_mask_frequency_plot(
        M_voice_hpss,
        sr,
        f"Masque HPSS (harmonique) - {base}",
        os.path.join(out_subdir, "mask_hpss_freqs.png")
    )

    # 3) VARIABILITÉ TEMPORELLE
    y_v_var, y_i_var = separate_variability(mix)
    sf.write(os.path.join(out_subdir, "voice_var.wav"),  y_v_var, sr)
    sf.write(os.path.join(out_subdir, "instr_var.wav"),  y_i_var, sr)
    print("  -> fichiers variabilité écrits")

    M_voice_var, M_instr_var = variability_mask(mag_mix, alpha=2.0)

    # 4) MASQUE HYBRIDE
    y_v_hybrid, y_i_hybrid, M_voice_hybrid = separate_hybrid(mix, sr)
    sf.write(os.path.join(out_subdir, "voice_hybrid.wav"), y_v_hybrid, sr)
    sf.write(os.path.join(out_subdir, "instr_hybrid.wav"), y_i_hybrid, sr)
    print("  -> fichiers hybrides écrits")

    # 5) DEMUCS (commenté pour l’instant)
    # try:
    #     separate_demucs(wav_path, out_subdir)
    # except Exception as e:
    #     print(f"  !! ERREUR DEMUCS : {e}")

    # 6) SAUVEGARDE DES PLOTS
    save_mask_frequency_plot(
        M_voice_hybrid,
        sr,
        f"Masque Hybride - {base}",
        os.path.join(out_subdir, "mask_hybrid_freqs.png")
    )

    save_mask_frequency_plot(
        M_voice_var,
        sr,
        f"Masque Variabilité - {base}",
        os.path.join(out_subdir, "mask_var_freqs.png")
    )

    print("  -> graphiques masques écrits dans :", out_subdir)


# ==============================
#  EVALUATION OBJECTIVE (SDR/SIR/SAR)
# ==============================

def evaluate_method(voice_est, instr_est, voice_ref, instr_ref):
    """
    Calcule SDR / SIR / SAR pour [voix, instru].
    Retourne un dict de métriques.
    """
    # Aligner les longueurs
    min_len = min(
        len(voice_est), len(instr_est),
        len(voice_ref), len(instr_ref)
    )

    voice_est  = voice_est[:min_len]
    instr_est  = instr_est[:min_len]
    voice_ref  = voice_ref[:min_len]
    instr_ref  = instr_ref[:min_len]

    # museval attend (nsrc, nsamples)
    ref = np.vstack([voice_ref, instr_ref])
    est = np.vstack([voice_est, instr_est])

    sdr, sir, sar, _ = bss_eval_sources(ref, est)

    return {
        "SDR_voice": float(sdr[0]),
        "SDR_instr": float(sdr[1]),
        "SIR_voice": float(sir[0]),
        "SAR_voice": float(sar[0])
    }

print("\n==============================")
print("  EVALUATION DES SEPARATIONS")
print("==============================")

METHODS = ["Bande", "HPSS", "Variabilité", "Hybride"]
METRICS = ["SDR_voice", "SDR_instr", "SIR_voice", "SAR_voice"]

global_results = {
    method: {metric: [] for metric in METRICS}
    for method in METHODS
}

for wav_path in wav_paths:
    base = os.path.splitext(os.path.basename(wav_path))[0]
    print(f"\n=== Evaluation pour : {base} ===")

    # chemins des pistes de référence
    ref_vocals_path = os.path.join(VOCALS_DIR, base + ".wav")
    ref_instr_path  = os.path.join(INSTR_DIR,  base + ".wav")

    if not (os.path.exists(ref_vocals_path) and os.path.exists(ref_instr_path)):
        print("  -> Références manquantes (Vocals/Instrumentals) : skip.")
        continue

    # chargement des références (mono, pas de normalisation)
    ref_voice, _ = load_audio(ref_vocals_path, sr=SR)
    ref_instr, _ = load_audio(ref_instr_path,  sr=SR)

    out_subdir = os.path.join(OUT_DIR, base)

    def load_est(path):
        if os.path.exists(path):
            y, _ = load_audio(path, sr=SR)
            return y
        return None

    results = {}

    # Bande
    v_est = load_est(os.path.join(out_subdir, "voice_band.wav"))
    i_est = load_est(os.path.join(out_subdir, "instr_band.wav"))
    if v_est is not None and i_est is not None:
        results["Bande"] = evaluate_method(v_est, i_est, ref_voice, ref_instr)

    # HPSS
    v_est = load_est(os.path.join(out_subdir, "harmonic_hpss.wav"))
    i_est = load_est(os.path.join(out_subdir, "percussive_hpss.wav"))
    if v_est is not None and i_est is not None:
        results["HPSS"] = evaluate_method(v_est, i_est, ref_voice, ref_instr)

    # Variabilité
    v_est = load_est(os.path.join(out_subdir, "voice_var.wav"))
    i_est = load_est(os.path.join(out_subdir, "instr_var.wav"))
    if v_est is not None and i_est is not None:
        results["Variabilité"] = evaluate_method(v_est, i_est, ref_voice, ref_instr)

    # Hybride
    v_est = load_est(os.path.join(out_subdir, "voice_hybrid.wav"))
    i_est = load_est(os.path.join(out_subdir, "instr_hybrid.wav"))
    if v_est is not None and i_est is not None:
        results["Hybride"] = evaluate_method(v_est, i_est, ref_voice, ref_instr)

    # Affichage + accumulation globale
    for method, metrics in results.items():
        print(f"\n  --- {method} ---")
        for k, v in metrics.items():
            print(f"   {k} = {v:.2f} dB")
            global_results[method][k].append(v)


# ==============================
#  GRAPHIQUES GLOBAUX
# ==============================

fig_dir = os.path.join(OUT_DIR, "figures")
os.makedirs(fig_dir, exist_ok=True)

# On ne garde que les méthodes qui ont au moins un score
methods_present = [
    m for m in METHODS
    if len(global_results[m]["SDR_voice"]) > 0
]

# Barplots des moyennes
for metric in METRICS:
    values_mean = [
        np.mean(global_results[m][metric])
        for m in methods_present
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(methods_present, values_mean)
    plt.ylabel(metric + " (dB)")
    plt.title(f"{metric} moyen par méthode")
    plt.tight_layout()
    out_path = os.path.join(fig_dir, f"bar_{metric}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"  -> Graphique sauvegardé : {out_path}")

# Boxplot SDR voix
plt.figure(figsize=(6, 4))
data_sdr_voice = [global_results[m]["SDR_voice"] for m in methods_present]
plt.boxplot(data_sdr_voice, labels=methods_present)
plt.ylabel("SDR_voice (dB)")
plt.title("Distribution du SDR voix par méthode")
plt.tight_layout()
out_path = os.path.join(fig_dir, "box_SDR_voice.png")
plt.savefig(out_path)
plt.close()
print(f"  -> Graphique sauvegardé : {out_path}")
