import os
import glob
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

## Paramètres globaux
SR = 22050 #Sample Rate (fréquence d'échantillonage)
N_FFT = 2048 #Nombre de découpages du son/Nombre de TFT par STFT
HOP = 512 #Pas temporel dans la STFT
WINDOW = "hann" #C'est une fonction porte douce

##Fonctions élémentaires:
#Chargement d'un fichier .wav
def load_audio(path, sr=SR):
    """
    Charge un fichier audio WAV.
    - Si le fichier est stéréo, le convertit automatiquement en mono
      (moyenne des deux canaux).
    - Si déjà mono, renvoie tel quel.
    """
    # Charge sans downmix pour détecter stéréo
    y, sr_loaded = librosa.load(path, sr=sr, mono=False)

    # Si stéréo → downmix manuel
    if y.ndim == 2:
        # y.shape = (n_channels, n_samples)
        print(f"  -> fichier stéréo détecté ({y.shape[0]} canaux), conversion en mono...")
        y = np.mean(y, axis=0)

    # Normalisation douce (évite clipping)
    y = y / (np.max(np.abs(y)) + 1e-8)

    return y, sr_loaded

#STFT - transformée de fourier locale/à court terme: domaine temporel -> domaine fréquentiel
def stft(y):
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window=WINDOW)
    mag, phase = librosa.magphase(S)
    return S, mag, phase

#STFT inverse: domaine fréquentielle -> domaine temporel
def istft(S):
    return librosa.istft(S, hop_length=HOP, window=WINDOW)

#Fonction de normalisation:
def normalize(y):
    m = np.max(np.abs(y)) + 1e-8
    return y / m

##Masques
#Masque 1 (le plus simple) - filtre par bandes de fréquences/"fonction porte"
def band_masks(sr, n_fft, f_low=80, f_high=4000):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    voice_band = (freqs >= f_low) & (freqs <= f_high)

    M_voice = voice_band[:, None].astype(float)  # (freqs, 1) → (freqs, time) par broadcast
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

#Masque 2 - HPSS (harmonique et percursif)
def separate_hpss(mix):
    S_mix, mag_mix, phase_mix = stft(mix)
    H, P = librosa.decompose.hpss(S_mix)

    y_h = normalize(istft(H))  # partie harmonique
    y_p = normalize(istft(P))  # partie percussive
    return y_h, y_p

# Masque 3 - Méthode par variabilité temporelle (temps-fréquence)
def variability_mask(mag_mix, alpha=2.0):
    """
    mag_mix : (freq, time)
    alpha   : contraste (plus grand = voix plus marquée)
    """
    # dérivée temporelle locale (variation dans le temps)
    diff_t = np.abs(np.diff(mag_mix, axis=1))  # (freq, time-1)

    # on remet à la même taille
    var_local = np.pad(diff_t, ((0, 0), (0, 1)), mode='edge')

    # normalisation 0..1
    var_norm = var_local / (np.max(var_local) + 1e-8)

    # masque soft : fréquences + instants très variables = plutôt voix
    M_voice = var_norm ** alpha        # 0..1
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

def soft_band_mask(sr, n_fft, f_low=80, f_high=4000, width=200):
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

    # 2) HPSS
    H, P = librosa.decompose.hpss(S_mix)
    mag_H = np.abs(H)
    mag_P = np.abs(P)
    M_hpss = mag_H / (mag_H + mag_P + 1e-8)  # voix ≈ partie harmonique

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
    M_voice : masque voix (freq x temps), valeurs 0..1
    On le réduit en 1D par fréquence, puis on trace
    en bleu les fréquences instrus, en rouge les voix.
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # On réduit le masque sur le temps : moyenne par fréquence
    voice_strength = M_voice.mean(axis=1)  # (n_freqs,)
    voice_bins = voice_strength > 0.5      # True = plutôt voix, False = plutôt instru

    plt.figure(figsize=(7, 3))
    # Instru en bleu (y=0)
    plt.scatter(freqs[~voice_bins],
                np.zeros(np.sum(~voice_bins)),
                c="b", s=10, label="Instru")
    # Voix en rouge (y=1)
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

##Comparaison avec modèle de ML
#Installer Spleeter ou Demucs (pas du python)
#Terminal -> spleeter separate -i data/track1.wav -p spleeter:2stems -o results_spleeter
#Création de : results_spleeter/track1/
#    vocals.wav
#    accompaniment.wav
#Chargement des fichiers:
#vocals, sr_v = librosa.load("results_spleeter/track1/vocals.wav", sr=SR, mono=True)
#accomp, sr_a = librosa.load("results_spleeter/track1/accompaniment.wav", sr=SR, mono=True)

##Visualisation:
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

    # Méthode bande
    y_voice_band, y_instr_band = separate_band(mix, sr)
    S_voice_band, _, _ = stft(y_voice_band)

    # Affichages
    plot_spectrogram(S_mix, sr, "Spectrogramme mélange")
    plot_spectrogram(S_voice_band, sr, "Spectrogramme voix (bande)")

##Boucle sur tous les fichiers du dataset:
DATA_DIR = r"C:\Users\emine\Desktop\projet_separation\data\Mixes"
OUT_DIR = r"C:\Users\emine\Desktop\projet_separation\results"

print("Répertoire courant :", os.getcwd())
print("DATA_DIR =", DATA_DIR)
print("OUT_DIR  =", OUT_DIR)

os.makedirs(OUT_DIR, exist_ok=True)

wav_paths = glob.glob(os.path.join(DATA_DIR, "*.wav"))
print("Fichiers .wav trouvés :", wav_paths)

for wav_path in wav_paths:
    print("\n=== Traitement de :", wav_path, "===")
    base = os.path.splitext(os.path.basename(wav_path))[0]
    out_subdir = os.path.join(OUT_DIR, base)
    os.makedirs(out_subdir, exist_ok=True)

    mix, sr = load_audio(wav_path)
    print("  - sr =", sr, " | longueur =", len(mix))

    # --- Calcul STFT une seule fois ---
    S_mix, mag_mix, phase_mix = stft(mix)

    # --------------------------------------------------------
    # 1) FILTRE PAR BANDE
    # --------------------------------------------------------
    y_v_band, y_i_band = separate_band(mix, sr)
    sf.write(os.path.join(out_subdir, "voice_band.wav"), y_v_band, sr)
    sf.write(os.path.join(out_subdir, "instr_band.wav"), y_i_band, sr)
    print("  -> fichiers bande écrits")

    # masque en fréquence = simple bande 80–4000 Hz
    M_voice_band, M_instr_band = band_masks(sr, N_FFT)
    save_mask_frequency_plot(
        M_voice_band,
        sr,
        f"Masque bande (80–4000 Hz) - {base}",
        os.path.join(out_subdir, "mask_band_freqs.png")
    )

    # --------------------------------------------------------
    # 2) HPSS (harmonique / percussif)
    # --------------------------------------------------------
    y_h, y_p = separate_hpss(mix)
    sf.write(os.path.join(out_subdir, "harmonic_hpss.wav"), y_h, sr)
    sf.write(os.path.join(out_subdir, "percussive_hpss.wav"), y_p, sr)
    print("  -> fichiers HPSS écrits")

    # On construit un masque voix ≈ partie harmonique (magnitude)
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

    # --------------------------------------------------------
    # 3) VARIABILITÉ TEMPORELLE
    # --------------------------------------------------------
    y_v_var, y_i_var = separate_variability(mix)
    sf.write(os.path.join(out_subdir, "voice_var.wav"), y_v_var, sr)
    sf.write(os.path.join(out_subdir, "instr_var.wav"), y_i_var, sr)
    print("  -> fichiers variabilité écrits")

    # masque variabilité

    M_voice_var, M_instr_var = variability_mask(mag_mix, alpha=2.0)

    # --------------------------------------------------------
    # 4) MASQUE HYBRIDE (bande douce + HPSS + variabilité)
    # --------------------------------------------------------
    y_v_hybrid, y_i_hybrid, M_voice_hybrid = separate_hybrid(mix, sr)

    sf.write(os.path.join(out_subdir, "voice_hybrid.wav"), y_v_hybrid, sr)
    sf.write(os.path.join(out_subdir, "instr_hybrid.wav"), y_i_hybrid, sr)
    print("  -> fichiers hybrides écrits")

    # Sauvegarde du masque hybride (en fréquence)
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

##Evaluation
#Vérification à faire ||mix||² ≈ ||sortie1||² + ||sortie2||²
