import os
import inspect
import glob
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
import numpy.linalg as LA


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

#Masque 3 - masque hybride
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

#Affichage graphique
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

##Méthodes plus efficaces
def nmf_component_voice_mask(W, sr, n_fft, f_low=200, f_high=4000, ratio_thresh=0.6):
    """
    Détermine quelles composantes NMF sont plutôt 'voix', en
    regardant la part d'énergie dans la bande [f_low, f_high].

    W : (freq, n_components)
    Retourne : indices_voix (liste d'entiers)
    """
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= f_low) & (freqs <= f_high)

    voice_like = []
    for k in range(W.shape[1]):
        w_k = W[:, k]
        total = np.sum(w_k) + 1e-8
        in_band = np.sum(w_k[band])
        ratio = in_band / total
        if ratio >= ratio_thresh:
            voice_like.append(k)

    return voice_like

def separate_nmf(mix, sr, n_components=8):
    """
    Sépare le mix via NMF sur le module du spectrogramme.
    - n_components : nombre de composantes NMF
    """
    S_mix, mag_mix, phase_mix = stft(mix)  # mag_mix : (freq, time)

    V = mag_mix  # alias

    # NMF sur V (non-négatif)
    model = NMF(n_components=n_components, init='random',
                random_state=0, max_iter=500)
    W = model.fit_transform(V)      # (freq, n_components)
    H = model.components_           # (n_components, time)

    # Choix des composantes 'voix'
    voice_ids = nmf_component_voice_mask(W, sr, N_FFT)
    if len(voice_ids) == 0:
        # fallback : si rien trouvé, on prend la composante la plus énergétique
        energies = W.sum(axis=0)
        voice_ids = [int(np.argmax(energies))]

    # Reconstruction V_voice, V_instr
    V_voice = np.zeros_like(V)
    for k in voice_ids:
        V_voice += np.outer(W[:, k], H[k, :])

    V_instr = np.clip(V - V_voice, 0, None)

    # Masques soft de type Wiener
    eps = 1e-8
    denom = V_voice + V_instr + eps
    M_voice = V_voice / denom
    M_instr = V_instr / denom

    # Application aux STFT complexes
    S_voice = M_voice * S_mix
    S_instr = M_instr * S_mix

    y_voice = normalize(istft(S_voice))
    y_instr = normalize(istft(S_instr))

    return y_voice, y_instr, M_voice

def rpca(M, lam=None, mu=None, max_iter=100, tol=1e-7):
    """
    RPCA via Inexact Augmented Lagrange Multiplier (Candes et al.)
    M : matrice (freq x time) non-négative (mag spectrogram)
    Retourne : L (low-rank), S (sparse)
    """
    M = M.astype(float)
    m, n = M.shape

    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    norm_M = LA.norm(M, ord='fro')

    # initialisation
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = M / max(LA.norm(M, 2), LA.norm(M, np.inf) / lam)

    if mu is None:
        mu = 1.25 / LA.norm(M, 2)  # estimation
    mu_bar = mu * 1e7
    rho = 1.5

    for it in range(max_iter):
        # 1) SVD sur (M - S + (1/mu)Y)
        U, sigma, Vt = LA.svd(M - S + (1.0 / mu) * Y, full_matrices=False)
        # seuillage des valeurs singulières
        sigma_thresh = np.maximum(sigma - 1.0 / mu, 0)
        rank = np.sum(sigma_thresh > 0)
        L = (U[:, :rank] * sigma_thresh[:rank]) @ Vt[:rank, :]

        # 2) seuillage L1 pour S
        residual = M - L + (1.0 / mu) * Y
        S = np.sign(residual) * np.maximum(np.abs(residual) - lam / mu, 0)

        # 3) mise à jour Y, mu
        Z = M - L - S
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        err = LA.norm(Z, 'fro') / (norm_M + 1e-8)
        if err < tol:
            break
    return L, S

def separate_rpca(mix, sr):
    """
    Sépare le mix en utilisant RPCA sur le module du spectrogramme.
    Low-rank = instru, Sparse = voix.
    """
    S_mix, mag_mix, phase_mix = stft(mix)

    # RPCA sur la magnitude
    L, S = rpca(mag_mix)

    V_instr = np.clip(L, 0, None)
    V_voice = np.clip(S, 0, None)

    eps = 1e-8
    denom = V_voice + V_instr + eps
    M_voice = V_voice / denom
    M_instr = V_instr / denom

    S_voice = M_voice * S_mix
    S_instr = M_instr * S_mix

    y_voice = normalize(istft(S_voice))
    y_instr = normalize(istft(S_instr))

    return y_voice, y_instr, M_voice

##Comparaison avec modèle de ML
#pip install demucs pour installer le modèle
import torchaudio
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model

def separate_demucs(wav_path, out_subdir, sr_target=44100):
    """
    Sépare un mix en (voice_demucs, instr_demucs) avec Demucs
    en utilisant librosa pour le chargement (pas torchaudio).
    """

    print("  -> Demucs en cours...")

    # 1) Charger le fichier audio
    y, sr = librosa.load(wav_path, sr=sr_target, mono=False)

    # Convertir en stéréo si mono
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)   # (2, T)

    # (channels, time) -> Tensor (1, channels, time)
    mix = torch.from_numpy(y).float().unsqueeze(0)  # (1, 2, T)

    # 2) Charger le modèle Demucs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model("htdemucs").to(device)
    model.eval()

    mix = mix.to(device)

    # 3) Appliquer le modèle
    with torch.no_grad():
        estimates = apply_model(model, mix, split=True, overlap=0.25)[0]

    # Ordre : [drums, bass, other, vocals]
    drums, bass, other, vocals = estimates

    # 4) Accompagnement = drums + bass + other
    accomp = drums + bass + other

    # 5) Convertir en numpy (T, C)
    vocals_np = vocals.cpu().numpy().transpose(1, 0)
    accomp_np = accomp.cpu().numpy().transpose(1, 0)

    # 6) Sauvegarde
    sf.write(os.path.join(out_subdir, "voice_demucs.wav"), vocals_np, sr_target)
    sf.write(os.path.join(out_subdir, "instr_demucs.wav"), accomp_np, sr_target)

    print("  -> Fichiers Demucs écrits.")

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
script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
script_dir  = os.path.dirname(script_path)

BASE_DIR = os.path.abspath(os.path.join(script_dir, ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "Mixes")
OUT_DIR  = os.path.join(BASE_DIR, "results")

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

    # --------------------------------------------------------
    # 5) DEMUCS (modèle ML SOTA)
    # --------------------------------------------------------
    try:
        separate_demucs(wav_path, out_subdir)
    except Exception as e:
        print(f"  !! ERREUR DEMUCS : {e}")

    # --------------------------------------------------------
    # 6) NMF
    # --------------------------------------------------------
    try:
        y_v_nmf, y_i_nmf, M_voice_nmf = separate_nmf(mix, sr, n_components=8)
        sf.write(os.path.join(out_subdir, "voice_nmf.wav"), y_v_nmf, sr)
        sf.write(os.path.join(out_subdir, "instr_nmf.wav"), y_i_nmf, sr)
        print("  -> fichiers NMF écrits")

        save_mask_frequency_plot(
            M_voice_nmf,
            sr,
            f"Masque NMF - {base}",
            os.path.join(out_subdir, "mask_nmf_freqs.png")
        )
    except Exception as e:
        print(f"  !! ERREUR NMF : {e}")

    # --------------------------------------------------------
    # 7) RPCA
    # --------------------------------------------------------
    try:
        y_v_rpca, y_i_rpca, M_voice_rpca = separate_rpca(mix, sr)
        sf.write(os.path.join(out_subdir, "voice_rpca.wav"), y_v_rpca, sr)
        sf.write(os.path.join(out_subdir, "instr_rpca.wav"), y_i_rpca, sr)
        print("  -> fichiers RPCA écrits")

        save_mask_frequency_plot(
            M_voice_rpca,
            sr,
            f"Masque RPCA - {base}",
            os.path.join(out_subdir, "mask_rpca_freqs.png")
        )
    except Exception as e:
        print(f"  !! ERREUR RPCA : {e}")

    # --------------------------------------------------------
    # 8) SAUVEGARDER LES PLOTS
    # --------------------------------------------------------
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
