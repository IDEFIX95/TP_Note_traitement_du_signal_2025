import os 
import numpy as np
from scipy.io import wavfile

def read_wav_file(file_path):
    sampling_rate, data = wavfile.read(file_path)
    return sampling_rate, data                      ##### fréquence et signal audio 

def to_mono_and_normalize(data):
    data = data.astype(np.float32)
    if data.ndim == 2:
        data= data.mean(axis=1)
    
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val           ####### normalisation de l'amplitude entre -1 et 1

    return data 

def load_and_prepare(file_path):
    fs, data = read_wav_file(file_path)
    data = to_mono_and_normalize(data)
    return fs, data                        ####### fréquence et signal mono normalisé


def mix_two_tracks(vocals, instru, vocals_gain=0.7, instru_gain=0.3):
    N = min(len(vocals), len(instru))           ##### choix de la plus petite longueur
    vocals = vocals[:N]
    instru = instru[:N]

    # mix = somme pondérée
    mix = vocals_gain * vocals + instru_gain * instru

    # renormalisation  
    max_val = np.max(np.abs(mix))
    if max_val > 0:
        mix = mix / max_val

    return mix


def generate_all_mixes(vocals_folder, instru_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    vocal_files = [f for f in os.listdir(vocals_folder) if f.lower().endswith(".wav")]
    instru_files = [f for f in os.listdir(instru_folder) if f.lower().endswith(".wav")]

    for vf in vocal_files:
        fs_v, vocal_data = load_and_prepare(os.path.join(vocals_folder, vf))

        for inf in instru_files:
            fs_i, instru_data = load_and_prepare(os.path.join(instru_folder, inf))

            if fs_v != fs_i:
                print(f"[WARN] Sampling rates differ for {vf} and {inf} — skipped")
                continue

            mix = mix_two_tracks(vocal_data, instru_data)

            out_name = f"mix_{vf[:-4]}_{inf[:-4]}.wav"
            out_path = os.path.join(output_folder, out_name)

            wavfile.write(out_path, fs_v, mix.astype(np.float32))
            print(f"✔ Saved {out_path}")


if __name__ == "__main__":
    base = "Dataset"

    vocals_folder = os.path.join(base, "Vocals")
    instru_folder = os.path.join(base, "Instrumentals")
    mixes_folder = os.path.join(base, "Mixes")

    generate_all_mixes(vocals_folder, instru_folder, mixes_folder)



