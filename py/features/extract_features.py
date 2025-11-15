import os
import numpy as np
import pandas as pd
import librosa

# Setări directoare
CLIPS_DIR = "../data/audio_clips"
OUTPUT_CSV = "../data/features/features.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Parametrii de cadrare
FRAME_SIZE = 0.025  # 25 ms
HOP_SIZE = 0.01     # 10 ms
N_MFCC = 13

def extract_features_from_file(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    if len(y) == 0:
        return None

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=512,
        hop_length=int(HOP_SIZE * sr),
        win_length=int(FRAME_SIZE * sr),
        window='hamming'
    )

    # Δ și ΔΔ
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Pitch (F0)
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=500,
                            sr=sr, hop_length=int(HOP_SIZE * sr),
                            win_length=int(FRAME_SIZE * sr))
    f0 = np.nan_to_num(f0)

    # Stack final: [MFCC + delta + deltadelta + f0]
    all_features = np.vstack((mfcc, delta_mfcc, delta2_mfcc, f0[np.newaxis, :]))

    # Media și std pe fiecare dimensiune
    means = np.mean(all_features, axis=1)
    stds = np.std(all_features, axis=1)

    return np.concatenate((means, stds))


# Lista de rezultate
feature_list = []

for filename in os.listdir(CLIPS_DIR):
    if not filename.endswith(".wav"):
        continue

    file_path = os.path.join(CLIPS_DIR, filename)
    features = extract_features_from_file(file_path)

    if features is not None:
        label = 1 if "lie" in filename else 0  # 1 = înșelător, 0 = sincer
        entry = [filename, label] + features.tolist()
        feature_list.append(entry)

# Nume coloane
columns = ["file", "label"] + \
          [f"mfcc{i+1}_mean" for i in range(N_MFCC)] + \
          [f"delta{i+1}_mean" for i in range(N_MFCC)] + \
          [f"deltadelta{i+1}_mean" for i in range(N_MFCC)] + \
          ["f0_mean"] + \
          [f"mfcc{i+1}_std" for i in range(N_MFCC)] + \
          [f"delta{i+1}_std" for i in range(N_MFCC)] + \
          [f"deltadelta{i+1}_std" for i in range(N_MFCC)] + \
          ["f0_std"]


# Salvăm în CSV
df = pd.DataFrame(feature_list, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\n Trăsături extrase pentru {len(df)} rostiri. Salvate în: {OUTPUT_CSV}")