import os
import numpy as np
import librosa

# Setări directoare
INPUT_DIR = "../data/audio_clips"
OUTPUT_DIR = "../data/spectrograms_200"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parametri pentru cadrare și spectrogramă
SAMPLE_RATE = 16000
FRAME_SIZE = 0.025  # 25ms
HOP_SIZE = 0.01     # 10ms
N_FFT = 512

# Dimensiune fixă finală a spectrogramelor (standardizăm pe timp)
MAX_FRAMES = 200  

def extract_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Zero-padding ca toate să aibă aceeași lungime (maxim 3 secunde)
    max_len = int(SAMPLE_RATE * (HOP_SIZE * (MAX_FRAMES - 1) + FRAME_SIZE))
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    # Calcul STFT și conversie la amplitudine logaritmică [dB]
    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=int(HOP_SIZE * sr),
        win_length=int(FRAME_SIZE * sr),
        window="hamming"
    )
    spectrogram = np.abs(stft)[:257, :]  # păstrăm doar frecvențele 0–8kHz (primii 257 coeficienți)
     # Corectare: crop/pad pentru a avea exact MAX_FRAMES coloane
    if spectrogram.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    else:
        spectrogram = spectrogram[:, :MAX_FRAMES]
        
    log_spectrogram = 10 * np.log10(spectrogram + 1e-10)  # evităm log(0)

    # Normalizare:  pentru rețea mai stabilă
    log_spectrogram = (log_spectrogram - np.mean(log_spectrogram)) / np.std(log_spectrogram)

    return log_spectrogram

# Parcurgem toate fișierele audio
num_saved = 0
for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".wav"):
        continue

    input_path = os.path.join(INPUT_DIR, fname)
    output_name = fname.replace(".wav", ".npy")
    output_path = os.path.join(OUTPUT_DIR, output_name)

    spec = extract_spectrogram(input_path)
    np.save(output_path, spec)
    num_saved += 1

print(f"\n Spectrograme extrase și salvate: {num_saved}")
print(f"Salvate în: {OUTPUT_DIR}")
