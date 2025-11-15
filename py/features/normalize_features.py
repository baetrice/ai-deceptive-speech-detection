import pandas as pd
import os

INPUT_CSV = "../data/features/features.csv"
OUTPUT_CSV = "../data/features/features_normalized.csv"

df = pd.read_csv(INPUT_CSV)

if "speaker_id" not in df.columns:
    df["speaker_id"] = df["file"].str.extract(r"(speaker\d+)")

meta_cols = ["file", "label", "speaker_id"]
feature_cols = [col for col in df.columns if col not in meta_cols]

df_norm = df.copy()
for speaker in df["speaker_id"].unique():
    speaker_idx = df["speaker_id"] == speaker
    speaker_data = df.loc[speaker_idx, feature_cols]
    mean = speaker_data.mean()
    std = speaker_data.std().replace(0, 1)
    df_norm.loc[speaker_idx, feature_cols] = (speaker_data - mean) / std

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_norm.to_csv(OUTPUT_CSV, index=False)
print(f"Fișier normalizat salvat în: {OUTPUT_CSV}")