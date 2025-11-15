import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from generate_folds_file import get_speaker_gender_map

df = pd.read_csv("../data/features/features.csv")
df["speaker_id"] = df["file"].str.extract(r"(speaker\d+)")
gender_map = get_speaker_gender_map()
df["gender"] = df["speaker_id"].map(gender_map)
df["label"] = df["label"].astype(int)

with open("../data/folds_indices.pkl", "rb") as f:
    folds = pickle.load(f)

stats = []
total_speakers = df["speaker_id"].nunique()

for i, (train_idx, val_idx) in enumerate(folds):
    row = {"fold": i}
    for name, idx in [("train", train_idx), ("val", val_idx)]:
        subset = df.iloc[idx]
        row[f"{name}_utt"] = len(subset)
        row[f"{name}_truth"] = (subset["label"] == 0).sum()
        row[f"{name}_lie"] = (subset["label"] == 1).sum()
        unique_speakers = subset["speaker_id"].drop_duplicates()
        row[f"{name}_speakers"] = unique_speakers.nunique()
        genders = subset.drop_duplicates("speaker_id")["gender"].value_counts()
        row[f"{name}_F"] = genders.get("F", 0)
        row[f"{name}_M"] = genders.get("M", 0)
    row["total_speakers"] = total_speakers
    row["train_speaker_pct"] = row["train_speakers"] / total_speakers * 100
    row["val_speaker_pct"] = row["val_speakers"] / total_speakers * 100
    stats.append(row)

stats_df = pd.DataFrame(stats)

fig, axs = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle("Distribuție Vorbitori: Antrenare vs Validare per Fold", fontsize=18)

for i in range(5):
    train_pct = stats_df.loc[i, "train_speaker_pct"]
    val_pct = stats_df.loc[i, "val_speaker_pct"]
    axs[i].pie(
        [train_pct, val_pct],
        labels=["Train", "Val"],
        autopct="%1.1f%%",
        colors=["lightgreen", "salmon"],
        startangle=90
    )
    axs[i].set_title(f"Fold {i+1}")

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

fig, axs = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("Distribuții pe Folduri", fontsize=16)

sns.barplot(x="fold", y="train_utt", data=stats_df, ax=axs[0, 0])
axs[0, 0].set_title("Antrenare: Număr de rostiri")

sns.barplot(x="fold", y="val_utt", data=stats_df, ax=axs[0, 1])
axs[0, 1].set_title("Validare: Număr de rostiri")

axs[1, 0].bar(stats_df["fold"], stats_df["train_truth"], label="Adevar")
axs[1, 0].bar(stats_df["fold"], stats_df["train_lie"], bottom=stats_df["train_truth"], label="Minciuna")
axs[1, 0].set_title("Antrenare: Adevar vs Minciuna")
axs[1, 0].legend()

axs[1, 1].bar(stats_df["fold"], stats_df["val_truth"], label="Adevar")
axs[1, 1].bar(stats_df["fold"], stats_df["val_lie"], bottom=stats_df["val_truth"], label="Minciuna")
axs[1, 1].set_title("Validare: Adevar vs Minciuna")
axs[1, 1].legend()

axs[2, 0].bar(stats_df["fold"], stats_df["train_F"], label="Femei")
axs[2, 0].bar(stats_df["fold"], stats_df["train_M"], bottom=stats_df["train_F"], label="Bărbați")
axs[2, 0].set_title("Antrenare: Gen vorbitori")
axs[2, 0].legend()

axs[2, 1].bar(stats_df["fold"], stats_df["val_F"], label="Femei")
axs[2, 1].bar(stats_df["fold"], stats_df["val_M"], bottom=stats_df["val_F"], label="Bărbați")
axs[2, 1].set_title("Validare: Gen vorbitori")
axs[2, 1].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
fig3, axs3 = plt.subplots(5, 4, figsize=(18, 18))
fig3.suptitle(" Distribuții per Fold - Procente", fontsize=20)

for i in range(5):
    axs3[i, 0].pie(
        [stats_df.loc[i, 'train_truth'], stats_df.loc[i, 'train_lie']],
        labels=["Adevar", "Minciuna"],
        autopct='%1.1f%%',
        colors=["skyblue", "coral"]
    )
    axs3[i, 0].set_title(f" Fold Antrenare {i}: Adevar vs Minciuna")

    axs3[i, 1].pie(
        [stats_df.loc[i, 'val_truth'], stats_df.loc[i, 'val_lie']],
        labels=["Adevar", "Minciuna"],
        autopct='%1.1f%%',
        colors=["skyblue", "coral"]
    )
    axs3[i, 1].set_title(f" Fold Validare {i}: Adevar vs Minciuna")

    axs3[i, 2].pie(
        [stats_df.loc[i, 'train_F'], stats_df.loc[i, 'train_M']],
        labels=["Femei", "Bărbați"],
        autopct='%1.1f%%',
        colors=["orchid", "steelblue"]
    )
    axs3[i, 2].set_title(f"Fold Antrenare {i}: Gen vorbitori")

    axs3[i, 3].pie(
        [stats_df.loc[i, 'val_F'], stats_df.loc[i, 'val_M']],
        labels=["Femei", "Bărbați"],
        autopct='%1.1f%%',
        colors=["orchid", "steelblue"]
    )
    axs3[i, 3].set_title(f"Fold Validare {i}: Gen vorbitori")

plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.show()