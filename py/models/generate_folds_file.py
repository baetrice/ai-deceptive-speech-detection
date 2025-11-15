import pandas as pd
import numpy as np
import os
import pickle

def get_speaker_gender_map(annotation_dir="../data/datasetAnnotation"):
    gender_map = {}
    for fname in os.listdir(annotation_dir):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(annotation_dir, fname))
        for _, row in df.iterrows():
            speaker = str(row["Speaker"]).strip()
            gender = str(row["Gender"]).strip()
            if speaker.upper() != "TM":
                gender_map[f"speaker{int(speaker)}"] = gender
    return gender_map

def print_detailed_distribution(df, folds):
    print("\n ANALIZĂ DETALIATĂ A FIECĂRUI FOLD\n")

    total_all = sum(len(train_idx) + len(val_idx) for train_idx, val_idx in folds)

    for i, (train_idx, val_idx) in enumerate(folds):
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]


        t_truth = (train_df["label"] == 0).sum()
        t_lie = (train_df["label"] == 1).sum()
        v_truth = (val_df["label"] == 0).sum()
        v_lie = (val_df["label"] == 1).sum()

        train_speakers = train_df[["speaker_id", "gender"]].drop_duplicates()
        val_speakers = val_df[["speaker_id", "gender"]].drop_duplicates()
        total_train = train_speakers["speaker_id"].nunique()
        total_val = val_speakers["speaker_id"].nunique()
        total_fold = total_train + total_val

        pct_train = total_train / total_fold * 100
        pct_val = total_val / total_fold * 100

        t_f = (train_speakers["gender"] == "F").sum()
        t_m = (train_speakers["gender"] == "M").sum()
        v_f = (val_speakers["gender"] == "F").sum()
        v_m = (val_speakers["gender"] == "M").sum()

        total_t_speakers = len(train_speakers)
        total_v_speakers = len(val_speakers)

        pct_t_f = t_f / total_t_speakers * 100 if total_t_speakers else 0
        pct_t_m = t_m / total_t_speakers * 100 if total_t_speakers else 0
        pct_v_f = v_f / total_v_speakers * 100 if total_v_speakers else 0
        pct_v_m = v_m / total_v_speakers * 100 if total_v_speakers else 0

        # Verificare independență
        overlap = set(train_speakers["speaker_id"]).intersection(set(val_speakers["speaker_id"]))
        independent = "Independent" if not overlap else f" ({len(overlap)} overlaps)"

        print(f"🔹 Fold {i+1}")
        print(f"Distribuție date: Train = {total_train} ({pct_train:.1f}%), Val = {total_val} ({pct_val:.1f}%), Total = {total_fold}")
        print(f"Independență speakeri: {independent}")

        # Gen
        print(f" Gen (train):  F = {t_f} ({pct_t_f:.1f}%), M = {t_m} ({pct_t_m:.1f}%), Total = {total_t_speakers}")
        print(f" Gen (valid):  F = {v_f} ({pct_v_f:.1f}%), M = {v_m} ({pct_v_m:.1f}%), Total = {total_v_speakers}")

        # Labeluri
        print(f" Antrenare: Truth = {t_truth} ({t_truth/(t_truth + t_lie)*100:.1f}%), "
              f"Lie = {t_lie} ({t_lie/(t_truth + t_lie)*100:.1f}%), Total = {t_truth + t_lie}")
        print(f" Validare:   Truth = {v_truth} ({v_truth/(v_truth + v_lie)*100:.1f}%), "
              f"Lie = {v_lie} ({v_lie/(v_truth + v_lie)*100:.1f}%), Total = {v_truth + v_lie}")
        print()


def create_balanced_folds(n_splits=5, seed=42):
    np.random.seed(seed)
    df = pd.read_csv("../data/features/features.csv")
    df["speaker_id"] = df["file"].str.extract(r"(speaker\d+)")
    df["label"] = df["label"].astype(int)

    gender_map = get_speaker_gender_map()
    df["gender"] = df["speaker_id"].map(gender_map)

    speakers = df.groupby("speaker_id").agg(
        size=("label", "count"),
        gender=("gender", "first"),
        truth_count=("label", lambda x: (x == 0).sum()),
        lie_count=("label", lambda x: (x == 1).sum())
    ).reset_index()

    speakers = speakers.sort_values(by="size", ascending=False)

    fold_speakers = [[] for _ in range(n_splits)]
    fold_stats = [{"size": 0, "f": 0, "m": 0, "truth": 0, "lie": 0} for _ in range(n_splits)]

    for _, row in speakers.iterrows():

        best_fold = min(range(n_splits), key=lambda i: (
            max(fold_stats[i]["f"], fold_stats[i]["m"]),
            fold_stats[i]["size"]
        ))
        fold_speakers[best_fold].append(row["speaker_id"])
        fold_stats[best_fold]["size"] += row["size"]
        fold_stats[best_fold]["truth"] += row["truth_count"]
        fold_stats[best_fold]["lie"] += row["lie_count"]
        if row["gender"] == "F":
            fold_stats[best_fold]["f"] += 1
        else:
            fold_stats[best_fold]["m"] += 1

    folds = []
    for i in range(n_splits):
        val_speakers = fold_speakers[i]
        val_idx = df[df["speaker_id"].isin(val_speakers)].index.to_numpy()
        train_idx = df.index.difference(val_idx).to_numpy()
        folds.append((train_idx, val_idx))
        print(f"Fold {i+1}: {len(val_speakers)} speakeri, {len(val_idx)} utterances")

    os.makedirs("../data", exist_ok=True)
    with open("../data/folds_indices.pkl", "wb") as f:
        pickle.dump(folds, f)

    print("Fold-urile echilibrate au fost salvate în ../data/folds_indices.pkl")

    print_detailed_distribution(df, folds)

if __name__ == "__main__":
    create_balanced_folds()