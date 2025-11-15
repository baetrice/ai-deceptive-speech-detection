import pandas as pd

def load_fixed_folds(df, folds_path="../data/folds_fixed.csv"):
    folds_df = pd.read_csv(folds_path)
    speaker_to_fold = dict(zip(folds_df["speaker_id"], folds_df["fold"]))

    df["speaker_id"] = df["file"].str.extract(r"(speaker\d+)")
    df["fold"] = df["speaker_id"].map(speaker_to_fold)

    if df["fold"].isnull().any():
        missing = df[df["fold"].isnull()]["speaker_id"].unique()
        raise ValueError(f" Unii speakeri nu au fold asignat: {missing}")

    return df