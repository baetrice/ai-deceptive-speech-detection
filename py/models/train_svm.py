import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
import os
import pickle

def load_fixed_folds():
    df = pd.read_csv("../data/features/features_normalized.csv")
    df["speaker_id"] = df["file"].str.extract(r"(speaker\d+)")
    df["label"] = df["label"].astype(int)

    with open("../data/folds_indices.pkl", "rb") as f:
        fold_indices = pickle.load(f)

    df["fold"] = -1
    for i, (_, val_idx) in enumerate(fold_indices):
        df.loc[val_idx, "fold"] = i

    return df
def evaluate_model(X, y, fold_speakers, df, configs):
    results = []
    for cfg in configs:
        fold_metrics = []
        for fold_id, speaker_ids in enumerate(fold_speakers):

            train_idx = df[~df['speaker_id'].isin(speaker_ids)].index
            test_idx = df[df['speaker_id'].isin(speaker_ids)].index

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = make_pipeline(
                RobustScaler(),
                RandomUnderSampler(random_state=42),
                SVC(
                    C=cfg['C'],
                    kernel=cfg['kernel'],
                    gamma='scale',
                    class_weight='balanced',
                    random_state=42
                )
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, pos_label=1)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            fold_metrics.append({
                'accuracy': acc,
                'f1': f1,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            })

        results.append({
            'config': cfg,
            'mean_accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
            'mean_f1': np.mean([m['f1'] for m in fold_metrics])
        })

    return results

def main():
    print("Încarc datele și foldurile...")
    df = load_fixed_folds()

    fold_speakers = []
    for i in range(5):
        speakers = df[df["fold"] == i]["speaker_id"].unique().tolist()
        fold_speakers.append(speakers)

    X = df.drop(columns=["file", "label", "speaker_id", "fold"]).values
    y = df["label"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    configs = [
        {'C': 0.1, 'kernel': 'rbf'},
        {'C': 1.0, 'kernel': 'rbf'},
        {'C': 10.0, 'kernel': 'rbf'},
        {'C': 0.1, 'kernel': 'linear'},
        {'C': 1.0, 'kernel': 'linear'},
        {'C': 10.0, 'kernel': 'linear'},
        {'C': 1.0, 'kernel': 'sigmoid'},
        {'C': 10.0, 'kernel': 'sigmoid'},
        {'C': 0.1, 'kernel': 'sigmoid'},  # nou
        {'C': 5.0, 'kernel': 'rbf'},
    ]

    print("\nEvaluare modele SVM...")
    results = evaluate_model(X, y, fold_speakers, df, configs)

    os.makedirs("../results", exist_ok=True)
    results_df = pd.DataFrame([{
        'config': str(r['config']),
        'accuracy': r['mean_accuracy'],
        'f1_score': r['mean_f1']
    } for r in results])

    results_df.to_csv("../results/svm_results.csv", index=False)
    print("\nRezultate salvate în results/svm_results.csv")


if __name__ == "__main__":
    main()