import os
import numpy as np
import pandas as pd
import pickle
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

def load_fixed_folds(features_path="../data/features/features_normalized.csv", folds_path="../data/folds_indices.pkl"):
    print(" Încarc datele...")
    df = pd.read_csv(features_path)
    df["speaker_id"] = df["file"].str.extract(r"(speaker\d+)")

    print(" Încarc foldurile din pickle...")
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)  #

    return df, folds


def build_fcnn_model(input_dim, config):
    model = Sequential()
    for i, units in enumerate(config["hidden_layers"]):
        if i == 0:
            model.add(Dense(units, activation=config["activation"],
                            kernel_regularizer=l2(config["l2"]), input_shape=(input_dim,)))
        else:
            model.add(Dense(units, activation=config["activation"],
                            kernel_regularizer=l2(config["l2"])))
        if config["batchnorm"]:
            model.add(BatchNormalization())
        if config["dropout"] > 0:
            model.add(Dropout(config["dropout"]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def generate_configs():
    activations = ['relu', 'tanh']
    layer_sizes = [
        [64],
        [128],
        [64, 32],
        [128, 64],
        [128, 64, 32]
    ]
    dropouts = [0.0, 0.2, 0.5]
    l2_vals = [0.0, 0.001]
    batchnorm_options = [True, False]

    configs = []
    for combo in product(layer_sizes, activations, dropouts, l2_vals, batchnorm_options):
        config = {
            "hidden_layers": combo[0],
            "activation": combo[1],
            "dropout": combo[2],
            "l2": combo[3],
            "batchnorm": combo[4]
        }
        configs.append(config)
    return configs[:100]


def main():
    df, folds = load_fixed_folds()

    X = df.drop(columns=["file", "label", "speaker_id"]).values
    y = df["label"].astype(int).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    configs = generate_configs()
    results = []

    print("\n Evaluare FCNN pe folduri...\n")
    for config_idx, config in enumerate(configs, 1):
        fold_accuracies = []
        fold_f1s = []

        for fold_id, (train_idx, val_idx) in enumerate(folds):
            if len(train_idx) == 0 or len(val_idx) == 0:
                print(f"⚠ Fold {fold_id+1} are seturi goale, ignorat.")
                continue

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = build_fcnn_model(X.shape[1], config)
            early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)

            model.fit(X_train, y_train, epochs=100, batch_size=32,
                      validation_data=(X_val, y_val),
                      callbacks=[early_stop], verbose=0)

            y_pred = model.predict(X_val).flatten()
            y_pred_binary = (y_pred >= 0.5).astype(int)

            acc = accuracy_score(y_val, y_pred_binary)
            f1 = f1_score(y_val, y_pred_binary, pos_label=1)

            fold_accuracies.append(acc)
            fold_f1s.append(f1)

        mean_acc = np.mean(fold_accuracies) if fold_accuracies else 0.0
        mean_f1 = np.mean(fold_f1s) if fold_f1s else 0.0

        results.append({
            "config": str(config),
            "accuracy": mean_acc,
            "f1_score": mean_f1
        })

        print(f"[{config_idx}/100] Acc: {mean_acc:.4f} | F1: {mean_f1:.4f} | Config: {config}")

    os.makedirs("../results", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)
    results_df.to_csv("../results/fcnn_results.csv", index=False)
    print("\nRezultatele au fost salvate în: results/fcnn_results.csv (sortate descrescător după F1)")


if __name__ == "__main__":
    main()
