
import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization,
    LeakyReLU, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score

# === CONFIG ===
SPECTRO_DIR    = "../data/spectrograms_200"
ANNOTATION_DIR = "../data/datasetAnnotation"
FOLDS_PICKLE   = "../data/folds_indices-fete.pkl"


NUM_EPOCHS     = 15
BATCH_SIZE     = 8
SEED           = 42
CSV_OUT        = "cnn_results_full.csv"

# === LOADING FUNCTIONS ===
def load_speaker_gender_map(annotation_dir):
    m = {}
    for fn in os.listdir(annotation_dir):
        if fn.endswith(".csv"):
            df = pd.read_csv(os.path.join(annotation_dir, fn))
            for _,r in df.iterrows():
                sp = str(r["Speaker"]).strip()
                g  = r["Gender"].strip().upper()
                if sp.upper()!="TM":
                    m[int(sp)] = g
    return m

def load_data():
    X,y,spk,gen = [],[],[],[]
    gm = load_speaker_gender_map(ANNOTATION_DIR)
    files = sorted(f for f in os.listdir(SPECTRO_DIR) if f.endswith(".npy"))
    for f in files:
        lbl = 1 if "lie" in f else 0
        s   = int([p.replace("speaker","") for p in f.split("_") if p.startswith("speaker")][0])
        if s not in gm: continue
        spec = np.load(os.path.join(SPECTRO_DIR,f))
        X.append(spec[...,np.newaxis]); y.append(lbl)
        spk.append(s); gen.append(gm[s])
    df_meta = pd.DataFrame({
        "speaker_id": spk,
        "gender":     gen,
        "label":      y,
        "index":      np.arange(len(y))
    })
    return np.array(X), np.array(y), df_meta

# === CNN MODEL FACTORY ===
def create_cnn_model(input_shape, cfg):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(cfg["filters1"], 3, padding="same"))
    if cfg["conv_activation"] == "leaky_relu":
        model.add(LeakyReLU(0.1))
    else:
        model.add(Activation(cfg["conv_activation"]))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(cfg["filters2"], 3, padding="same", activation=cfg["conv_activation"]))
    model.add(MaxPooling2D(2))
    if cfg["filters3"]:
        model.add(Conv2D(cfg["filters3"], 3, padding="same", activation=cfg["conv_activation"]))
        model.add(MaxPooling2D(2))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(cfg["dense1"], activation=cfg["dense_activation"]))
    if cfg["dense2"]:
        model.add(Dense(cfg["dense2"], activation=cfg["dense_activation"]))
    model.add(Dropout(cfg["dropout"]))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(cfg["learning_rate"]),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# === CONFIGURATIONS ===
def build_all_configs():
    configs = []

    # Bloc 1 – 30 configuri de bază
    conv_acts  = ["relu", "tanh", "leaky_relu"]
    dense_acts = ["relu", "tanh"]
    for f1 in [8,16]:
        for f2 in [16,32]:
            for f3 in [0,32]:
                if f3 and f3 <= f2: continue
                for d1 in [64,128]:
                    for d2 in [0,64]:
                        for dr in [0.3,0.4]:
                            for ca in conv_acts:
                                for da in dense_acts:
                                    if ca=="tanh" and da=="tanh": continue
                                    configs.append({
                                        "filters1":f1, "filters2":f2, "filters3":f3,
                                        "dense1":d1, "dense2":d2, "dropout":dr,
                                        "conv_activation":ca,
                                        "dense_activation":da,
                                        "learning_rate":1e-3
                                    })

    # Bloc 2 – 15 configuri extinse cu activări noi + learning_rate variabil
    conv_acts      = ["relu", "tanh", "leaky_relu"]
    dense_acts_ext = ["relu", "tanh", "selu"]
    learning_rates = [5e-4, 2e-4]
    for f1 in [8,16,32]:
        for f2 in [16,32,64]:
            if f1 >= f2: continue
            for f3 in [0,32,64]:
                if f3 and f3 <= f2: continue
                for d1 in [64,128]:
                    for d2 in [0,64]:
                        for dr in [0.3,0.4]:
                            for ca in conv_acts:
                                for da in dense_acts_ext:
                                    for lr in learning_rates:
                                        if ca=="tanh" and da=="tanh": continue
                                        configs.append({
                                            "filters1":f1, "filters2":f2, "filters3":f3,
                                            "dense1":d1, "dense2":d2, "dropout":dr,
                                            "conv_activation":ca,
                                            "dense_activation":da,
                                            "learning_rate":lr
                                        })
    configs = configs[:45]  # 30 + 15

    # Bloc 3 – 10 configuri cu filtre variabile, restul fix
    base_cfg = {
        "dense1":64, "dense2":0,
        "dropout":0.3,
        "conv_activation":"relu",
        "dense_activation":"relu",
        "learning_rate":1e-3
    }
    filters1 = [8, 16, 32]
    filters2 = [16, 32, 64]
    filters3 = [0, 32, 64]
    cnt = 0
    for f1 in filters1:
        for f2 in filters2:
            if f2 <= f1: continue
            configs.append({**base_cfg, "filters1":f1, "filters2":f2, "filters3":0})
            cnt += 1
            for f3 in filters3[1:]:
                if f3 <= f2: continue
                configs.append({**base_cfg, "filters1":f1, "filters2":f2, "filters3":f3})
                cnt += 1
                if cnt >= 10:
                    break
            if cnt >= 10:
                break
        if cnt >= 10:
            break
    return configs

# === MAIN TRAINING LOOP ===
def train_with_cv():
    X, y, df_meta = load_data()
    with open(FOLDS_PICKLE, "rb") as f:
        folds = pickle.load(f)

    configs = build_all_configs()
    print(f"Total configs to run: {len(configs)}")

    records = []
    np.random.seed(SEED)
    for i, cfg in enumerate(configs, 1):
        accs, f1s = [], []
        for train_idx, val_idx in folds:
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            model = create_cnn_model(X.shape[1:], cfg)
            es = EarlyStopping('val_loss', patience=3, restore_best_weights=True)
            model.fit(X_tr, y_tr,
                      validation_data=(X_va, y_va),
                      epochs=NUM_EPOCHS,
                      batch_size=BATCH_SIZE,
                      callbacks=[es],
                      verbose=0)
            y_pred = (model.predict(X_va, verbose=0).ravel() >= 0.5).astype(int)
            accs.append((y_pred == y_va).mean())
            f1s.append(f1_score(y_va, y_pred))
        rec = {**cfg,
               "avg_accuracy": np.mean(accs),
               "avg_f1":       np.mean(f1s)}
        records.append(rec)
        print(f"Config {i}/{len(configs)}: Acc={rec['avg_accuracy']:.3f}, F1={rec['avg_f1']:.3f}")

    pd.DataFrame(records).to_csv(CSV_OUT, index=False, sep=";")
    print(f"\n All results saved to: {CSV_OUT}")

if __name__ == "__main__":
    train_with_cv()
