import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/features/features_normalized.csv")
df["speaker_id"] = df["file"].str.extract(r"(speaker\d+)")
df["label"] = df["label"].astype(int)


feature_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in ["mfcc", "delta", "deltadelta", "f0"])]
X = df[feature_cols].values
y = df["label"].values


scaler = StandardScaler()
X = scaler.fit_transform(X)


model = RandomForestClassifier(
    n_estimators=1,
    max_depth=3,  
    criterion="gini",
    random_state=42
)
model.fit(X, y)

plt.figure(figsize=(16, 10))
plot_tree(
    model.estimators_[0],
    feature_names=feature_cols,
    class_names=["truth", "lie"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Arbore RF simplificat (1 arbore, max_depth=3)")
plt.tight_layout()
plt.savefig("../results/rf_tree_simple.png", dpi=300)
plt.show()



