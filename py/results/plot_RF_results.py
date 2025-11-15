import pandas as pd
import matplotlib.pyplot as plt
import ast


df = pd.read_csv("../results/rf_results.csv")


df["config"] = df["config"].apply(ast.literal_eval)


df["config_str"] = df["config"].apply(
    lambda c: f'{c["n_estimators"]}-d={c["max_depth"]}-{c["criterion"]}'
)


df_sorted = df.sort_values(by="accuracy", ascending=False)


df_sorted["accuracy_percent"] = df_sorted["accuracy"] * 100


plt.figure(figsize=(12, 6))
bars = plt.bar(df_sorted["config_str"], df_sorted["accuracy_percent"], color="pink")


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.2f}%", ha="center", va='bottom', fontsize=8)

plt.title("Performanța RF")
plt.ylabel("Acuratețe (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
