import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_df = pd.read_csv("../results/svm_results.csv")

results_df["label"] = results_df["config"].apply(
    lambda x: f"{eval(x)['kernel']} (C={eval(x)['C']})"
)

x = np.arange(len(results_df))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, results_df["accuracy"], width, label='Acuratete', color='skyblue')
bars2 = ax.bar(x + width/2, results_df["f1_score"], width, label='Scor F1', color='coral')

ax.set_xlabel("Configurații SVM (kernel și C)")
ax.set_ylabel("Scor")
ax.set_title("Performanța modelelor SVM ")
ax.set_xticks(x)
ax.set_xticklabels(results_df["label"], rotation=45, ha='right')
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()
