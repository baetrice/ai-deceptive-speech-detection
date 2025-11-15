import pandas as pd
import matplotlib.pyplot as plt
import os


base_path = "../results"

files = {
    "SVM": os.path.join(base_path, "svm_results.csv"),
    "RF": os.path.join(base_path, "rf_results.csv"),
    "FCNN": os.path.join(base_path, "fcnn_results.csv"),
    "CNN_mel": os.path.join(base_path, "cnn_results_mel.csv"),
    "CNN_lin": os.path.join(base_path, "cnn_results_lin.csv")
}

best_configs = []

for model_name, file_path in files.items():
    df = pd.read_csv(file_path)

    # Normalizează denumirea coloanei de acuratețe
    if "accuracy" in df.columns:
        acc_col = "accuracy"
    elif "mean_accuracy" in df.columns:
        acc_col = "mean_accuracy"
    else:
        raise ValueError(f"Coloana de acuratețe nu există în {file_path}")

    best_row = df.loc[df[acc_col].idxmax()].copy()
    best_configs.append({
        "Model": model_name,
        "Best Config": best_row["config"] if "config" in best_row else "default",
        "Accuracy": round(best_row[acc_col], 4),
        "F1 Score": round(best_row["f1_score"], 4) if "f1_score" in best_row else "N/A"
    })

summary_df = pd.DataFrame(best_configs)
summary_df.to_csv(os.path.join(base_path, "best_models_summary.csv"), index=False)
print("✅ best_models_summary.csv a fost salvat.")


plt.figure(figsize=(8, 6))
accuracies_pct = summary_df["Accuracy"] * 100
bars = plt.bar(summary_df["Model"], accuracies_pct, color="steelblue")

plt.title("Cea mai bună acuratețe (%) per model")
plt.ylabel("Acuratețe (%)")
plt.ylim(0, 100)


for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.2f}%", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(base_path, "best_model_plot.png"))
plt.show()
