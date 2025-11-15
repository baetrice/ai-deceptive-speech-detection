
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("../data/features/features_normalized.csv")  


feature_cols = [col for col in df.columns if any(
    col.startswith(prefix) for prefix in ["mfcc", "delta", "deltadelta", "f0"]
)]
X = df[feature_cols].values
y = df["label"].values  


best_rf = RandomForestClassifier(
    n_estimators=40,
    max_depth=50,
    criterion="gini",
    random_state=42
)
best_rf.fit(X, y)


importances = best_rf.feature_importances_
feature_ranking = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(
    feature_ranking["Feature"][:10], 
    feature_ranking["Importance"][:10],
    color='skyblue'
)
plt.xlabel("Importanță")
plt.title("Top 10 cele mai importante trăsături (Random Forest)")
plt.gca().invert_yaxis()  
plt.tight_layout()

plt.savefig("../results/rf_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()