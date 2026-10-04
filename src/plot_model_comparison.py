import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/model_comparison.csv")

# Accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(df["Model"], df["Accuracy"] * 100)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("results/model_accuracy_comparison.png", dpi=300)
plt.close()

# F1 comparison
plt.figure(figsize=(10, 6))
plt.bar(df["Model"], df["F1"] * 100)
plt.ylabel("F1 Score (%)")
plt.title("Model F1 Score Comparison")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig("results/model_f1_comparison.png", dpi=300)
plt.close()

print("Comparison plots generated successfully.")
