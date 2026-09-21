import matplotlib.pyplot as plt
import numpy as np

models = ["Naive Bayes", "Random Forest", "XGBoost"]

accuracy  = [33.33, 88.89, 77.78]
precision = [33.33, 75.00, 66.67]
recall    = [100.00, 100.00, 66.67]
f1        = [50.00, 85.71, 66.67]

x = np.arange(len(models))
width = 0.18

plt.figure(figsize=(11, 6))

plt.bar(x - 1.5*width, accuracy,  width, label="Accuracy")
plt.bar(x - 0.5*width, precision, width, label="Precision")
plt.bar(x + 0.5*width, recall,    width, label="Recall")
plt.bar(x + 1.5*width, f1,        width, label="F1-Score")

plt.xlabel("Machine Learning Model")
plt.ylabel("Score (%)")
plt.title("Machine Learning Model Performance Comparison")

plt.xticks(x, models)
plt.ylim(0, 110)
plt.legend()
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    "Mealora_ML_Results/ml_model_comparison_ppt.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
