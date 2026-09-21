import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


# =========================================================
# PATHS
# =========================================================

DATA = "datasets/processed/supervised_training_dataset.csv"

RESULTS = "datasets/processed/cross_validation_results.csv"

RESULTS_TEXT = "datasets/processed/cross_validation_results.txt"

GRAPH_DIR = "datasets/processed/ml_graphs"

os.makedirs(GRAPH_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("MEALORA 5-FOLD CROSS-VALIDATION")
print("=" * 70)

df = pd.read_csv(DATA)

features = [
    "pantry_match_pct",
    "missing_ingredient_count",
    "expiry_score",
    "meal_match",
    "diet_match",
    "cuisine_match",
    "time_match",
    "context_score",
    "ingredient_count",
    "cooking_time_minutes",
]

X = df[features]
y = df["target"]


print("\nDataset samples:", len(df))
print("Features:", len(features))

print("\nTarget distribution:")
print(y.value_counts())


# =========================================================
# MODELS
# =========================================================

models = {

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GaussianNB())
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42
    )
}


# =========================================================
# 5-FOLD CROSS VALIDATION
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


results = []


print("\n" + "=" * 70)
print("RUNNING 5-FOLD CROSS-VALIDATION")
print("=" * 70)


for name, model in models.items():

    print("\nRunning:", name)

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    result = {
        "model": name,

        "accuracy_mean":
            scores["test_accuracy"].mean(),

        "accuracy_std":
            scores["test_accuracy"].std(),

        "precision_mean":
            scores["test_precision"].mean(),

        "precision_std":
            scores["test_precision"].std(),

        "recall_mean":
            scores["test_recall"].mean(),

        "recall_std":
            scores["test_recall"].std(),

        "f1_mean":
            scores["test_f1"].mean(),

        "f1_std":
            scores["test_f1"].std(),

        "roc_auc_mean":
            scores["test_roc_auc"].mean(),

        "roc_auc_std":
            scores["test_roc_auc"].std()
    }

    results.append(result)

    print(
        f"Accuracy : {result['accuracy_mean']:.4f} "
        f"+/- {result['accuracy_std']:.4f}"
    )

    print(
        f"Precision: {result['precision_mean']:.4f} "
        f"+/- {result['precision_std']:.4f}"
    )

    print(
        f"Recall   : {result['recall_mean']:.4f} "
        f"+/- {result['recall_std']:.4f}"
    )

    print(
        f"F1 Score : {result['f1_mean']:.4f} "
        f"+/- {result['f1_std']:.4f}"
    )

    print(
        f"ROC-AUC  : {result['roc_auc_mean']:.4f} "
        f"+/- {result['roc_auc_std']:.4f}"
    )


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "f1_mean",
    ascending=False
)


# =========================================================
# SAVE CSV
# =========================================================

results_df.to_csv(
    RESULTS,
    index=False
)


# =========================================================
# SAVE TEXT REPORT
# =========================================================

with open(RESULTS_TEXT, "w") as f:

    f.write("=" * 70 + "\n")
    f.write("MEALORA 5-FOLD CROSS-VALIDATION RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Dataset samples: {len(df)}\n")
    f.write(f"Number of folds: 5\n")
    f.write(f"Number of features: {len(features)}\n\n")

    f.write("FEATURES USED:\n")

    for feature in features:
        f.write(f"- {feature}\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write("MODEL COMPARISON\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        results_df.to_string(
            index=False
        )
    )

    f.write("\n\n" + "=" * 70 + "\n")

    best_model = results_df.iloc[0]["model"]

    f.write(
        f"BEST MODEL BASED ON MEAN F1 SCORE: {best_model}\n"
    )

    f.write("=" * 70 + "\n")


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nBest model based on mean F1:", best_model)


# =========================================================
# GRAPH — CROSS VALIDATION MODEL COMPARISON
# =========================================================

metrics = [
    "accuracy_mean",
    "precision_mean",
    "recall_mean",
    "f1_mean",
    "roc_auc_mean"
]

plot_df = results_df.set_index(
    "model"
)[metrics]

plot_df.columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]


ax = plot_df.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Mealora — 5-Fold Cross-Validation Model Comparison",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Mean Score"
)

plt.ylim(
    0,
    1.05
)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Evaluation Metric",
    loc="lower right"
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_DIR}/cross_validation_comparison.png",
    dpi=200
)

plt.close()


# =========================================================
# GRAPH — F1 SCORE WITH ERROR BARS
# =========================================================

plt.figure(
    figsize=(10, 6)
)

models_names = results_df["model"]

means = results_df["f1_mean"]

stds = results_df["f1_std"]

plt.bar(
    models_names,
    means,
    yerr=stds,
    capsize=6
)

plt.title(
    "Mealora — 5-Fold Cross-Validation F1 Score",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Mean F1 Score"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_DIR}/cross_validation_f1.png",
    dpi=200
)

plt.close()


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 70)
print("FILES GENERATED")
print("=" * 70)

print(
    RESULTS
)

print(
    RESULTS_TEXT
)

print(
    f"{GRAPH_DIR}/cross_validation_comparison.png"
)

print(
    f"{GRAPH_DIR}/cross_validation_f1.png"
)

print("\n" + "=" * 70)
print("CROSS-VALIDATION COMPLETE")
print("=" * 70)
