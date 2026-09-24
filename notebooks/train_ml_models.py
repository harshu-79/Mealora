import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# =========================================================
# PATHS
# =========================================================

DATA = "datasets/processed/supervised_training_dataset.csv"
RESULTS = "datasets/processed/model_comparison.csv"
RESULTS_TEXT = "datasets/processed/ml_results.txt"
GRAPH_DIR = "datasets/processed/ml_graphs"

# FINAL SELECTED MODEL = RANDOM FOREST
MODEL_PATH = "datasets/processed/mealora_rf_model.joblib"


os.makedirs(GRAPH_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 70)
print("MEALORA RANDOM FOREST TRAINING")
print("=" * 70)

df = pd.read_csv(DATA)

print("\nDataset loaded successfully.")
print("Total samples:", len(df))


# =========================================================
# FEATURES
# =========================================================

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


# Check that all required features exist
missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:
    print("\nERROR: Missing features:")
    for feature in missing_features:
        print("-", feature)
    raise ValueError("Required features are missing from the dataset.")


if "target" not in df.columns:
    raise ValueError("Target column not found in dataset.")


X = df[features].copy()
y = df["target"].copy()


# =========================================================
# CLEAN DATA
# =========================================================

X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(0)

print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nTarget distribution:")
print(y.value_counts())


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# NAIVE BAYES
# =========================================================

print("\nTraining Naive Bayes...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nb = GaussianNB()

nb.fit(
    X_train_scaled,
    y_train
)

nb_pred = nb.predict(X_test_scaled)
nb_prob = nb.predict_proba(X_test_scaled)[:, 1]


# =========================================================
# RANDOM FOREST
# =========================================================

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]


# =========================================================
# HISTOGRAM GRADIENT BOOSTING
# =========================================================

print("Training HistGradientBoosting...")

hgb = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.05,
    max_leaf_nodes=31,
    random_state=42
)

hgb.fit(
    X_train,
    y_train
)

hgb_pred = hgb.predict(X_test)
hgb_prob = hgb.predict_proba(X_test)[:, 1]


# =========================================================
# EVALUATION
# =========================================================

models = {
    "Naive Bayes": (nb_pred, nb_prob),
    "Random Forest": (rf_pred, rf_prob),
    "HistGradientBoosting": (hgb_pred, hgb_prob),
}


results = []
reports = {}


for name, (pred, prob) in models.items():

    accuracy = accuracy_score(
        y_test,
        pred
    )

    precision = precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        prob
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc
    })

    reports[name] = {
        "classification": classification_report(
            y_test,
            pred,
            target_names=[
                "Not Suitable",
                "Suitable"
            ],
            zero_division=0
        ),
        "confusion": confusion_matrix(
            y_test,
            pred
        )
    }


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "f1_score",
    ascending=False
)


results_df.to_csv(
    RESULTS,
    index=False
)


# =========================================================
# RANDOM FOREST IS THE SELECTED MODEL
# =========================================================

selected_model = "Random Forest"


# IMPORTANT:
# Save RANDOM FOREST, not HGB.

joblib.dump(
    {
        "model": rf,
        "features": features,
        "model_name": "Random Forest"
    },
    MODEL_PATH
)


# =========================================================
# SAVE COMPLETE TEXT RESULTS
# =========================================================

with open(
    RESULTS_TEXT,
    "w"
) as f:

    f.write("=" * 70 + "\n")
    f.write("MEALORA MACHINE LEARNING MODEL EVALUATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total dataset samples: {len(df)}\n"
    )

    f.write(
        f"Training samples: {len(X_train)}\n"
    )

    f.write(
        f"Testing samples: {len(X_test)}\n\n"
    )

    f.write("FEATURES USED:\n")

    for feature in features:
        f.write(
            f"- {feature}\n"
        )

    f.write("\n" + "=" * 70 + "\n")
    f.write("MODEL COMPARISON\n")
    f.write("=" * 70 + "\n\n")

    f.write(
        results_df.to_string(index=False)
    )

    f.write("\n\n")

    for name in models:

        f.write("\n" + "=" * 70 + "\n")
        f.write(name.upper() + "\n")
        f.write("=" * 70 + "\n\n")

        pred, prob = models[name]

        f.write(
            classification_report(
                y_test,
                pred,
                target_names=[
                    "Not Suitable",
                    "Suitable"
                ],
                zero_division=0
            )
        )

        f.write("\nConfusion Matrix:\n")

        f.write(
            str(
                confusion_matrix(
                    y_test,
                    pred
                )
            )
        )

        f.write("\n")

    f.write("\n" + "=" * 70 + "\n")
    f.write(
        f"SELECTED MODEL: {selected_model}\n"
    )
    f.write("=" * 70 + "\n")


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)

print(
    "\nSELECTED MODEL: Random Forest"
)

print(
    "\nRandom Forest model saved to:"
)

print(
    MODEL_PATH
)


# =========================================================
# GRAPH 1 — MODEL COMPARISON
# =========================================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc"
]


plot_df = results_df.set_index(
    "model"
)[metrics]


ax = plot_df.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Mealora Model Performance Comparison",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Score"
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
    f"{GRAPH_DIR}/model_comparison.png",
    dpi=200
)

plt.close()


# =========================================================
# GRAPH 2–4 — CONFUSION MATRICES
# =========================================================

confusion_models = {
    "Naive Bayes": nb_pred,
    "Random Forest": rf_pred,
    "HistGradientBoosting": hgb_pred,
}


for name, pred in confusion_models.items():

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        display_labels=[
            "Not Suitable",
            "Suitable"
        ],
        cmap="Blues",
        ax=ax
    )

    plt.title(
        f"{name} — Confusion Matrix",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    filename = (
        name
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        f"{GRAPH_DIR}/confusion_matrix_{filename}.png",
        dpi=200
    )

    plt.close()


# =========================================================
# GRAPH 5 — RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

importance = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values(
    ascending=True
)


plt.figure(
    figsize=(10, 7)
)

importance.plot(
    kind="barh"
)

plt.title(
    "Random Forest Feature Importance",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_DIR}/feature_importance_random_forest.png",
    dpi=200
)

plt.close()


# =========================================================
# GRAPH 6 — TARGET DISTRIBUTION
# =========================================================

target_counts = y.value_counts().sort_index()

labels = [
    "Not Suitable",
    "Suitable"
]


plt.figure(
    figsize=(7, 7)
)

plt.pie(
    target_counts.values,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Supervised Dataset Target Distribution",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    f"{GRAPH_DIR}/target_distribution.png",
    dpi=200
)

plt.close()


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\n" + "=" * 60)
print("FILES GENERATED")
print("=" * 60)

print(
    "\nModel:"
)

print(
    MODEL_PATH
)

print(
    "\nResults:"
)

print(
    RESULTS
)

print(
    RESULTS_TEXT
)

print(
    "\nGraphs:"
)

for file in sorted(
    os.listdir(GRAPH_DIR)
):

    print(
        os.path.join(
            GRAPH_DIR,
            file
        )
    )

print("\n" + "=" * 60)
print("RANDOM FOREST TRAINING COMPLETE")
print("=" * 60)