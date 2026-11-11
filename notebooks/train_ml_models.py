import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.model_selection import train_test_split
import joblib

DATA = "datasets/processed/supervised_training_dataset.csv"
RESULTS = "datasets/processed/model_comparison.csv"
RESULTS_TEXT = "datasets/processed/ml_results.txt"
MODEL_PATH = "datasets/processed/mealora_hgb_model.joblib"
RF_MODEL_PATH = "datasets/processed/mealora_rf_model.joblib"

df = pd.read_csv(DATA)

FEATURES = [
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

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = df["target"]

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

print("========================================")
print("MEALORA ML TRAINING")
print("========================================")
print("Dataset:", DATA)
print("Rows:", len(df))
print("Features:", len(FEATURES))
print("\nFeatures:")
for i, feature in enumerate(FEATURES, 1):
    print(f"{i}. {feature}")

print("\nTarget distribution:")
print(y.value_counts().sort_index())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=42
    ),
}

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    if name == "HistGradientBoosting":
        joblib.dump(model, MODEL_PATH)
        print("Saved HGB model:", MODEL_PATH)

    if name == "Random Forest":
        joblib.dump(model, RF_MODEL_PATH)
        print("Saved RF model:", RF_MODEL_PATH)


results_df = pd.DataFrame(results)

results_df.to_csv(RESULTS, index=False)

with open(RESULTS_TEXT, "w") as f:
    f.write("MEALORA ML MODEL RESULTS\n")
    f.write("========================\n\n")
    f.write("FEATURES\n")
    for feature in FEATURES:
        f.write(f"- {feature}\n")

    f.write("\nMODEL RESULTS\n")
    f.write(results_df.to_string(index=False))
    f.write("\n")

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")
print("\nModel comparison:")
print(results_df.to_string(index=False))

print("\nHGB model:")
print(MODEL_PATH)

print("\nRF model:")
print(RF_MODEL_PATH)

print("\nResults:")
print(RESULTS)
