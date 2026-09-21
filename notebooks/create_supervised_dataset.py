import pandas as pd
import numpy as np

RECIPES = "datasets/processed/recipe_master_dataset.csv"
PANTRY = "datasets/processed/recipe_pantry_features.csv"
EXPIRY = "datasets/processed/recipe_expiry_features.csv"
OUTPUT = "datasets/processed/supervised_training_dataset.csv"

recipes = pd.read_csv(RECIPES, low_memory=False)
pantry = pd.read_csv(PANTRY)
expiry = pd.read_csv(EXPIRY)

df = recipes.merge(
    pantry[["recipe_id", "pantry_match_pct", "missing_ingredient_count"]],
    on="recipe_id",
    how="inner"
)

df = df.merge(
    expiry[["recipe_id", "expiry_score"]],
    on="recipe_id",
    how="left"
)

print("Recipes after merge:", len(df))

numeric_cols = [
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

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# Independent suitability target
df["suitability_score"] = (
    0.35 * df["pantry_match_pct"]
    + 0.20 * df["expiry_score"]
    + 0.20 * (df["context_score"] / 0.20)
    + 0.15 * df["time_match"]
    + 0.10 * df["diet_match"]
)

threshold = df["suitability_score"].quantile(0.60)

df["target"] = (
    df["suitability_score"] >= threshold
).astype(int)

feature_cols = [
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

output_cols = [
    "recipe_id",
    "recipe_name",
] + feature_cols + [
    "suitability_score",
    "target",
]

training = df[output_cols].copy()

training = training.replace(
    [np.inf, -np.inf],
    np.nan
)

training = training.dropna()

training.to_csv(OUTPUT, index=False)

print("\n========================================")
print("SUPERVISED DATASET CREATED")
print("========================================")
print("Shape:", training.shape)
print("\nTarget distribution:")
print(training["target"].value_counts().sort_index())
print("\nTarget percentage:")
print(training["target"].value_counts(normalize=True).sort_index())
print("\nFeatures:")
print(feature_cols)
print("\nSaved to:")
print(OUTPUT)
