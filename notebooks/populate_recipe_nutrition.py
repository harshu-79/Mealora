import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher

RECIPE_FILE = "datasets/processed/recipe_master_dataset.csv"
NUTRITION_FILE = "datasets/raw/Indian_Food_Ingredients_Nutrition_CookingMethods.csv"

OUTPUT_FILE = "datasets/processed/recipe_master_dataset.csv"
BACKUP_FILE = "datasets/processed/recipe_master_dataset_before_nutrition.csv"

NUTRITION_COLUMNS = {
    "Calories (kcal)": "calories",
    "Carbohydrates (g)": "carbohydrates",
    "Protein (g)": "protein",
    "Fats (g)": "fat",
    "Fibre (g)": "fibre",
    "Sodium (mg)": "sodium",
    "Calcium (mg)": "calcium",
    "Iron (mg)": "iron",
    "Vitamin C (mg)": "vitamin_c",
    "Folate (µg)": "folate",
}


def normalize(text):
    text = str(text).lower()

    text = re.sub(r"\b(recipe|recipes)\b", " ", text)
    text = re.sub(r"\b(style|indian|south indian|north indian)\b", " ", text)

    text = re.sub(r"\([^)]*\)", " ", text)

    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


print("=" * 70)
print("MEALORA NUTRITION DATA POPULATION")
print("=" * 70)

print("\nLoading recipe dataset...")
recipes = pd.read_csv(RECIPE_FILE)
print("Recipes:", len(recipes))

print("\nLoading nutrition dataset...")
nutrition = pd.read_csv(NUTRITION_FILE)
print("Nutrition records:", len(nutrition))

# ------------------------------------------------------------
# BACKUP
# ------------------------------------------------------------

print("\nCreating backup...")
recipes.to_csv(BACKUP_FILE, index=False)
print("Backup:", BACKUP_FILE)

# ------------------------------------------------------------
# PREPARE NAMES
# ------------------------------------------------------------

recipes["_match_name"] = recipes["recipe_name"].fillna("").apply(normalize)

nutrition["_match_name"] = nutrition["final_food_name"].fillna("").apply(normalize)

# Also use best_match_clean when available
nutrition["_best_match"] = nutrition["best_match_clean"].fillna("").apply(normalize)

# ------------------------------------------------------------
# ENSURE OUTPUT COLUMNS EXIST
# ------------------------------------------------------------

for col in NUTRITION_COLUMNS.values():
    if col not in recipes.columns:
        recipes[col] = np.nan

# ------------------------------------------------------------
# EXACT MATCH DICTIONARIES
# ------------------------------------------------------------

nutrition_by_name = {}

for _, row in nutrition.iterrows():

    names = [
        row["_match_name"],
        row["_best_match"]
    ]

    for name in names:

        if not name:
            continue

        # Keep first reliable record
        if name not in nutrition_by_name:
            nutrition_by_name[name] = row

# ------------------------------------------------------------
# MATCHING
# ------------------------------------------------------------

exact_matches = 0
fuzzy_matches = 0
no_matches = 0

match_scores = []

nutrition_names = list(nutrition_by_name.keys())

print("\nMatching recipes...")

for idx in range(len(recipes)):

    recipe_name = recipes.iloc[idx]["_match_name"]

    if not recipe_name:
        no_matches += 1
        match_scores.append(0)
        continue

    matched_row = None
    score = 0

    # --------------------------------------------------------
    # EXACT MATCH
    # --------------------------------------------------------

    if recipe_name in nutrition_by_name:

        matched_row = nutrition_by_name[recipe_name]
        score = 1.0
        exact_matches += 1

    # --------------------------------------------------------
    # FUZZY MATCH
    # --------------------------------------------------------

    else:

        best_name = None
        best_score = 0

        recipe_tokens = set(recipe_name.split())

        for candidate in nutrition_names:

            candidate_tokens = set(candidate.split())

            if not recipe_tokens or not candidate_tokens:
                continue

            token_overlap = len(
                recipe_tokens.intersection(candidate_tokens)
            ) / max(
                len(recipe_tokens.union(candidate_tokens)), 1
            )

            sequence_score = SequenceMatcher(
                None,
                recipe_name,
                candidate
            ).ratio()

            combined_score = (
                0.6 * token_overlap +
                0.4 * sequence_score
            )

            if combined_score > best_score:

                best_score = combined_score
                best_name = candidate

        # Conservative threshold
        if best_name is not None and best_score >= 0.78:

            matched_row = nutrition_by_name[best_name]
            score = best_score
            fuzzy_matches += 1

    # --------------------------------------------------------
    # WRITE NUTRITION
    # --------------------------------------------------------

    if matched_row is not None:

        for source_col, target_col in NUTRITION_COLUMNS.items():

            value = pd.to_numeric(
                matched_row[source_col],
                errors="coerce"
            )

            recipes.at[idx, target_col] = value

        match_scores.append(score)

    else:

        no_matches += 1
        match_scores.append(0)


recipes["nutrition_match_score"] = match_scores

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

recipes.drop(
    columns=["_match_name"],
    inplace=True,
    errors="ignore"
)

recipes.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

nutrition_available = recipes["calories"].notna().sum()

print("\n" + "=" * 70)
print("NUTRITION POPULATION COMPLETE")
print("=" * 70)

print("\nTotal recipes:", len(recipes))

print("Exact matches:", exact_matches)
print("Fuzzy matches:", fuzzy_matches)
print("No matches:", no_matches)

print(
    "Recipes with nutrition:",
    nutrition_available
)

print(
    "Nutrition coverage:",
    round(
        nutrition_available / len(recipes) * 100,
        2
    ),
    "%"
)

print("\nNutrition columns populated:")

for col in NUTRITION_COLUMNS.values():

    print(
        f"{col:20s}: "
        f"{recipes[col].notna().sum()} values"
    )

print("\nBackup saved at:")
print(BACKUP_FILE)

print("\nUpdated dataset:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
