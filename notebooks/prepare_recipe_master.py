from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Find the 7,288-recipe dataset automatically
main_file = None

for f in RAW.glob("*.csv"):
    try:
        sample = pd.read_csv(f, nrows=2)
        cols = {c.strip() for c in sample.columns}

        required = {
            "name_of_Dish",
            "Cuisine_name",
            "Ingredients_of_Dish",
        }

        if required.issubset(cols):
            main_file = f
            break
    except Exception:
        pass

if main_file is None:
    raise FileNotFoundError(
        "Could not find the 7,288-recipe dataset."
    )

print(f"Loading: {main_file.name}")

df = pd.read_csv(
    main_file,
    low_memory=False
)

print(f"Rows loaded: {len(df):,}")

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def clean_text(value):
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(value).strip()
    )

def normalize_cuisine(value):
    text = clean_text(value).lower()

    # Remove dataset labels / wrappers
    text = re.sub(
        r"cuisine\s*:\s*",
        "",
        text
    )

    text = re.sub(
        r"[\[\]'\"()]",
        "",
        text
    )

    # Normalize common names
    replacements = {
        "north indian recipes": "north indian",
        "south indian recipes": "south indian",
        "gujarati recipes": "gujarati",
        "bengali recipes": "bengali",
        "punjabi recipes": "punjabi",
        "kerala recipes": "kerala",
        "tamil nadu recipes": "tamil nadu",
        "andhra recipes": "andhra",
        "maharashtrian recipes": "maharashtrian",
        "karnataka recipes": "karnataka",
        "rajasthani recipes": "rajasthani",
        "goan recipes": "goan",
        "chettinad recipes": "chettinad",
        "kashmiri recipes": "kashmiri",
    }

    return replacements.get(text, text)

# --------------------------------------------------
# Rename columns
# --------------------------------------------------

rename_map = {
    "name_of_Dish": "recipe_name",
    "Diet_Type": "diet_type",
    "Course_name": "meal_type",
    "Discrption_of_Dish": "description",
    "Cuisine_name": "cuisine",
    "Ratings_of_Dish": "rating",
    "Ingredients_of_Dish": "ingredients",
    "Prepration_time": "prep_time",
    "Cooking_time": "cook_time",
    "Total_time": "total_time",
    "Makes": "servings",
    "Recipe_Instructions": "instructions",
}

df = df.rename(
    columns=rename_map
)

# --------------------------------------------------
# Clean core fields
# --------------------------------------------------

for col in [
    "recipe_name",
    "diet_type",
    "meal_type",
    "description",
    "cuisine",
    "ingredients",
    "prep_time",
    "cook_time",
    "total_time",
    "servings",
    "instructions",
]:
    if col in df.columns:
        df[col] = df[col].map(clean_text)

# --------------------------------------------------
# Normalize cuisine
# --------------------------------------------------

df["cuisine"] = (
    df["cuisine"]
    .map(normalize_cuisine)
)

# --------------------------------------------------
# Add recipe scope
# Keep ALL recipes.
# --------------------------------------------------

indian_keywords = [
    "indian",
    "north indian",
    "south indian",
    "bengali",
    "maharashtrian",
    "tamil nadu",
    "kerala",
    "gujarati",
    "karnataka",
    "andhra",
    "rajasthani",
    "punjabi",
    "goan",
    "chettinad",
    "kashmiri",
    "konkan",
    "hyderabadi",
    "odisha",
    "oriya",
    "assamese",
    "lucknowi",
    "awadhi",
    "mughlai",
]

def classify_scope(cuisine):
    text = str(cuisine).lower()

    return (
        "indian"
        if any(
            keyword in text
            for keyword in indian_keywords
        )
        else "non_indian"
    )

df["recipe_scope"] = (
    df["cuisine"]
    .map(classify_scope)
)

# --------------------------------------------------
# Add IDs
# --------------------------------------------------

df.insert(
    0,
    "recipe_id",
    range(1, len(df) + 1)
)

# --------------------------------------------------
# Ingredient count
# --------------------------------------------------

def ingredient_count(value):
    if not value:
        return 0

    text = str(value)

    # Dataset stores ingredients roughly as
    # list-like strings.
    parts = re.split(
        r",\s*(?=['\"]?\w)",
        text
    )

    return max(
        1,
        len(
            [
                p
                for p in parts
                if p.strip()
            ]
        )
    )

df["ingredient_count"] = (
    df["ingredients"]
    .map(ingredient_count)
)

# --------------------------------------------------
# Remove exact duplicate recipes
# --------------------------------------------------

before = len(df)

df = df.drop_duplicates(
    subset=[
        "recipe_name",
        "ingredients",
        "instructions",
    ],
    keep="first"
)

print(
    f"Duplicates removed: "
    f"{before - len(df):,}"
)

# --------------------------------------------------
# Final column order
# --------------------------------------------------

columns = [
    "recipe_id",
    "recipe_name",
    "ingredients",
    "cuisine",
    "recipe_scope",
    "diet_type",
    "meal_type",
    "prep_time",
    "cook_time",
    "total_time",
    "servings",
    "rating",
    "ingredient_count",
    "description",
    "instructions",
]

columns = [
    c for c in columns
    if c in df.columns
]

df = df[columns]

# --------------------------------------------------
# Save
# --------------------------------------------------

output = (
    OUT /
    "recipes_master_cleaned.csv"
)

df.to_csv(
    output,
    index=False
)

print("\n===================================")
print("RECIPE MASTER DATASET CREATED")
print("===================================")

print(f"Output: {output}")
print(f"Final rows: {len(df):,}")

print("\nRecipe scope:")
print(
    df["recipe_scope"]
    .value_counts()
)

print("\nCuisine examples:")
print(
    df["cuisine"]
    .value_counts()
    .head(15)
)
