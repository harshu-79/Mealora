from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


MASTER_COLUMNS = [
    "recipe_id",
    "recipe_name",
    "ingredients",
    "cuisine",
    "state",
    "region",
    "meal_type",
    "diet_type",
    "prep_time",
    "cook_time",
    "total_time",
    "nutrition",
    "season",
    "weather_tags",
    "spice_level",
    "difficulty",
    "ingredient_count",
    "rating",
    "instructions",
    "image_url",
    "source",
    "recipe_scope",
]


def clean(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_name(value):
    return re.sub(
        r"[^a-z0-9\s]",
        "",
        clean(value).lower()
    )


def normalize_cuisine(value):
    text = clean(value).lower()

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


INDIAN_KEYWORDS = [
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
    text = clean(cuisine).lower()

    return (
        "indian"
        if any(k in text for k in INDIAN_KEYWORDS)
        else "non_indian"
    )


def count_ingredients(value):
    text = clean(value)

    if not text:
        return 0

    parts = re.split(
        r",|\n|;",
        text
    )

    return len(
        [p for p in parts if p.strip()]
    )


def prepare(df, source, mapping):
    out = pd.DataFrame()

    for master_col in MASTER_COLUMNS:
        source_col = mapping.get(master_col)

        if source_col and source_col in df.columns:
            out[master_col] = df[source_col].map(clean)
        else:
            out[master_col] = ""

    out["source"] = source

    out["cuisine"] = (
        out["cuisine"]
        .map(normalize_cuisine)
    )

    out["recipe_scope"] = (
        out["cuisine"]
        .map(classify_scope)
    )

    out["ingredient_count"] = (
        out["ingredients"]
        .map(count_ingredients)
    )

    return out


# ============================================================
# DATASET DETECTION
# ============================================================

frames = []

for file in RAW.glob("*.csv"):
    try:
        sample = pd.read_csv(
            file,
            nrows=2,
            low_memory=False
        )

        cols = set(sample.columns)

        # 7,288 recipe dataset
        if {
            "name_of_Dish",
            "Ingredients_of_Dish",
        }.issubset(cols):

            df = pd.read_csv(
                file,
                low_memory=False
            )

            frames.append(
                prepare(
                    df,
                    "main_7288",
                    {
                        "recipe_name":
                            "name_of_Dish",
                        "ingredients":
                            "Ingredients_of_Dish",
                        "cuisine":
                            "Cuisine_name",
                        "diet_type":
                            "Diet_Type",
                        "meal_type":
                            "Course_name",
                        "prep_time":
                            "Prepration_time",
                        "cook_time":
                            "Cooking_time",
                        "total_time":
                            "Total_time",
                        "rating":
                            "Ratings_of_Dish",
                        "instructions":
                            "Recipe_Instructions",
                    },
                )
            )

            print(
                f"Detected 7,288-style dataset: "
                f"{file.name}"
            )

        # ELR dataset
        elif {
            "RecipeName",
            "TranslatedRecipeName",
            "TranslatedIngredients",
        }.issubset(cols):

            df = pd.read_csv(
                file,
                low_memory=False
            )

            frames.append(
                prepare(
                    df,
                    "ELR_1000",
                    {
                        "recipe_name":
                            "TranslatedRecipeName",
                        "ingredients":
                            "TranslatedIngredients",
                        "cuisine":
                            "Cuisine",
                        "diet_type":
                            "Diet",
                        "meal_type":
                            "Course",
                        "prep_time":
                            "PrepTimeInMins",
                        "cook_time":
                            "CookTimeInMins",
                        "total_time":
                            "TotalTimeInMins",
                        "instructions":
                            "TranslatedInstructions",
                    },
                )
            )

            print(
                f"Detected ELR dataset: "
                f"{file.name}"
            )

        # HuggingFace-style recipe dataset
        elif {
            "name",
            "ingredients",
            "cuisine",
            "course",
            "diet",
        }.issubset(cols):

            df = pd.read_csv(
                file,
                low_memory=False
            )

            frames.append(
                prepare(
                    df,
                    "huggingface_recipe",
                    {
                        "recipe_name":
                            "name",
                        "ingredients":
                            "ingredients",
                        "cuisine":
                            "cuisine",
                        "diet_type":
                            "diet",
                        "meal_type":
                            "course",
                        "prep_time":
                            "prep_time",
                        "instructions":
                            "instructions",
                        "image_url":
                            "image_url",
                    },
                )
            )

            print(
                f"Detected HuggingFace-style dataset: "
                f"{file.name}"
            )

        # 255 regional dataset
        elif {
            "name",
            "ingredients",
            "state",
            "region",
        }.issubset(cols):

            df = pd.read_csv(
                file,
                low_memory=False
            )

            frames.append(
                prepare(
                    df,
                    "regional_255",
                    {
                        "recipe_name":
                            "name",
                        "ingredients":
                            "ingredients",
                        "diet_type":
                            "diet",
                        "meal_type":
                            "course",
                        "prep_time":
                            "prep_time",
                        "cook_time":
                            "cook_time",
                        "state":
                            "state",
                        "region":
                            "region",
                    },
                )
            )

            print(
                f"Detected regional dataset: "
                f"{file.name}"
            )

    except Exception as e:
        print(
            f"Skipping {file.name}: {e}"
        )


if not frames:
    raise RuntimeError(
        "No recognized recipe datasets found."
    )


# ============================================================
# COMBINE
# ============================================================

master = pd.concat(
    frames,
    ignore_index=True
)

print(
    f"\nCombined rows: {len(master):,}"
)


# ============================================================
# REMOVE EMPTY RECIPE NAMES
# ============================================================

master = master[
    master["recipe_name"].str.len() > 1
].copy()


# ============================================================
# NORMALIZED DEDUPLICATION KEY
# ============================================================

master["_recipe_key"] = (
    master["recipe_name"]
    .map(normalize_name)
    + "|"
    + master["ingredients"]
    .map(normalize_name)
)


# Prefer rows with more information.
master["_filled_fields"] = (
    master[
        [
            "ingredients",
            "cuisine",
            "state",
            "region",
            "meal_type",
            "diet_type",
            "cook_time",
            "instructions",
        ]
    ]
    .apply(
        lambda row:
        sum(bool(clean(x)) for x in row),
        axis=1
    )
)

master = master.sort_values(
    "_filled_fields",
    ascending=False
)

master = master.drop_duplicates(
    "_recipe_key",
    keep="first"
)


# ============================================================
# IDS
# ============================================================

master = master.reset_index(
    drop=True
)

master["recipe_id"] = (
    master.index + 1
)


# ============================================================
# FINAL CLEANUP
# ============================================================

master = master[
    MASTER_COLUMNS
]


# ============================================================
# SAVE
# ============================================================

output = (
    OUT /
    "recipe_master_dataset.csv"
)

master.to_csv(
    output,
    index=False
)


print("\n====================================")
print("MEALORA RECIPE MASTER CREATED")
print("====================================")

print(
    f"Output: {output}"
)

print(
    f"Final recipes: {len(master):,}"
)

print("\nSources:")
print(
    master["source"]
    .value_counts()
)

print("\nRecipe scope:")
print(
    master["recipe_scope"]
    .value_counts()
)

print("\nMissing fields:")
print(
    master
    .replace("", pd.NA)
    .isna()
    .sum()
    .sort_values()
)
