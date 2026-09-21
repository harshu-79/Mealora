from pathlib import Path
import html
import re

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "master_grocery_vocabulary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "mealora_grocery_vocabulary_clean.csv"
)


# ============================================================
# KEYWORDS
# ============================================================

FOOD_KEYWORDS = [
    "rice",
    "wheat",
    "flour",
    "atta",
    "dal",
    "lentil",
    "bean",
    "beans",
    "potato",
    "tomato",
    "onion",
    "carrot",
    "spinach",
    "cabbage",
    "cauliflower",
    "brinjal",
    "eggplant",
    "capsicum",
    "pepper",
    "chilli",
    "chili",
    "ginger",
    "garlic",
    "apple",
    "banana",
    "orange",
    "mango",
    "lemon",
    "lime",
    "milk",
    "curd",
    "yogurt",
    "yoghurt",
    "paneer",
    "cheese",
    "butter",
    "ghee",
    "egg",
    "chicken",
    "fish",
    "meat",
    "bread",
    "pasta",
    "noodle",
    "cereal",
    "corn flakes",
    "oat",
    "oats",
    "sugar",
    "salt",
    "oil",
    "biscuit",
    "biscuits",
    "cookie",
    "cookies",
    "snack",
    "juice",
    "sauce",
    "jam",
    "honey",
    "peanut",
    "peanuts",
    "almond",
    "almonds",
    "cashew",
    "cashews",
    "walnut",
    "walnuts",
    "dry fruit",
    "dry fruits",
]


NON_FOOD_KEYWORDS = [
    "detergent",
    "washing powder",
    "washing liquid",
    "soap",
    "shampoo",
    "conditioner",
    "toothpaste",
    "toothbrush",
    "sanitary",
    "diaper",
    "battery",
    "batteries",
    "screw",
    "screws",
    "bolt",
    "bolts",
    "backpack",
    "school bag",
    "stationery",
    "pen",
    "pencil",
    "cleaner",
    "floor cleaner",
    "toilet cleaner",
    "dishwash",
    "dish washer",
    "dishwashing",
    "laundry",
    "razor",
    "cosmetic",
    "cosmetics",
    "perfume",
    "deodorant",
    "lotion",
    "medicine",
    "tablet",
    "capsule",
    "wine",
    "beer",
    "whisky",
    "vodka",
    "cigarette",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_series(series):
    series = (
        series
        .fillna("")
        .astype(str)
        .map(html.unescape)
        .str.lower()
        .str.strip()
    )

    # Remove escaped HTML leftovers.
    series = series.str.replace(
        r"&[a-z0-9#]+;",
        " ",
        regex=True,
    )

    # Remove URLs.
    series = series.str.replace(
        r"https?://\S+",
        " ",
        regex=True,
    )

    # Replace weird punctuation with spaces.
    series = series.str.replace(
        r"[^a-z0-9\s&'-]",
        " ",
        regex=True,
    )

    # Normalize whitespace.
    series = series.str.replace(
        r"\s+",
        " ",
        regex=True,
    )

    return series.str.strip()


def normalize_name_series(series):
    series = clean_series(series)

    # Remove package sizes.
    series = series.str.replace(
        r"\b\d+(?:\.\d+)?\s*"
        r"(kg|kgs|kilo|kilos|kilogram|kilograms|"
        r"g|gm|gram|grams|"
        r"ml|l|ltr|liter|litre|litres|"
        r"pcs|pc|piece|pieces|"
        r"pack|packs|box|boxes|bag|bags|"
        r"bottle|bottles|can|cans|"
        r"jar|jars)\b",
        " ",
        regex=True,
    )

    # Remove "2 x 500 g" style quantities.
    series = series.str.replace(
        r"\b\d+\s*x\s*\d+(?:\.\d+)?\s*"
        r"(kg|g|ml|l|ltr)\b",
        " ",
        regex=True,
    )

    # Remove leading product codes.
    series = series.str.replace(
        r"^\d{4,14}\s+",
        " ",
        regex=True,
    )

    # Normalize common plural forms.
    singular_replacements = {
        r"\bpotatoes\b": "potato",
        r"\btomatoes\b": "tomato",
        r"\bonions\b": "onion",
        r"\bcarrots\b": "carrot",
        r"\bapples\b": "apple",
        r"\bbananas\b": "banana",
        r"\boranges\b": "orange",
        r"\bmangoes\b": "mango",
        r"\beggs\b": "egg",
        r"\bcucumbers\b": "cucumber",
        r"\bcapsicums\b": "capsicum",
        r"\bpeppers\b": "pepper",
        r"\bloaves\b": "loaf",
        r"\bbox(es)?\b": "box",
        r"\bbags\b": "bag",
        r"\bbottles\b": "bottle",
        r"\bjars\b": "jar",
        r"\bcans\b": "can",
        r"\bpacks\b": "pack",
    }

    for pattern, replacement in singular_replacements.items():
        series = series.str.replace(
            pattern,
            replacement,
            regex=True,
        )

    series = series.str.replace(
        r"\s+",
        " ",
        regex=True,
    )

    return series.str.strip()


# ============================================================
# QUALITY FILTERS
# ============================================================

def contains_keywords(series, keywords):
    mask = pd.Series(
        False,
        index=series.index,
    )

    for keyword in keywords:
        mask |= series.str.contains(
            re.escape(keyword),
            regex=True,
            na=False,
        )

    return mask


# ============================================================
# MAIN CLEANING
# ============================================================

def build_clean_vocabulary():

    print("\nLoading Mealora vocabulary...")

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        low_memory=False,
    )

    df = df.fillna("")

    print(
        f"Input rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Clean original fields
    # --------------------------------------------------------

    df["product_name"] = clean_series(
        df["product_name"]
    )

    df["normalized_name"] = normalize_name_series(
        df["normalized_name"]
    )

    df["brand"] = clean_series(
        df["brand"]
    )

    df["category"] = clean_series(
        df["category"]
    )

    df["subcategory"] = clean_series(
        df["subcategory"]
    )

    df["source"] = clean_series(
        df["source"]
    )

    # --------------------------------------------------------
    # Combined searchable text
    # --------------------------------------------------------

    combined = (
        df["product_name"]
        + " "
        + df["normalized_name"]
        + " "
        + df["category"]
        + " "
        + df["subcategory"]
    )

    combined = combined.str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()

    # --------------------------------------------------------
    # Strong food / non-food signals
    # --------------------------------------------------------

    food_mask = (
        df["food_or_non_food"]
        .eq("food")
        |
        contains_keywords(
            combined,
            FOOD_KEYWORDS,
        )
    )

    non_food_mask = contains_keywords(
        combined,
        NON_FOOD_KEYWORDS,
    )

    # Food wins unless there is a strong non-food
    # signal such as detergent, shampoo, backpack, etc.
    final_food_mask = (
        food_mask
        & ~non_food_mask
    )

    df = df[
        final_food_mask
    ].copy()

    # --------------------------------------------------------
    # Remove obviously bad names
    # --------------------------------------------------------

    df = df[
        df["normalized_name"].str.len()
        >= 3
    ]

    # Must contain at least one alphabetic character.
    df = df[
        df["normalized_name"].str.contains(
            r"[a-z]",
            regex=True,
            na=False,
        )
    ]

    # Remove rows where the normalized name is mostly
    # codes/numbers.
    df = df[
        ~df["normalized_name"].str.match(
            r"^[0-9\s&'-]+$",
            na=False,
        )
    ]

    # Remove very noisy names containing long digit runs.
    df = df[
        ~df["normalized_name"].str.contains(
            r"\d{6,}",
            regex=True,
            na=False,
        )
    ]

    # --------------------------------------------------------
    # Prefer meaningful names
    # --------------------------------------------------------

    df["name_word_count"] = (
        df["normalized_name"]
        .str.split()
        .str.len()
    )

    df = df[
        df["name_word_count"] <= 25
    ]

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "normalized_name",
            "brand",
        ],
        keep="first",
    )

    df = df.drop(
        columns=[
            "name_word_count",
        ]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "normalized_name",
            "brand",
            "source",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n======================================"
    )
    print(
        "CLEAN MEALORA VOCABULARY CREATED"
    )
    print(
        "======================================"
    )

    print(
        f"\nOutput:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        f"\nFinal rows: "
        f"{len(df):,}"
    )

    print(
        "\nSources:"
    )

    print(
        df["source"].value_counts()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_clean_vocabulary()