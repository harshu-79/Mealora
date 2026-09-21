from pathlib import Path
import pandas as pd
import re


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "mealora_grocery_vocabulary_clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "canonical_ingredients.csv"
)


# ============================================================
# CANONICAL INGREDIENTS
# ============================================================

CANONICAL = {
    "onion": {
        "category": "vegetables",
        "aliases": [
            "onion",
            "onions",
            "onion loose",
            "shallot",
            "shallots",
            "pyaz",
        ],
    },
    "tomato": {
        "category": "vegetables",
        "aliases": [
            "tomato",
            "tomatoes",
            "tomato loose",
            "tamatar",
        ],
    },
    "potato": {
        "category": "vegetables",
        "aliases": [
            "potato",
            "potatoes",
            "russet potato",
            "baby potato",
            "aloo",
        ],
    },
    "sweet potato": {
        "category": "vegetables",
        "aliases": [
            "sweet potato",
            "sweet potatoes",
            "shakarkandi",
        ],
    },
    "carrot": {
        "category": "vegetables",
        "aliases": [
            "carrot",
            "carrots",
            "gajar",
        ],
    },
    "spinach": {
        "category": "vegetables",
        "aliases": [
            "spinach",
            "palak",
        ],
    },
    "cabbage": {
        "category": "vegetables",
        "aliases": [
            "cabbage",
        ],
    },
    "cauliflower": {
        "category": "vegetables",
        "aliases": [
            "cauliflower",
            "gobi",
        ],
    },
    "brinjal": {
        "category": "vegetables",
        "aliases": [
            "brinjal",
            "brinjals",
            "eggplant",
            "baingan",
        ],
    },
    "capsicum": {
        "category": "vegetables",
        "aliases": [
            "capsicum",
            "bell pepper",
            "bell peppers",
        ],
    },
    "green chilli": {
        "category": "vegetables",
        "aliases": [
            "green chilli",
            "green chillies",
            "green chili",
            "green chilies",
            "hari mirch",
        ],
    },
    "ginger": {
        "category": "vegetables",
        "aliases": [
            "ginger",
            "adrak",
        ],
    },
    "garlic": {
        "category": "vegetables",
        "aliases": [
            "garlic",
            "lehsun",
        ],
    },
    "cucumber": {
        "category": "vegetables",
        "aliases": [
            "cucumber",
            "cucumbers",
            "kheera",
        ],
    },

    "apple": {
        "category": "fruits",
        "aliases": [
            "apple",
            "apples",
        ],
    },
    "banana": {
        "category": "fruits",
        "aliases": [
            "banana",
            "bananas",
            "kela",
        ],
    },
    "orange": {
        "category": "fruits",
        "aliases": [
            "orange",
            "oranges",
            "santra",
        ],
    },
    "mango": {
        "category": "fruits",
        "aliases": [
            "mango",
            "mangoes",
            "aam",
        ],
    },
    "lemon": {
        "category": "fruits",
        "aliases": [
            "lemon",
            "lemons",
            "nimbu",
        ],
    },

    "rice": {
        "category": "grains",
        "aliases": [
            "rice",
            "raw rice",
            "basmati rice",
            "sona masuri",
            "rice grain",
        ],
    },
    "wheat flour": {
        "category": "grains",
        "aliases": [
            "wheat flour",
            "atta",
            "whole wheat flour",
            "gehun atta",
        ],
    },
    "rava": {
        "category": "grains",
        "aliases": [
            "rava",
            "sooji",
            "semolina",
        ],
    },
    "oats": {
        "category": "grains",
        "aliases": [
            "oats",
            "oat",
        ],
    },

    "toor dal": {
        "category": "pulses",
        "aliases": [
            "toor dal",
            "arhar dal",
            "tuvar dal",
            "pigeon pea",
        ],
    },
    "moong dal": {
        "category": "pulses",
        "aliases": [
            "moong dal",
            "mung dal",
            "green gram",
        ],
    },
    "chana dal": {
        "category": "pulses",
        "aliases": [
            "chana dal",
            "split chickpeas",
        ],
    },
    "urad dal": {
        "category": "pulses",
        "aliases": [
            "urad dal",
            "black gram",
        ],
    },

    "milk": {
        "category": "dairy",
        "aliases": [
            "milk",
            "cow milk",
            "full cream milk",
            "toned milk",
        ],
    },
    "curd": {
        "category": "dairy",
        "aliases": [
            "curd",
            "yogurt",
            "yoghurt",
            "dahi",
        ],
    },
    "paneer": {
        "category": "dairy",
        "aliases": [
            "paneer",
            "cottage cheese",
        ],
    },
    "butter": {
        "category": "dairy",
        "aliases": [
            "butter",
        ],
    },
    "cheese": {
        "category": "dairy",
        "aliases": [
            "cheese",
        ],
    },

    "egg": {
        "category": "protein",
        "aliases": [
            "egg",
            "eggs",
        ],
    },
    "chicken": {
        "category": "protein",
        "aliases": [
            "chicken",
            "chicken breast",
            "chicken breasts",
        ],
    },
    "fish": {
        "category": "protein",
        "aliases": [
            "fish",
        ],
    },

    "bread": {
        "category": "bakery",
        "aliases": [
            "bread",
            "loaf",
            "loaves",
        ],
    },

    "sugar": {
        "category": "staples",
        "aliases": [
            "sugar",
            "brown sugar",
            "palm sugar",
            "jaggery",
            "gur",
        ],
    },
    "salt": {
        "category": "spices",
        "aliases": [
            "salt",
            "table salt",
        ],
    },
    "cooking oil": {
        "category": "staples",
        "aliases": [
            "cooking oil",
            "vegetable oil",
            "sunflower oil",
            "groundnut oil",
        ],
    },
    "ghee": {
        "category": "dairy",
        "aliases": [
            "ghee",
        ],
    },

    "red chilli powder": {
        "category": "spices",
        "aliases": [
            "red chilli powder",
            "chilli powder",
            "chili powder",
        ],
    },
    "turmeric powder": {
        "category": "spices",
        "aliases": [
            "turmeric",
            "turmeric powder",
            "haldi",
        ],
    },
    "cumin": {
        "category": "spices",
        "aliases": [
            "cumin",
            "jeera",
        ],
    },
    "coriander powder": {
        "category": "spices",
        "aliases": [
            "coriander powder",
            "dhaniya powder",
        ],
    },
}


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_dataset():
    print("\nCreating canonical ingredient dataset...")

    # We read this file mainly to keep our canonical vocabulary
    # tied to the real processed dataset.
    source_df = pd.read_csv(
        INPUT_FILE,
        dtype=str,
        low_memory=False,
    )

    print(
        f"Processed source rows available: "
        f"{len(source_df):,}"
    )

    rows = []

    for canonical_name, info in CANONICAL.items():
        for alias in info["aliases"]:
            rows.append(
                {
                    "canonical_name": canonical_name,
                    "alias": alias,
                    "normalized_alias": normalize(alias),
                    "category": info["category"],
                }
            )

    output_df = pd.DataFrame(rows)

    output_df = output_df.drop_duplicates(
        subset=[
            "canonical_name",
            "normalized_alias",
        ]
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n======================================"
    )
    print(
        "CANONICAL INGREDIENT DATASET CREATED"
    )
    print(
        "======================================"
    )
    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )
    print(
        f"\nCanonical ingredients: "
        f"{output_df['canonical_name'].nunique()}"
    )
    print(
        f"Total aliases: "
        f"{len(output_df):,}"
    )


if __name__ == "__main__":
    build_dataset()