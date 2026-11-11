#!/usr/bin/env python3

"""
MEALORA - BATCHED GEMINI NUTRITION POPULATION

This version sends MANY recipes in ONE Gemini request.
It is designed to avoid the 503 / 429 problem caused by making
one API request per recipe.

Run from Mealora root:

    python3 mealora_final_patches/populate_nutrition_gemini_batched.py

Recommended tonight:

    python3 mealora_final_patches/populate_nutrition_gemini_batched.py \
        --batch-size 20 \
        --workers 1

If that works, you can try:

    --batch-size 30 --workers 1

Existing nutrition values are preserved.
Gemini-generated values are marked:
    Gemini estimated per serving
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai is not installed.")
    print("Run: pip3 install -U google-genai")
    sys.exit(1)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

MASTER_PATH = (
    ROOT
    / "datasets"
    / "processed"
    / "recipe_master_dataset.csv"
)

CHECKPOINT_PATH = (
    ROOT
    / "datasets"
    / "processed"
    / "recipe_master_dataset_gemini_checkpoint.csv"
)

BACKUP_PATH = (
    ROOT
    / "datasets"
    / "processed"
    / "recipe_master_dataset_before_gemini_nutrition.csv"
)

ENV_PATH = ROOT / "backend" / ".env"


# ============================================================
# NUTRITION
# ============================================================

NUTRITION_COLUMNS = [
    "calories",
    "carbohydrates",
    "protein",
    "fat",
    "fibre",
    "sodium",
    "calcium",
    "iron",
    "vitamin_c",
    "folate",
]


# ============================================================
# GEMINI
# ============================================================

MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

_thread_local = threading.local()


def load_env():
    """
    Read backend/.env without requiring python-dotenv.
    """

    if not ENV_PATH.exists():
        return

    for raw in ENV_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split(
            "=",
            1
        )

        key = key.strip()
        value = value.strip()

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        if key:
            os.environ.setdefault(
                key,
                value
            )


load_env()


def get_client():

    if not hasattr(
        _thread_local,
        "client"
    ):

        key = os.environ.get(
            "GEMINI_API_KEY",
            ""
        ).strip()

        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY not found in backend/.env"
            )

        _thread_local.client = genai.Client(
            api_key=key
        )

    return _thread_local.client


# ============================================================
# DATASET HELPERS
# ============================================================

def find_column(
    df,
    names
):

    lower = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:

        if name.lower() in lower:
            return lower[name.lower()]

    return None


def clean(value, limit=1800):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text[:limit]


def missing(value):

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    text = str(value).strip().lower()

    return text in {
        "",
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        "-",
    }


# ============================================================
# GEMINI BATCH PROMPT
# ============================================================

def build_prompt(
    recipes
):

    recipe_text = []

    for item in recipes:

        recipe_text.append(
            {
                "index": item["index"],
                "recipe_name": item["recipe_name"],
                "ingredients": item["ingredients"],
            }
        )

    return f"""
You are estimating nutrition for an Indian recipe dataset.

For every recipe below, estimate nutrition for ONE NORMAL ADULT
SERVING.

Return exactly ONE JSON object containing an "items" array.

Each item MUST contain:

index
calories
carbohydrates
protein
fat
fibre
sodium
calcium
iron
vitamin_c
folate

Units:

calories = kcal
carbohydrates = grams
protein = grams
fat = grams
fibre = grams
sodium = mg
calcium = mg
iron = mg
vitamin_c = mg
folate = micrograms

Rules:

1. Return one item for EVERY input recipe.
2. Keep the same index.
3. Return numeric values only.
4. Never return null.
5. Nutrition values must not be negative.
6. Use the recipe name and ingredients.
7. Use reasonable Indian-food serving estimates.
8. These are estimated values for a recommendation system,
   not laboratory measurements.

Recipes:

{json.dumps(recipe_text, ensure_ascii=False)}
"""


SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {
                        "type": "INTEGER"
                    },
                    "calories": {
                        "type": "NUMBER"
                    },
                    "carbohydrates": {
                        "type": "NUMBER"
                    },
                    "protein": {
                        "type": "NUMBER"
                    },
                    "fat": {
                        "type": "NUMBER"
                    },
                    "fibre": {
                        "type": "NUMBER"
                    },
                    "sodium": {
                        "type": "NUMBER"
                    },
                    "calcium": {
                        "type": "NUMBER"
                    },
                    "iron": {
                        "type": "NUMBER"
                    },
                    "vitamin_c": {
                        "type": "NUMBER"
                    },
                    "folate": {
                        "type": "NUMBER"
                    },
                },
                "required": [
                    "index",
                    *NUTRITION_COLUMNS
                ],
            },
        }
    },
    "required": [
        "items"
    ],
}


# ============================================================
# ONE GEMINI REQUEST FOR MANY RECIPES
# ============================================================

def request_batch(
    recipes,
    retries=5
):

    client = get_client()

    prompt = build_prompt(
        recipes
    )

    last_error = None

    for attempt in range(
        1,
        retries + 1
    ):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SCHEMA,
                ),
            )

            raw = response.text or ""

            if not raw:
                raise RuntimeError(
                    "Gemini returned empty response."
                )

            data = json.loads(
                raw
            )

            items = data.get(
                "items",
                []
            )

            if len(items) != len(
                recipes
            ):
                raise RuntimeError(
                    f"Gemini returned "
                    f"{len(items)} results for "
                    f"{len(recipes)} recipes."
                )

            return items

        except Exception as exc:

            last_error = exc

            print(
                f"  Gemini batch attempt "
                f"{attempt}/{retries} failed:"
            )
            print(
                f"  {exc}"
            )

            if attempt < retries:

                # Longer waits for rate limits.
                wait = min(
                    60,
                    5 * attempt
                )

                print(
                    f"  Waiting {wait}s..."
                )

                time.sleep(
                    wait
                )

    raise RuntimeError(
        f"Gemini batch failed after "
        f"{retries} attempts: "
        f"{last_error}"
    )


# ============================================================
# SAVE
# ============================================================

def save_checkpoint(
    df
):

    df.to_csv(
        CHECKPOINT_PATH,
        index=False,
        encoding="utf-8-sig",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all missing recipes",
    )

    args = parser.parse_args()

    print(
        "=" * 70
    )
    print(
        "MEALORA GEMINI BATCHED NUTRITION POPULATION"
    )
    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    key = os.environ.get(
        "GEMINI_API_KEY",
        ""
    ).strip()

    if not key:

        raise SystemExit(
            "GEMINI_API_KEY was not found in backend/.env"
        )

    print(
        "API key found: YES"
    )

    print(
        f"Gemini model: {MODEL}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Workers: {args.workers}"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    if not MASTER_PATH.exists():

        raise SystemExit(
            f"Dataset not found:\n{MASTER_PATH}"
        )

    if CHECKPOINT_PATH.exists():

        print(
            "\nExisting checkpoint found."
        )

        print(
            "Resuming from checkpoint..."
        )

        df = pd.read_csv(
            CHECKPOINT_PATH,
            low_memory=False,
        )

    else:

        print(
            "\nLoading master dataset..."
        )

        df = pd.read_csv(
            MASTER_PATH,
            low_memory=False,
        )

    print(
        f"Recipes loaded: {len(df)}"
    )

    # --------------------------------------------------------
    # ENSURE COLUMNS
    # --------------------------------------------------------

    for column in NUTRITION_COLUMNS:

        if column not in df.columns:

            df[column] = pd.NA

    if "nutrition_source" not in df.columns:

        df["nutrition_source"] = pd.NA

    name_col = find_column(
        df,
        [
            "recipe_name",
            "name",
            "recipe",
            "title",
            "final_food_name",
        ],
    )

    ingredient_col = find_column(
        df,
        [
            "ingredients",
            "ingredient_list",
            "ingredient_names",
            "Cleaned-Ingredients",
            "cleaned_ingredients",
            "TranslatedIngredients",
        ],
    )

    if not name_col:

        raise SystemExit(
            "Recipe name column not found."
        )

    print(
        f"Recipe name column: {name_col}"
    )

    print(
        f"Ingredient column: {ingredient_col}"
    )

    # --------------------------------------------------------
    # BACKUP
    # --------------------------------------------------------

    if not BACKUP_PATH.exists():

        shutil.copy2(
            MASTER_PATH,
            BACKUP_PATH
        )

        print(
            f"Backup created:\n{BACKUP_PATH}"
        )

    # --------------------------------------------------------
    # FIND MISSING
    # --------------------------------------------------------

    missing_indices = []

    for index, row in df.iterrows():

        needs = any(
            missing(
                row[column]
            )
            for column in NUTRITION_COLUMNS
        )

        if needs:

            missing_indices.append(
                index
            )

    if args.limit > 0:

        missing_indices = (
            missing_indices[
                :args.limit
            ]
        )

    print()
    print(
        f"Recipes needing nutrition: "
        f"{len(missing_indices)}"
    )

    if not missing_indices:

        print(
            "Everything already has nutrition."
        )

        return

    # --------------------------------------------------------
    # BUILD BATCHES
    # --------------------------------------------------------

    batches = []

    for start in range(
        0,
        len(missing_indices),
        args.batch_size
    ):

        indices = missing_indices[
            start:
            start + args.batch_size
        ]

        recipes = []

        for index in indices:

            row = df.loc[index]

            ingredients = ""

            if ingredient_col:

                ingredients = clean(
                    row.get(
                        ingredient_col,
                        ""
                    )
                )

            recipes.append(
                {
                    "index": int(index),
                    "recipe_name": clean(
                        row.get(
                            name_col,
                            ""
                        ),
                        500
                    ),
                    "ingredients": ingredients,
                }
            )

        batches.append(
            recipes
        )

    print(
        f"Total Gemini requests required: "
        f"{len(batches)}"
    )

    print()

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    completed = 0
    failed = 0

    # With workers=1, requests happen sequentially.
    # This is intentionally safer for the API.
    for batch_number, recipes in enumerate(
        batches,
        start=1
    ):

        print(
            "=" * 70
        )

        print(
            f"BATCH {batch_number}/{len(batches)}"
        )

        print(
            f"Recipes in this request: "
            f"{len(recipes)}"
        )

        print(
            f"Progress before batch: "
            f"{completed}/{len(missing_indices)}"
        )

        print(
            "=" * 70
        )

        try:

            results = request_batch(
                recipes
            )

            by_index = {
                int(item["index"]): item
                for item in results
            }

            for recipe in recipes:

                index = recipe["index"]

                item = by_index.get(
                    index
                )

                if not item:

                    raise RuntimeError(
                        f"Missing result for "
                        f"recipe index {index}"
                    )

                for column in NUTRITION_COLUMNS:

                    value = item.get(
                        column
                    )

                    try:

                        value = float(
                            value
                        )

                    except Exception:

                        raise RuntimeError(
                            f"Invalid {column} "
                            f"for recipe "
                            f"{recipe['recipe_name']}"
                        )

                    if value < 0:

                        value = 0.0

                    df.at[
                        index,
                        column
                    ] = value

                df.at[
                    index,
                    "nutrition_source"
                ] = (
                    "Gemini estimated "
                    "per serving"
                )

                completed += 1

                print(
                    f"  OK: "
                    f"{recipe['recipe_name']}"
                )

            # Save after EVERY successful request.
            save_checkpoint(
                df
            )

            print()
            print(
                f"CHECKPOINT SAVED"
            )

            print(
                f"Completed: "
                f"{completed}/"
                f"{len(missing_indices)}"
            )

        except Exception as exc:

            failed += len(
                recipes
            )

            print()
            print(
                "BATCH FAILED:"
            )

            print(
                exc
            )

            # Save whatever has already completed.
            save_checkpoint(
                df
            )

            print()
            print(
                "Checkpoint saved."
            )

            print(
                "Stopping safely."
            )

            print(
                "Run the same command again "
                "after the API rate limit clears."
            )

            return

    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "ALL BATCHES COMPLETE"
    )

    print(
        "=" * 70
    )

    save_checkpoint(
        df
    )

    df.to_csv(
        MASTER_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    complete = df[
        NUTRITION_COLUMNS
    ].notna().all(
        axis=1
    )

    count = int(
        complete.sum()
    )

    coverage = (
        count / len(df) * 100
    )

    print()
    print(
        f"Nutrition complete: "
        f"{count}/{len(df)}"
    )

    print(
        f"Nutrition coverage: "
        f"{coverage:.2f}%"
    )

    print()
    print(
        f"Updated dataset:\n"
        f"{MASTER_PATH}"
    )

    print()
    print(
        f"Checkpoint:\n"
        f"{CHECKPOINT_PATH}"
    )

    print()
    print(
        "DONE."
    )


if __name__ == "__main__":
    main()
