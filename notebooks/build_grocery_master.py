from pathlib import Path
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

BIGBASKET_FILE = RAW_DIR / "BigBasket.csv"
UPC_FILE = RAW_DIR / "upc_corpus 2.csv"
OFF_FILE = RAW_DIR / "en.openfoodfacts.org.products.tsv"

OUTPUT_FILE = (
    PROCESSED_DIR / "master_grocery_vocabulary.csv"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)

    return value


def normalize_product_name(name):
    name = clean_text(name)

    name = re.sub(
        r"\b\d+(?:\.\d+)?\s*(kg|kgs|kilo|kilos|kilogram|kilograms|"
        r"g|gm|gram|grams|l|ltr|liter|liters|litre|litres|ml|"
        r"pc|pcs|piece|pieces|pack|packs|box|boxes|bag|bags|"
        r"bottle|bottles)\b",
        " ",
        name,
        flags=re.IGNORECASE,
    )

    name = re.sub(
        r"\b\d+\s*x\s*\d+(?:\.\d+)?\s*(kg|g|ml|l|ltr)\b",
        " ",
        name,
        flags=re.IGNORECASE,
    )

    name = re.sub(
        r"https?://\S+",
        " ",
        name,
    )

    name = re.sub(
        r"[^a-z0-9\s&'-]",
        " ",
        name,
    )

    singular_map = {
        "potatoes": "potato",
        "tomatoes": "tomato",
        "onions": "onion",
        "carrots": "carrot",
        "apples": "apple",
        "bananas": "banana",
        "oranges": "orange",
        "mangoes": "mango",
        "eggs": "egg",
        "cucumbers": "cucumber",
        "capsicums": "capsicum",
        "peppers": "pepper",
        "loaves": "loaf",
        "boxes": "box",
        "bags": "bag",
        "bottles": "bottle",
        "jars": "jar",
        "cans": "can",
        "packs": "pack",
    }

    name = " ".join(
        singular_map.get(
            word,
            word,
        )
        for word in name.split()
    )

    return name.strip()


# ============================================================
# FOOD / NON-FOOD TERMS
# ============================================================

NON_FOOD_TERMS = [
    "detergent",
    "washing powder",
    "soap",
    "shampoo",
    "conditioner",
    "toothpaste",
    "toothbrush",
    "sanitary",
    "diaper",
    "battery",
    "screw",
    "bolt",
    "backpack",
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

FOOD_TERMS = [
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
    "apple",
    "banana",
    "orange",
    "mango",
    "milk",
    "curd",
    "yogurt",
    "yoghurt",
    "paneer",
    "cheese",
    "butter",
    "egg",
    "chicken",
    "fish",
    "meat",
    "bread",
    "pasta",
    "noodle",
    "cereal",
    "oat",
    "sugar",
    "salt",
    "oil",
    "ghee",
    "spice",
    "chilli",
    "pepper",
    "turmeric",
    "cumin",
    "coriander",
    "biscuit",
    "cookie",
    "snack",
    "juice",
    "sauce",
]


def classify_series(
    product_series,
    category_series=None,
    subcategory_series=None,
):
    """
    Vectorized food/non-food classification.
    """

    combined = (
        product_series.fillna("").astype(str)
        + " "
    )

    if category_series is not None:
        combined += (
            category_series.fillna("").astype(str)
            + " "
        )

    if subcategory_series is not None:
        combined += (
            subcategory_series.fillna("").astype(str)
            + " "
        )

    combined = (
        combined
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    result = pd.Series(
        "unknown",
        index=combined.index,
        dtype="object",
    )

    for term in NON_FOOD_TERMS:
        mask = combined.str.contains(
            re.escape(term),
            regex=True,
            na=False,
        )

        result.loc[mask] = "non_food"

    for term in FOOD_TERMS:
        mask = combined.str.contains(
            re.escape(term),
            regex=True,
            na=False,
        )

        result.loc[
            mask & (result == "unknown")
        ] = "food"

    return result


# ============================================================
# BIGBASKET
# ============================================================

def load_bigbasket():
    print("\nLoading BigBasket...")

    df = pd.read_csv(
        BIGBASKET_FILE,
        on_bad_lines="skip",
        low_memory=False,
    )

    df = df[
        [
            "ProductName",
            "Brand",
            "Quantity",
            "Category",
            "SubCategory",
        ]
    ].copy()

    df["product_name"] = (
        df["ProductName"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["brand"] = (
        df["Brand"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["category"] = (
        df["Category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["subcategory"] = (
        df["SubCategory"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["normalized_name"] = (
        df["product_name"]
        .map(normalize_product_name)
    )

    df["food_or_non_food"] = classify_series(
        df["product_name"],
        df["category"],
        df["subcategory"],
    )

    df["source"] = "bigbasket"

    return df[
        [
            "product_name",
            "normalized_name",
            "brand",
            "category",
            "subcategory",
            "food_or_non_food",
            "source",
        ]
    ]


# ============================================================
# UPC
# ============================================================

def load_upc():
    print("\nLoading UPC database...")

    df = pd.read_csv(
        UPC_FILE,
        on_bad_lines="skip",
        low_memory=False,
    )

    df = df[
        [
            "ean",
            "name",
        ]
    ].copy()

    df["product_name"] = (
        df["name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["brand"] = ""
    df["category"] = ""
    df["subcategory"] = ""

    df["normalized_name"] = (
        df["product_name"]
        .map(normalize_product_name)
    )

    df["food_or_non_food"] = classify_series(
        df["product_name"]
    )

    df["source"] = "upc"

    return df[
        [
            "product_name",
            "normalized_name",
            "brand",
            "category",
            "subcategory",
            "food_or_non_food",
            "source",
        ]
    ]


# ============================================================
# OPEN FOOD FACTS
# ============================================================

def load_open_food_facts():
    print("\nLoading Open Food Facts in chunks...")

    selected_columns = [
        "product_name",
        "generic_name",
        "brands",
        "categories_en",
        "main_category_en",
    ]

    output_file = (
        PROCESSED_DIR
        / "openfoodfacts_cleaned.csv"
    )

    # Remove an incomplete previous output.
    if output_file.exists():
        output_file.unlink()

    first_chunk = True
    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            OFF_FILE,
            sep="\t",
            usecols=selected_columns,
            dtype=str,
            chunksize=100_000,
            low_memory=False,
            on_bad_lines="skip",
        ),
        start=1,
    ):
        print(
            f"  Processing chunk {chunk_number}..."
        )

        chunk = chunk.fillna("")

        chunk["product_name"] = (
            chunk["product_name"]
            .str.strip()
        )

        chunk = chunk[
            chunk["product_name"] != ""
        ].copy()

        if chunk.empty:
            continue

        chunk["category"] = (
            chunk["main_category_en"]
            .str.strip()
        )

        empty_category = (
            chunk["category"] == ""
        )

        chunk.loc[
            empty_category,
            "category",
        ] = (
            chunk.loc[
                empty_category,
                "categories_en",
            ]
            .str.split(",")
            .str[0]
            .fillna("")
            .str.strip()
        )

        chunk["subcategory"] = (
            chunk["generic_name"]
            .str.strip()
        )

        chunk["brand"] = (
            chunk["brands"]
            .str.split(",")
            .str[0]
            .fillna("")
            .str.strip()
        )

        chunk["normalized_name"] = (
            chunk["product_name"]
            .map(normalize_product_name)
        )

        chunk["food_or_non_food"] = classify_series(
            chunk["product_name"],
            chunk["category"],
            chunk["subcategory"],
        )

        chunk["source"] = (
            "openfoodfacts"
        )

        cleaned = chunk[
            [
                "product_name",
                "normalized_name",
                "brand",
                "category",
                "subcategory",
                "food_or_non_food",
                "source",
            ]
        ]

        cleaned.to_csv(
            output_file,
            mode="a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False
        total_rows += len(cleaned)

    print(
        f"\nOpen Food Facts cleaned rows: {total_rows:,}"
    )

    return output_file


# ============================================================
# BUILD MASTER DATASET
# ============================================================

def build_master():
    bigbasket = load_bigbasket()

    upc = load_upc()

    openfoodfacts_file = (
        load_open_food_facts()
    )

    print("\nCombining cleaned datasets...")

    openfoodfacts = pd.read_csv(
        openfoodfacts_file,
        dtype=str,
        low_memory=False,
    )

    master = pd.concat(
        [
            bigbasket,
            openfoodfacts,
            upc,
        ],
        ignore_index=True,
    )

    master = master.fillna("")

    master = master[
        master["normalized_name"]
        .str.strip()
        != ""
    ]

    master = master[
        master["normalized_name"]
        .str.len()
        >= 2
    ]

    master = master.drop_duplicates(
        subset=[
            "normalized_name",
            "brand",
        ],
        keep="first",
    )

    master = master.sort_values(
        by=[
            "normalized_name",
            "source",
        ]
    ).reset_index(
        drop=True
    )

    master.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n======================================"
    )

    print(
        "MASTER DATASET CREATED"
    )

    print(
        "======================================"
    )

    print(
        f"\nOutput:"
        f"\n{OUTPUT_FILE}"
    )

    print(
        f"\nTotal rows: "
        f"{len(master):,}"
    )

    print(
        "\nFood status:"
    )

    print(
        master[
            "food_or_non_food"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nSources:"
    )

    print(
        master[
            "source"
        ].value_counts()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    build_master()