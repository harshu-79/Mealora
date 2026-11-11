import os
import re
import ast
import json
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# FILE PATHS
# ============================================================

BASE = "datasets/processed"

RECIPES = os.path.join(
    BASE,
    "recipe_master_dataset.csv"
)

NUTRITION = os.path.join(
    BASE,
    "recipe_master_dataset_nutrition_CLEAN.csv"
)

FEATURES = os.path.join(
    BASE,
    "recipe_feature_matrix.npz"
)

PANTRY = os.path.join(
    BASE,
    "recipe_pantry_features.csv"
)

EXPIRY = os.path.join(
    BASE,
    "recipe_expiry_features.csv"
)

INTERACTIONS = os.path.join(
    BASE,
    "user_recipe_interactions.csv"
)

RF_MODEL_CANDIDATES = [
    os.path.join(BASE, "mealora_rf_model.joblib"),
    os.path.join(BASE, "mealora_ml_models", "mealora_selected_ml_model.joblib"),
]

HAN_SCORES = os.path.join(
    BASE,
    "han_model",
    "han_recipe_scores.csv"
)


# ============================================================
# RANDOM FOREST FEATURES
# ============================================================

RF_FEATURES = [
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


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

print("\nLoading Mealora ML model...")

RF_FEATURES = [
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

try:
    RF_MODEL = next((path for path in RF_MODEL_CANDIDATES if os.path.exists(path)), RF_MODEL_CANDIDATES[0])
    saved_object = joblib.load(RF_MODEL)
except Exception as exc:
    saved_object = None
    print("RF model could not be loaded at import time:", exc)

if isinstance(saved_object, dict):
    rf_model = saved_object.get("model")
    rf_features = saved_object.get("features", RF_FEATURES)
    rf_model_name = saved_object.get("model_name", "Random Forest")
else:
    rf_model = saved_object
    rf_features = RF_FEATURES
    rf_model_name = "Random Forest"

if rf_model is not None:
    print("Loaded selected ML model:", rf_model_name)
    print("RF ranking features:", rf_features)
else:
    print("WARNING: Random Forest model is unavailable. Candidate ranking will use base score until the model is trained.")


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

# ============================================================
# AUTOMATIC CONTEXT HELPERS
# ============================================================

def _norm_text(value):
    return str(value or "").strip().lower()


def _keyword_score(df, keywords):
    """
    Returns a soft 0..1 score based on recipe text.
    Used for weather/festival signals without changing the
    trained 10-feature HGB input.
    """
    text_cols = [
        c for c in [
            "recipe_name", "ingredients", "ingredient_names",
            "instructions", "cuisine", "diet", "meal_type"
        ]
        if c in df.columns
    ]

    if not text_cols:
        return pd.Series(0.0, index=df.index)

    combined = (
        df[text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )

    hits = np.zeros(len(df), dtype=float)
    for keyword, weight in keywords:
        hits += combined.str.contains(
            re.escape(keyword.lower()), regex=True, na=False
        ).astype(float).to_numpy() * float(weight)

    return pd.Series(np.clip(hits, 0.0, 1.0), index=df.index)


def _automatic_context_scores(df, weather=None, festival=None, location=None):
    weather = weather or {}
    festival = festival or {}
    location = location or {}

    condition = _norm_text(weather.get("condition"))
    temperature = weather.get("temperature")
    festival_name = _norm_text(festival.get("name"))

    weather_score = pd.Series(0.0, index=df.index)

    # Warm/rainy/cold weather food affinity.
    if "rain" in condition or "drizzle" in condition or "thunder" in condition:
        weather_score = _keyword_score(df, [
            ("rasam", 1.0), ("soup", 1.0), ("sambar", 0.8),
            ("khichdi", 0.9), ("tea", 0.6), ("pakora", 0.7),
            ("curry", 0.5), ("stew", 0.8)
        ])
    elif "snow" in condition or "cold" in condition:
        weather_score = _keyword_score(df, [
            ("soup", 1.0), ("stew", 1.0), ("rasam", 0.8),
            ("khichdi", 0.9), ("curry", 0.7)
        ])
    else:
        try:
            temp = float(temperature)
        except (TypeError, ValueError):
            temp = None

        if temp is not None and temp >= 32:
            weather_score = _keyword_score(df, [
                ("salad", 1.0), ("raita", 0.9), ("buttermilk", 0.9),
                ("lassi", 0.9), ("lemon", 0.7), ("coconut", 0.6),
                ("juice", 0.8)
            ])
        elif temp is not None and temp <= 20:
            weather_score = _keyword_score(df, [
                ("soup", 1.0), ("rasam", 0.8), ("khichdi", 0.9),
                ("stew", 0.9), ("curry", 0.7)
            ])
        elif temp is not None and 20 < temp < 32:
            # Neutral/pleasant weather still participates as a moderate signal.
            weather_score = pd.Series(0.5, index=df.index)

    festival_score = pd.Series(0.0, index=df.index)

    festival_keywords = {
        "diwali": [
            ("laddu", 1.0), ("ladoo", 1.0), ("halwa", 0.9),
            ("barfi", 1.0), ("kheer", 0.9), ("gulab jamun", 0.9),
            ("jalebi", 0.9), ("sweet", 0.7), ("chakli", 0.7),
            ("murukku", 0.7), ("mixture", 0.6)
        ],
        "ganesh": [
            ("modak", 1.0), ("kozhukattai", 1.0),
            ("laddu", 0.9), ("ladoo", 0.9), ("sweet", 0.6)
        ],
        "pongal": [
            ("pongal", 1.0), ("sweet pongal", 1.0),
            ("ven pongal", 1.0), ("sakkarai pongal", 1.0),
            ("sambar", 0.5), ("coconut", 0.4)
        ],
        "makar sankranti": [
            ("til", 0.9), ("sesame", 0.9), ("jaggery", 0.9),
            ("pongal", 0.9), ("sweet", 0.7)
        ],
        "onam": [
            ("sadya", 1.0), ("avial", 1.0), ("payasam", 0.9),
            ("thoran", 0.9), ("olan", 0.9), ("parippu", 0.7),
            ("kalan", 0.8), ("pachadi", 0.8)
        ],
        "raksha": [
            ("laddu", 0.9), ("ladoo", 0.9), ("kheer", 0.8),
            ("barfi", 1.0), ("sweet", 0.7)
        ],
        "holi": [
            ("gujiya", 1.0), ("thandai", 1.0), ("malpua", 0.9),
            ("dahi vada", 0.8), ("sweet", 0.7)
        ],
        "janmashtami": [
            ("makhan", 1.0), ("mishri", 1.0), ("poha", 0.7),
            ("sabudana", 0.9), ("kuttu", 0.9), ("singhara", 0.9),
            ("panjiri", 1.0), ("peda", 0.9), ("ladoo", 0.8),
            ("laddoo", 0.8), ("kheer", 0.8), ("sweet", 0.6)
        ],
        "krishna": [
            ("makhan", 1.0), ("mishri", 1.0), ("peda", 0.9),
            ("panjiri", 1.0), ("ladoo", 0.8), ("kheer", 0.8)
        ],
        "navratri": [
            ("sabudana", 1.0), ("kuttu", 0.9), ("singhara", 0.9),
            ("samak", 0.9), ("vrat", 1.0), ("rajgira", 0.9),
            ("farali", 1.0)
        ],
        "shivaratri": [
            ("sabudana", 0.9), ("kuttu", 0.9), ("singhara", 0.9),
            ("rajgira", 0.9), ("vrat", 1.0), ("fruit", 0.6)
        ],
        "ugadi": [
            ("pulihora", 0.9), ("pachadi", 1.0), ("bobbatlu", 1.0),
            ("obattu", 1.0), ("holige", 1.0), ("payasam", 0.8)
        ],
        "gudi padwa": [
            ("puran poli", 1.0), ("shrikhand", 0.9), ("poori", 0.6),
            ("sweet", 0.6)
        ],
        "eid": [
            ("biryani", 0.8), ("sheer khurma", 1.0),
            ("korma", 0.9), ("sewai", 1.0), ("kebab", 0.7),
            ("haleem", 0.9), ("nihari", 0.8)
        ],
        "ramzan": [
            ("biryani", 0.8), ("haleem", 1.0), ("kebab", 0.8),
            ("sewai", 0.9), ("samosa", 0.6)
        ],
        "rath yatra": [
            ("khichdi", 0.8), ("mahaprasad", 1.0),
            ("dalma", 1.0), ("sweet", 0.6)
        ],
        "guru purnima": [
            ("kheer", 0.8), ("halwa", 0.8), ("sweet", 0.7)
        ],

        "dussehra": [
            ("poori", 0.6), ("halwa", 0.8), ("sweet", 0.7),
            ("kheer", 0.7)
        ],
        "durga": [
            ("khichdi", 0.8), ("labra", 1.0), ("payesh", 1.0),
            ("luchi", 0.9), ("sandesh", 1.0), ("sweet", 0.7)
        ],
        "navami": [
            ("puri", 0.6), ("halwa", 0.8), ("chana", 0.8),
            ("sweet", 0.6)
        ],
        "chhath": [
            ("thekua", 1.0), ("kheer", 0.8), ("rasiyaw", 1.0),
            ("rice", 0.5), ("jaggery", 0.7)
        ],
        "guru nanak": [
            ("langar", 1.0), ("dal", 0.8), ("roti", 0.7),
            ("halwa", 0.8), ("kheer", 0.7)
        ],
        "buddha": [
            ("kheer", 0.9), ("payasam", 0.8), ("sweet", 0.6)
        ],
        "ravi": [
            ("puran poli", 0.8), ("sweet", 0.6)
        ],
        "baisakhi": [
            ("makki", 0.8), ("sarson", 0.9), ("lassi", 0.7),
            ("paratha", 0.7)
        ],
        "lohri": [
            ("til", 0.9), ("sesame", 0.9), ("jaggery", 1.0),
            ("gajak", 1.0), ("rewri", 1.0), ("makki", 0.7)
        ],
        "maha": [
            ("sabudana", 0.8), ("vrat", 0.9), ("kuttu", 0.8),
            ("singhara", 0.8)
        ],
    }

    for festival_key, keywords in festival_keywords.items():
        if festival_key in festival_name:
            festival_score = _keyword_score(df, keywords)
            break

    # Location-aware scoring. City-specific regional terms are stronger
    # than a broad cuisine label: Hyderabad -> Telangana/Andhra/Telugu/
    # Hyderabadi, Chennai -> Tamil Nadu/Tamil/Chettinad, etc.
    location_score = pd.Series(0.0, index=df.index)

    city = _norm_text(
        location.get("city")
        or location.get("location_city")
        or location.get("name")
        or location.get("location_name")
    )

    location_cuisine = _norm_text(location.get("cuisine_hint"))

    city_region_terms = {
        "hyderabad": ["telangana", "andhra pradesh", "andhra", "telugu", "hyderabadi", "nizami", "deccan", "hyderabad"],
        "vijayawada": ["andhra pradesh", "andhra", "telugu", "vijayawada"],
        "visakhapatnam": ["andhra pradesh", "andhra", "telugu", "vizag", "visakhapatnam"],
        "guntur": ["andhra pradesh", "andhra", "telugu", "guntur"],
        "chennai": ["tamil nadu", "tamil", "chettinad", "chennai"],
        "coimbatore": ["tamil nadu", "tamil", "kongu", "coimbatore"],
        "madurai": ["tamil nadu", "tamil", "madurai"],
        "trichy": ["tamil nadu", "tamil", "trichy", "tiruchirappalli"],
        "salem": ["tamil nadu", "tamil", "salem"],
        "bengaluru": ["karnataka", "kannada", "udupi", "bengaluru", "bangalore"],
        "bangalore": ["karnataka", "kannada", "udupi", "bengaluru", "bangalore"],
        "mysore": ["karnataka", "kannada", "mysore", "mysuru"],
        "mysuru": ["karnataka", "kannada", "mysore", "mysuru"],
        "kochi": ["kerala", "malabar", "malayali", "kochi"],
        "thiruvananthapuram": ["kerala", "malayali", "thiruvananthapuram", "trivandrum"],
        "trivandrum": ["kerala", "malayali", "thiruvananthapuram", "trivandrum"],
        "mumbai": ["maharashtra", "marathi", "konkani", "mumbai"],
        "pune": ["maharashtra", "marathi", "pune"],
        "kolkata": ["west bengal", "bengali", "kolkata"],
        "jaipur": ["rajasthan", "rajasthani", "jaipur"],
        "udaipur": ["rajasthan", "rajasthani", "udaipur"],
        "lucknow": ["uttar pradesh", "awadhi", "lucknow"],
        "delhi": ["delhi", "delhi ncr", "punjabi", "mughlai", "north indian"],
        "new delhi": ["delhi", "delhi ncr", "punjabi", "mughlai", "north indian"],
    }

    regional_columns = [
        c for c in [
            "state", "state_name", "region", "regional",
            "regional_cuisine", "food_region", "cuisine", "recipe_name"
        ] if c in df.columns
    ]

    if city in city_region_terms and regional_columns:
        regional_text = (
            df[regional_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )
        hits = np.zeros(len(df), dtype=float)
        for term in city_region_terms[city]:
            hits += regional_text.str.contains(term, regex=False, na=False).astype(float).to_numpy()
        location_score = pd.Series(np.clip(hits, 0.0, 1.0), index=df.index)
    elif location_cuisine:
        location_score = _keyword_score(df, [(location_cuisine, 1.0)])

    return weather_score, festival_score, location_score




# ============================================================
# LIVE PANTRY + EXPIRY SUPPORT
# ============================================================
# The recommendation engine receives the CURRENT Supabase pantry
# from the frontend. The precomputed pantry/expiry CSVs remain as
# fallbacks, but they are not allowed to override the live pantry.

_LIVE_INGREDIENT_ALIASES = {
    "tomatoes": "tomato",
    "onions": "onion",
    "potatoes": "potato",
    "carrots": "carrot",
    "capsicums": "capsicum",
    "chillies": "chilli",
    "chilies": "chilli",
    "eggs": "egg",
    "prawns": "prawn",
    "shrimps": "shrimp",
    "yogurt": "curd",
    "yoghurt": "curd",
    "dahi": "curd",
    "cilantro": "coriander leaves",
    "coriander leaf": "coriander leaves",
    "curry leaf": "curry leaves",
    "curry leaves": "curry leaves",
}

_LIVE_UNITS = (
    "kilograms", "kilogram", "kgs", "kg", "grams", "gram", "g",
    "litres", "litre", "liter", "liters", "l", "millilitres",
    "millilitre", "milliliters", "milliliter", "ml", "tablespoons",
    "tablespoon", "tbsp", "teaspoons", "teaspoon", "tsp", "cups",
    "cup", "pieces", "piece", "dozen"
)


def _live_clean_ingredient(value):
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return ""

    text = re.sub(r"\([^)]*\)", " ", text)
    units = "|".join(re.escape(x) for x in _LIVE_UNITS)
    text = re.sub(
        rf"^\s*\d+(?:\.\d+)?\s*(?:{units})?\s*(?:of\s+)?",
        "",
        text,
    )
    text = re.sub(
        r"\b(chopped|sliced|diced|minced|grated|crushed|ground|"
        r"finely|roughly|freshly|to taste|as needed)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text in _LIVE_INGREDIENT_ALIASES:
        text = _LIVE_INGREDIENT_ALIASES[text]

    # Simple plural normalization without damaging multi-word names.
    if text.endswith("ies") and len(text) > 4:
        text = text[:-3] + "y"
    elif text.endswith("s") and len(text) > 3 and not text.endswith("ss"):
        text = text[:-1]

    return _LIVE_INGREDIENT_ALIASES.get(text, text)


def _live_parse_recipe_ingredients(value):
    if isinstance(value, list):
        return [x for x in (_live_clean_ingredient(v) for v in value) if x]

    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [x for x in (_live_clean_ingredient(v) for v in parsed) if x]
            except Exception:
                pass

    parts = re.split(r"\n|;|\|", text)
    if len(parts) == 1:
        parts = [x.strip() for x in text.split(",")]

    return [x for x in (_live_clean_ingredient(v) for v in parts) if x]


def _live_ingredient_match(recipe_ingredient, pantry_name):
    a = _live_clean_ingredient(recipe_ingredient)
    b = _live_clean_ingredient(pantry_name)
    if not a or not b:
        return False
    if a == b:
        return True
    # Allow specific pantry names to satisfy a more detailed recipe name,
    # e.g. "basmati rice" -> "rice", but avoid one-character matches.
    if len(b) >= 4 and (b in a or a in b):
        return True
    return False


def _apply_live_pantry(df, pantry_items):
    """Apply CURRENT pantry quickly; avoid nested recipe x pantry loops."""
    if not pantry_items:
        return False

    ingredient_col = next((c for c in [
        "ingredient_names", "ingredients", "ingredient_list",
        "Cleaned-Ingredients", "TranslatedIngredients"
    ] if c in df.columns), None)

    if not ingredient_col:
        print("LIVE pantry warning: no recipe ingredient column found; using precomputed pantry features.")
        return False

    from datetime import date, datetime as _dt
    today = date.today()
    pantry_names = []
    expiry_weights = {}

    for item in pantry_items:
        if not isinstance(item, dict):
            continue
        name = item.get("normalized_name") or item.get("ingredient_name") or item.get("name")
        cleaned = _live_clean_ingredient(name)
        if not cleaned:
            continue
        try:
            q = item.get("quantity")
            if q is not None and str(q).strip() and float(q) <= 0:
                continue
        except (TypeError, ValueError):
            pass
        if cleaned not in pantry_names:
            pantry_names.append(cleaned)
        exp = item.get("expiry_date")
        if exp:
            try:
                days = (_dt.strptime(str(exp)[:10], "%Y-%m-%d").date() - today).days
                weight = 0.0 if days < 0 else 1.0 if days <= 3 else 0.85 if days <= 7 else 0.60 if days <= 14 else 0.30 if days <= 30 else 0.0
                expiry_weights[cleaned] = max(expiry_weights.get(cleaned, 0.0), weight)
            except (TypeError, ValueError):
                pass

    if not pantry_names:
        df["pantry_match_pct"] = 0.0
        df["missing_ingredient_count"] = 0.0
        df["pantry_score"] = 0.0
        df["expiry_score"] = 0.0
        return True

    # One compiled regex replaces the previous recipe-ingredient x pantry nested loop.
    # Longest terms first prevents a generic term from stealing a more specific match.
    pattern = re.compile(
        "|".join(re.escape(x) for x in sorted(pantry_names, key=len, reverse=True)),
        flags=re.IGNORECASE,
    )

    # Parse each unique recipe ingredient string only once.
    parse_cache = {}
    pantry_scores = np.zeros(len(df), dtype=float)
    missing_counts = np.zeros(len(df), dtype=float)
    expiry_scores = np.zeros(len(df), dtype=float)

    values = df[ingredient_col].fillna("").astype(str).tolist()
    for i, raw in enumerate(values):
        ingredients = parse_cache.get(raw)
        if ingredients is None:
            ingredients = _live_parse_recipe_ingredients(raw)
            parse_cache[raw] = ingredients
        if not ingredients:
            continue

        match_count = 0
        best_expiry = 0.0
        for ingredient in ingredients:
            m = pattern.search(ingredient)
            if m:
                matched = _live_clean_ingredient(m.group(0))
                match_count += 1
                best_expiry = max(best_expiry, expiry_weights.get(matched, 0.0))

        pantry_scores[i] = match_count / len(ingredients)
        missing_counts[i] = len(ingredients) - match_count
        expiry_scores[i] = best_expiry

    df["pantry_match_pct"] = pantry_scores
    df["missing_ingredient_count"] = missing_counts
    df["pantry_score"] = np.clip(pantry_scores, 0, 1)
    df["expiry_score"] = np.clip(expiry_scores, 0, 1)

    print(
        f"LIVE pantry applied: {len(pantry_names)} available items; "
        f"best pantry match={df['pantry_score'].max():.0%}; "
        f"recipes with pantry match={(df['pantry_score'] > 0).sum()}"
    )
    return True


def recommend(
    reference_recipe="Balu shahi",
    target_user=1,
    top_k=5,
    meal_type="Dinner",
    diet="Vegetarian",
    cuisine="South Indian",
    max_time=30,
    weather=None,
    festival=None,
    location=None,
    pantry_items=None,
):

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        RECIPES,
        low_memory=False
    )

    X = load_npz(FEATURES)

    pantry = pd.read_csv(PANTRY)

    expiry = pd.read_csv(EXPIRY)

    print(
        "Recipes loaded:",
        len(df)
    )

    # --------------------------------------------------------
    # MERGE HAN GRAPH SCORES
    # --------------------------------------------------------
    original_recipe_count = len(df)
    if os.path.exists(HAN_SCORES):
        print("\nMerging HAN graph scores...")
        han = pd.read_csv(HAN_SCORES, low_memory=False)
        if "recipe_id" in df.columns and "recipe_id" in han.columns:
            han_small = han[["recipe_id", "han_score"]].drop_duplicates("recipe_id")
            df = df.merge(han_small, on="recipe_id", how="left", validate="one_to_one")
        elif "recipe_name" in df.columns and "recipe_name" in han.columns:
            han_small = han[["recipe_name", "han_score"]].drop_duplicates("recipe_name")
            df = df.merge(han_small, on="recipe_name", how="left", validate="many_to_one")
        else:
            df["han_score"] = 0.5
        if len(df) != original_recipe_count:
            raise RuntimeError("HAN merge changed the recipe count. Duplicate recipe rows were created.")
        df["han_score"] = pd.to_numeric(df["han_score"], errors="coerce").fillna(0.5).clip(0, 1)
        print("HAN scores merged:", len(han))
    else:
        df["han_score"] = 0.5
        print("WARNING: HAN score file not found. Using neutral HAN score 0.5.")

    # --------------------------------------------------------
    # MERGE PANTRY FEATURES
    # --------------------------------------------------------

    print("\nMerging pantry features...")

    pantry_columns = [
        "recipe_id",
        "pantry_match_pct",
        "missing_ingredient_count",
    ]

    available_pantry_columns = [
        col
        for col in pantry_columns
        if col in pantry.columns
    ]

    pantry_merge = pantry[
        available_pantry_columns
    ].copy()

    df = df.merge(
        pantry_merge,
        on="recipe_id",
        how="left"
    )

    print(
        "Pantry features merged."
    )

    # --------------------------------------------------------
    # FIX MISSING PANTRY VALUES
    # --------------------------------------------------------

    if "pantry_match_pct" not in df.columns:
        df["pantry_match_pct"] = 0.0

    if "missing_ingredient_count" not in df.columns:
        df["missing_ingredient_count"] = 0.0

    df["pantry_match_pct"] = pd.to_numeric(
        df["pantry_match_pct"],
        errors="coerce"
    ).fillna(0)

    df["missing_ingredient_count"] = pd.to_numeric(
        df["missing_ingredient_count"],
        errors="coerce"
    ).fillna(0)

    # --------------------------------------------------------
    # REFERENCE RECIPE
    # --------------------------------------------------------

    matches = df[
        df["recipe_name"]
        .fillna("")
        .str.contains(
            reference_recipe,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matches.empty:
        raise ValueError(
            f"Reference recipe not found: "
            f"{reference_recipe}"
        )

    ref_idx = matches.index[0]

    print(
        "Reference recipe:",
        df.loc[
            ref_idx,
            "recipe_name"
        ]
    )

    # ========================================================
    # 1. TF-IDF + COSINE SIMILARITY
    # ========================================================

    print(
        "\nCalculating content similarity..."
    )

    content_scores = cosine_similarity(
        X[ref_idx],
        X
    ).flatten()

    df["content_score"] = content_scores

    # ========================================================
    # 2. CONTENT-BASED RECOMMENDATION
    # ========================================================

    # Mealora is a single-user application.
    # No collaborative or multi-user score is used.
    df["personal_behavior_score"] = 0.0

    # Content similarity is the primary recipe similarity signal.
    df["hybrid_score"] = df["content_score"]

    # ========================================================
    # 4. PANTRY COMPATIBILITY
    # ========================================================

    print(
        "Calculating pantry compatibility..."
    )

    df["pantry_score"] = (
        df["pantry_match_pct"]
        .clip(
            lower=0,
            upper=1
        )
    )

    # ========================================================
    # 5. EXPIRY SCORE
    # ========================================================

    print("Calculating expiry score...")

    # IMPORTANT PERFORMANCE FIX:
    # When the frontend sends the CURRENT pantry, do not build/map the
    # 18k-row expiry CSV. Live pantry expiry is calculated directly below.
    # The CSV is used only as the fallback when no live pantry is supplied.
    if pantry_items:
        df["expiry_score"] = 0.0
        _live_pantry_used = _apply_live_pantry(df, pantry_items)
    else:
        expiry_map = (
            expiry.drop_duplicates("recipe_id")
            .set_index("recipe_id")["expiry_score"]
        )
        df["expiry_score"] = pd.to_numeric(
            df["recipe_id"].map(expiry_map),
            errors="coerce"
        ).fillna(0).clip(0, 1)
        _live_pantry_used = False

    # --------------------------------------------------------
    # AUTOMATIC REAL-WORLD CONTEXT SCORES
    # These columns MUST be created before context_score uses them.
    # --------------------------------------------------------
    (
        df["weather_score"],
        df["festival_score"],
        df["location_score"],
    ) = _automatic_context_scores(
        df,
        weather=weather,
        festival=festival,
        location=location,
    )

    # Ensure numeric, bounded scores.
    for _col in ["weather_score", "festival_score", "location_score"]:
        df[_col] = pd.to_numeric(
            df[_col], errors="coerce"
        ).fillna(0).clip(0, 1)

    # ========================================================
    # 6. DYNAMIC CONTEXT SCORE
    # ========================================================

    print("\nApplying user context...")

    context = {
        "meal_type": str(meal_type or "").strip(),
        "diet": str(diet or "").strip(),
        "cuisine": str(cuisine or "").strip(),
        "max_time": max_time,
    }

    # Normalize text safely
    meal_text = df["meal_type"].fillna("").astype(str).str.lower().str.strip()
    diet_column = "diet_type" if "diet_type" in df.columns else ("diet" if "diet" in df.columns else None)
    if diet_column is None:
        df["diet_type"] = ""
        diet_column = "diet_type"
    diet_text = df[diet_column].fillna("").astype(str).str.lower().str.strip()
    cuisine_text = df["cuisine"].fillna("").astype(str).str.lower().str.strip()

    requested_meal = context["meal_type"].lower()
    requested_diet = context["diet"].lower()
    requested_cuisine = context["cuisine"].lower()

    # --------------------------------------------------------
    # Meal type match
    # --------------------------------------------------------

    df["meal_match"] = (
        meal_text.str.contains(requested_meal, na=False, regex=False)
        if requested_meal
        else 1.0
    )
    df["meal_match"] = df["meal_match"].astype(float)

    # --------------------------------------------------------
    # Diet compatibility
    # --------------------------------------------------------

    # Use explicit recipe labels first.
    df["diet_match"] = diet_text.str.contains(
        requested_diet,
        na=False,
        regex=False
    ).astype(float) if requested_diet else 1.0

    # Stronger compatibility rules for common diet requests.
    if requested_diet in {"vegetarian", "vegan", "jain", "non-vegetarian"}:

        nonveg_pattern = (
            r"non.?vegetarian|nonveg|chicken|mutton|lamb|"
            r"beef|pork|fish|prawn|shrimp|crab|meat|egg"
        )

        dairy_pattern = r"milk|curd|yogurt|yoghurt|paneer|cheese|ghee|butter"

        onion_garlic_pattern = r"onion|garlic"

        if requested_diet == "vegetarian":
            df["diet_match"] = (
                ~diet_text.str.contains(
                    nonveg_pattern,
                    case=False,
                    na=False,
                    regex=True
                )
            ).astype(float)

        elif requested_diet == "vegan":
            df["diet_match"] = (
                ~(
                    diet_text.str.contains(
                        nonveg_pattern,
                        case=False,
                        na=False,
                        regex=True
                    )
                    |
                    diet_text.str.contains(
                        dairy_pattern,
                        case=False,
                        na=False,
                        regex=True
                    )
                )
            ).astype(float)

        elif requested_diet == "jain":
            df["diet_match"] = (
                ~(
                    diet_text.str.contains(
                        nonveg_pattern,
                        case=False,
                        na=False,
                        regex=True
                    )
                    |
                    diet_text.str.contains(
                        onion_garlic_pattern,
                        case=False,
                        na=False,
                        regex=True
                    )
                )
            ).astype(float)

        elif requested_diet == "non-vegetarian":
            df["diet_match"] = (
                diet_text.str.contains(
                    nonveg_pattern,
                    case=False,
                    na=False,
                    regex=True
                )
            ).astype(float)

    # --------------------------------------------------------
    # Cuisine match
    # --------------------------------------------------------

    df["cuisine_match"] = (
        cuisine_text.str.contains(
            requested_cuisine,
            na=False,
            regex=False
        )
        if requested_cuisine
        else 1.0
    )
    df["cuisine_match"] = df["cuisine_match"].astype(float)

    # --------------------------------------------------------
    # Cooking time match
    # --------------------------------------------------------

    time = pd.to_numeric(
        df["cooking_time_minutes"],
        errors="coerce"
    )

    try:
        max_time_value = float(context["max_time"])
    except (TypeError, ValueError):
        max_time_value = 30.0

    # Missing/invalid time is NOT treated as a match.
    df["time_match"] = (
        time.notna() & time.le(max_time_value)
    ).astype(float)

    # --------------------------------------------------------
    # Context score
    # --------------------------------------------------------

    df["context_score"] = (
        0.25 * df["meal_match"]
        + 0.25 * df["diet_match"]
        + 0.20 * df["cuisine_match"]
        + 0.15 * df["time_match"]
        + 0.10 * df["weather_score"]
        + 0.05 * df["festival_score"]
    ).clip(0, 1)

    print(
        "Context:",
        context
    )

    print(
        "Context matches:",
        f"meal={df['meal_match'].sum():.0f},",
        f"diet={df['diet_match'].sum():.0f},",
        f"cuisine={df['cuisine_match'].sum():.0f},",
        f"time={df['time_match'].sum():.0f}"
    )

    # ========================================================
    # NUTRITION
    # ========================================================
    # IMPORTANT:
    # Nutrition is DISPLAY-ONLY.
    #
    # It does NOT:
    # - train the recommendation model
    # - filter candidates
    # - affect candidate_base_score
    # - affect final_score
    #
    # Nutrition values are attached AFTER the ML ranking below.
    # ========================================================

    # ========================================================
    # 7. MULTI-FACTOR CANDIDATE FILTER
    # ========================================================
    # IMPORTANT: Random Forest is NOT used to search all 18,568
    # recipes. The recommendation signals first create an eligible
    # candidate pool. RF then reranks only that filtered pool.

    if "recipe_scope" in df.columns:
        result = df[
            df["recipe_scope"].fillna("").str.lower().eq("indian")
            & (df.index != ref_idx)
        ].copy()
    else:
        result = df[df.index != ref_idx].copy()

    def apply_constraint(current, mask, label):
        filtered = current[mask.loc[current.index]].copy()
        minimum = max(top_k, 5)
        if len(filtered) >= minimum:
            print(f"Candidate filter applied: {label} -> {len(filtered)}")
            return filtered
        print(f"Candidate filter relaxed: {label} (only {len(filtered)} candidates)")
        return current

    # All existing recommendation factors participate before RF.
    if requested_meal:
        result = apply_constraint(result, df["meal_match"].eq(1), "meal type")
    if requested_diet:
        result = apply_constraint(result, df["diet_match"].eq(1), "diet")
    if requested_cuisine:
        result = apply_constraint(result, df["cuisine_match"].eq(1), "cuisine")
    result = apply_constraint(result, df["time_match"].eq(1), "maximum cooking time")

    # Location is a strong preference, but pantry remains the primary factor.
    # Do not hard-filter here: a recipe with a stronger pantry match must not
    # disappear merely because it is less regional. Location participates in
    # the cumulative suitability score below.
    result["location_match"] = result["location_score"].clip(0, 1)

    # Pantry + expiry + location + context are combined into a transparent
    # cumulative score. Pantry is deliberately the largest component.
    result["candidate_base_score"] = (
        0.50 * result["pantry_score"]
        + 0.15 * result["expiry_score"]
        + 0.12 * result["location_match"]
        + 0.07 * result["festival_score"]
        + 0.05 * result["weather_score"]
        + 0.05 * result["context_score"]
        + 0.03 * result["content_score"]
        + 0.03 * result["han_score"]
    ).clip(0, 1)

    # IMPORTANT:
    # Do NOT truncate candidates using candidate_base_score.
    # Pantry priority must be evaluated across the entire eligible pool.
    # RF is applied to the complete eligible pool before the final
    # priority ordering.
    result = result.copy()

    # ========================================================
    # 8. RANDOM FOREST FINAL RERANKING
    # ========================================================
    if rf_model is not None:
        rf_input = pd.DataFrame(index=result.index)
        for feature in rf_features:
            if feature in result.columns:
                rf_input[feature] = pd.to_numeric(
                    result[feature], errors="coerce"
                ).fillna(0.0)
            else:
                rf_input[feature] = 0.0
        rf_input = rf_input[rf_features]

        try:
            rf_probability = rf_model.predict_proba(rf_input)[:, 1]
            result["rf_suitability_score"] = np.clip(
                np.asarray(rf_probability, dtype=float), 0.0, 1.0
            )
        except Exception as exc:
            print("RF reranking failed:", exc)
            result["rf_suitability_score"] = result["candidate_base_score"]
    else:
        result["rf_suitability_score"] = result["candidate_base_score"]

    # ========================================================
    # 9. PRIORITY-BASED FINAL RANKING
    # ========================================================
    # IMPORTANT:
    # Mealora uses a strict priority order for recommendation suitability:
    #
    #   1. Pantry availability
    #   2. Expiry / food-waste priority
    #   3. Context suitability
    #   4. Content similarity
    #   5. Random Forest suitability
    #   6. HAN graph score
    #
    # These are NOT combined into one weighted average.
    # A lower-priority signal must NOT overturn a higher-priority signal.
    #
    # Example:
    # A recipe with 13% pantry match must rank above one with 7% pantry
    # match, even if the 7% recipe has a higher expiry or content score.
    #
    # Nutrition is still completely excluded from ranking.
    # ========================================================

    # Transparent cumulative score shown as "Overall fit" in the UI.
    # Pantry is the main signal; ML is deliberately a small final signal.
    result["final_score"] = (
        0.55 * result["pantry_score"]
        + 0.15 * result["location_match"]
        + 0.10 * result["expiry_score"]
        + 0.07 * result["festival_score"]
        + 0.05 * result["weather_score"]
        + 0.04 * result["context_score"]
        + 0.02 * result["content_score"]
        + 0.015 * result["rf_suitability_score"]
        + 0.005 * result["han_score"]
    ).clip(0, 1)

    # The cumulative score balances every signal while keeping pantry
    # availability clearly dominant. Location is the strongest contextual
    # preference, and ML remains the smallest learned ranking signal.
    result = result.sort_values(
        ["final_score", "pantry_score", "location_match", "expiry_score"],
        ascending=[False, False, False, False],
        kind="mergesort"
    ).head(top_k).copy()

    # ========================================================
    # ATTACH NUTRITION FOR DISPLAY ONLY
    # ========================================================
    #
    # IMPORTANT:
    # Nutrition is attached ONLY AFTER recommendation ranking.
    #
    # Nutrition does NOT:
    # - train the ML model
    # - filter candidates
    # - affect candidate_base_score
    # - affect rf_suitability_score
    # - affect final_score
    #
    # The master recipe dataset already contains nutrition columns,
    # so remove those columns from the ranked result first.
    # Then attach the CLEAN nutrition dataset explicitly.
    # ========================================================

    # Actual nutrition values.
    # These fields alone determine whether nutrition is available.
    nutrition_fields = [
        "calories",
        "protein",
        "carbohydrates",
        "fat",
        "fibre",
        "sodium",
        "calcium",
        "iron",
        "vitamin_c",
        "folate",
    ]

    # Nutrition metadata is display/audit information only.
    # Metadata MUST NOT make nutrition_available=True.
    nutrition_metadata_fields = [
        "nutrition_source",
        "nutrition_confidence",
        "nutrition_coverage_pct",
    ]

    all_nutrition_display_fields = (
        nutrition_fields + nutrition_metadata_fields
    )

    # Remove any nutrition columns inherited from the master dataset.
    # This prevents pandas _x/_y merge conflicts.
    result = result.drop(
        columns=[
            c for c in nutrition_fields
            if c in result.columns
        ],
        errors="ignore"
    )

    result["nutrition_available"] = False

    if os.path.exists(NUTRITION):

        try:
            nutrition_df = pd.read_csv(
                NUTRITION,
                low_memory=False
            )

            if "recipe_id" not in nutrition_df.columns:
                raise RuntimeError(
                    "CLEAN nutrition dataset has no recipe_id column."
                )

            nutrition_df["recipe_id"] = (
                nutrition_df["recipe_id"]
                .astype(str)
                .str.strip()
            )

            result["_nutrition_recipe_id"] = (
                result["recipe_id"]
                .astype(str)
                .str.strip()
            )

            # Keep actual nutrition values and metadata that exist.
            available_nutrition_fields = [
                c for c in nutrition_fields
                if c in nutrition_df.columns
            ]

            available_nutrition_metadata_fields = [
                c for c in nutrition_metadata_fields
                if c in nutrition_df.columns
            ]

            available_nutrition_display_fields = (
                available_nutrition_fields
                + available_nutrition_metadata_fields
            )

            nutrition_small = nutrition_df[
                ["recipe_id"] + available_nutrition_display_fields
            ].copy()

            # One row per recipe.
            nutrition_small = (
                nutrition_small
                .drop_duplicates(
                    subset=["recipe_id"],
                    keep="first"
                )
            )

            # Rename recipe_id so there is no collision with result.recipe_id.
            nutrition_small = nutrition_small.rename(
                columns={
                    "recipe_id": "_nutrition_recipe_id"
                }
            )

            result = result.merge(
                nutrition_small,
                on="_nutrition_recipe_id",
                how="left"
            )

            if available_nutrition_fields:

                # IMPORTANT:
                # Only actual nutrition values determine availability.
                # Source/confidence/coverage metadata do NOT count.
                result["nutrition_available"] = (
                    result[available_nutrition_fields]
                    .notna()
                    .any(axis=1)
                )

            print(
                "Nutrition attached for DISPLAY ONLY:",
                f"{int(result['nutrition_available'].sum())}/{len(result)}"
            )

            result.drop(
                columns=["_nutrition_recipe_id"],
                inplace=True,
                errors="ignore"
            )

        except Exception as exc:

            print(
                "Nutrition display attachment warning:",
                exc
            )

            result.drop(
                columns=["_nutrition_recipe_id"],
                inplace=True,
                errors="ignore"
            )

    else:

        print(
            "Nutrition display dataset not found:",
            NUTRITION
        )

    # Guarantee stable frontend fields even when nutrition is unavailable.
    for _nutrition_column in all_nutrition_display_fields:

        if _nutrition_column not in result.columns:
            result[_nutrition_column] = np.nan

    result["nutrition_available"] = (
        result["nutrition_available"]
        .fillna(False)
        .astype(bool)
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return result[
        [
            "recipe_id",
            "recipe_name",
            "content_score",
            "hybrid_score",
            "pantry_score",
            "missing_ingredient_count",
            "expiry_score",
            "context_score",
            "time_match",
            "weather_score",
            "festival_score",
            "location_score",
            "location_match",
            "rf_suitability_score",
            "han_score",
            "nutrition_available",
            "calories",
            "protein",
            "carbohydrates",
            "fat",
            "fibre",
            "sodium",
            "calcium",
            "iron",
            "vitamin_c",
            "folate",
            "nutrition_source",
            "nutrition_confidence",
            "nutrition_coverage_pct",
            "candidate_base_score",
            "final_score",
        ]
    ]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = recommend(
        reference_recipe="Balu shahi",
        target_user=1,
        top_k=5,
        meal_type="Dinner",
        diet="Vegetarian",
        cuisine="South Indian",
        max_time=30
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "MEALORA FINAL "
        "RECOMMENDATION RANKER"
    )

    print(
        "=" * 100
    )

    print(
        "\nTop Recommendations:\n"
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RECOMMENDATION PIPELINE"
    )

    print(
        "=" * 100
    )

    print(
        "\n"
        "1. TF-IDF + Cosine Similarity\n"
        "2. Content-Based Recipe Similarity\n"
        "3. Pantry Compatibility\n"
        "4. Expiry / Food-Waste Score\n"
        "5. Context Score\n"
        "6. Multi-factor Candidate Filtering\n"
        "7. Random Forest Final Reranking\n"
        "8. HAN Graph Reranking\n"
        "9. Nutrition Information Display (non-ranking)"
    )

    print(
        "\n"
        + "=" * 100
    )