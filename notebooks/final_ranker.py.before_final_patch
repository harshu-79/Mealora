import os
import re
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

    # Location-aware scoring. The backend normally supplies a
    # cuisine_hint, but accepting the city here makes the ranker
    # robust for both API and direct Python calls.
    location_score = pd.Series(0.0, index=df.index)

    city = _norm_text(
        location.get("city")
        or location.get("location_city")
        or location.get("name")
        or location.get("location_name")
    )

    location_cuisine = _norm_text(location.get("cuisine_hint"))

    city_cuisine = {
        "chennai": "south indian",
        "coimbatore": "south indian",
        "madurai": "south indian",
        "trichy": "south indian",
        "tiruchirappalli": "south indian",
        "salem": "south indian",
        "hyderabad": "andhra",
        "vijayawada": "andhra",
        "visakhapatnam": "andhra",
        "bengaluru": "south indian",
        "bangalore": "south indian",
        "mysore": "south indian",
        "mysuru": "south indian",
        "kochi": "kerala",
        "thiruvananthapuram": "kerala",
        "trivandrum": "kerala",
        "mumbai": "maharashtrian",
        "pune": "maharashtrian",
        "kolkata": "bengali",
        "jaipur": "rajasthani",
        "udaipur": "rajasthani",
        "lucknow": "north indian",
        "delhi": "north indian",
        "new delhi": "north indian",
    }

    if not location_cuisine and city:
        location_cuisine = city_cuisine.get(city, "")

    if location_cuisine:
        location_score = _keyword_score(
            df,
            [(location_cuisine, 1.0)]
        )

    return weather_score, festival_score, location_score



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

    interactions = pd.read_csv(
        INTERACTIONS
    )

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
    # 2. PERSONAL BEHAVIOUR LEARNING
    # ========================================================

    print(
        "Calculating personal behaviour score..."
    )

    interaction_weights = {
        "cooked": 1.00,
        "liked": 0.85,
        "saved": 0.70,
        "skipped": 0.20,
        "disliked": 0.00,
    }

    if "interaction" not in interactions.columns:
        interactions["interaction"] = ""

    interactions["preference"] = (
        interactions["interaction"]
        .astype(str)
        .str.lower()
        .map(interaction_weights)
    )

    if "rating" in interactions.columns:
        interactions["preference"] = interactions["preference"].fillna(
            pd.to_numeric(interactions["rating"], errors="coerce") / 5.0
        )

    interactions["preference"] = interactions["preference"].fillna(0).clip(0, 1)

    personal = (
        interactions[interactions["user_id"] == target_user]
        .groupby("recipe_id")["preference"]
        .mean()
        if "user_id" in interactions.columns and "recipe_id" in interactions.columns
        else pd.Series(dtype=float)
    )

    if len(personal) > 0 and personal.max() > 0:
        personal = personal / personal.max()

    df["personal_behavior_score"] = (
        df["recipe_id"].map(personal).fillna(0).clip(0, 1)
    )

    # ========================================================
    # 3. HYBRID CONTENT + PERSONAL BEHAVIOUR
    # ========================================================

    df["hybrid_score"] = (
        0.70 * df["content_score"]
        +
        0.30 * df["personal_behavior_score"]
    )

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

    print(
        "Calculating expiry score..."
    )

    expiry_map = expiry.set_index(
        "recipe_id"
    )["expiry_score"]

    df["expiry_score"] = (
        df["recipe_id"]
        .map(expiry_map)
        .fillna(0)
    )

    df["expiry_score"] = pd.to_numeric(
        df["expiry_score"],
        errors="coerce"
    ).fillna(0)

    df["expiry_score"] = (
        df["expiry_score"]
        .clip(
            lower=0,
            upper=1
        )
    )

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
        0.22 * df["meal_match"]
        + 0.22 * df["diet_match"]
        + 0.16 * df["cuisine_match"]
        + 0.15 * df["time_match"]
        + 0.10 * df["weather_score"]
        + 0.10 * df["festival_score"]
        + 0.05 * df["location_score"]
    )

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
    # NUTRITION SCORE
    # ========================================================
    # Only recipes with real nutrition values are scored. Missing
    # nutrition is neutral and never treated as zero nutrition.
    nutrition_cols = {
        "calories": ["calories", "Calories (kcal)"],
        "protein": ["protein", "Protein (g)"],
        "carbohydrates": ["carbohydrates", "Carbohydrates (g)"],
        "fat": ["fat", "Fats (g)"],
        "fibre": ["fibre", "Fibre (g)"],
    }

    def _find_col(options):
        for c in options:
            if c in df.columns:
                return c
        return None

    nmap = {k: _find_col(v) for k, v in nutrition_cols.items()}
    nutrition_available = pd.Series(True, index=df.index)
    for c in nmap.values():
        if c is not None:
            nutrition_available &= pd.to_numeric(df[c], errors="coerce").notna()
        else:
            nutrition_available &= False

    df["nutrition_available"] = nutrition_available.astype(float)
    df["nutrition_score"] = 0.5

    if nutrition_available.any():
        valid = df.loc[nutrition_available].copy()

        def _norm_col(key):
            c = nmap[key]
            x = pd.to_numeric(valid[c], errors="coerce")
            if x.max() == x.min():
                return pd.Series(0.5, index=valid.index)
            return ((x - x.min()) / (x.max() - x.min())).clip(0, 1).fillna(0.5)

        protein_n = _norm_col("protein")
        fibre_n = _norm_col("fibre")
        calories = pd.to_numeric(valid[nmap["calories"]], errors="coerce")
        fat = pd.to_numeric(valid[nmap["fat"]], errors="coerce")
        cal_med = calories.median()
        fat_med = fat.median()
        cal_dev = (calories - cal_med).abs()
        fat_dev = (fat - fat_med).abs()
        calorie_score = 1 - (cal_dev / cal_dev.max()).replace([np.inf, -np.inf], np.nan).fillna(0.5) if cal_dev.max() else pd.Series(0.5, index=valid.index)
        fat_score = 1 - (fat_dev / fat_dev.max()).replace([np.inf, -np.inf], np.nan).fillna(0.5) if fat_dev.max() else pd.Series(0.5, index=valid.index)
        df.loc[valid.index, "nutrition_score"] = (
            0.35 * protein_n + 0.30 * fibre_n + 0.20 * calorie_score + 0.15 * fat_score
        ).clip(0, 1)

    print(
        "Nutrition coverage in current recipes:",
        f"{int(df['nutrition_available'].sum())}/{len(df)}",
        f"({100 * df['nutrition_available'].mean():.2f}%)"
    )

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

    # Pantry + expiry + contextual compatibility are soft-ranked
    # within the eligible pool so sparse pantry data does not empty it.
    result["candidate_base_score"] = (
        0.25 * result["hybrid_score"]
        + 0.25 * result["pantry_score"]
        + 0.15 * result["expiry_score"]
        + 0.15 * result["context_score"]
        + 0.10 * result["nutrition_score"]
        + 0.05 * result["time_match"]
        + 0.03 * result["weather_score"]
        + 0.02 * result["festival_score"]
    ).clip(0, 1)

    # Keep a bounded candidate pool. RF is the final supervised
    # reranker, not the initial filter.
    candidate_limit = max(100, top_k * 40)
    result = result.sort_values(
        ["candidate_base_score", "pantry_score", "expiry_score"],
        ascending=False
    ).head(candidate_limit).copy()

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
    # 9. FINAL RF + HAN HYBRID RERANKING
    # ========================================================
    # RF and HAN operate on the SAME already-filtered candidate pool.
    # They are final learned ranking signals, not broad candidate filters.
    result["final_score"] = (
        0.20 * result["content_score"]
        + 0.15 * result["pantry_score"]
        + 0.10 * result["expiry_score"]
        + 0.10 * result["context_score"]
        + 0.10 * result["nutrition_score"]
        + 0.20 * result["rf_suitability_score"]
        + 0.15 * result["han_score"]
    ).clip(0, 1)

    result = result.sort_values(
        ["final_score", "rf_suitability_score", "han_score", "candidate_base_score"],
        ascending=False
    ).head(top_k).copy()

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return result[
        [
            "recipe_id",
            "recipe_name",
            "content_score",
            "personal_behavior_score",
            "hybrid_score",
            "pantry_score",
            "missing_ingredient_count",
            "expiry_score",
            "context_score",
            "time_match",
            "weather_score",
            "festival_score",
            "location_score",
            "rf_suitability_score",
            "han_score",
            "nutrition_score",
            "nutrition_available",
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
        "2. Personal Behaviour Learning\n"
        "3. Hybrid Content + Personal Behaviour Score\n"
        "4. Pantry Compatibility\n"
        "5. Expiry / Food-Waste Score\n"
        "6. Context Score\n"
        "7. Multi-factor Candidate Filtering\n"
        "8. Random Forest Final Reranking\n"
        "9. HAN Graph Reranking\n"
        "10. Nutrition-aware Hybrid Final Ranking"
    )

    print(
        "\n"
        + "=" * 100
    )