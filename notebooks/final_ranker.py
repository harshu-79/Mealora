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

HGB_MODEL = os.path.join(
    BASE,
    "mealora_hgb_model.joblib"
)


# ============================================================
# HGB FEATURES
# ============================================================

HGB_FEATURES = [
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
# LOAD HGB MODEL
# ============================================================

print("\nLoading Mealora data...")

saved_object = joblib.load(HGB_MODEL)

print(
    "Saved HGB object type:",
    type(saved_object)
)

if isinstance(saved_object, dict):

    print(
        "Saved HGB object is a dictionary."
    )

    print(
        "Dictionary keys:",
        list(saved_object.keys())
    )

    if "model" not in saved_object:
        raise ValueError(
            "HGB dictionary does not contain 'model'."
        )

    hgb_model = saved_object["model"]

    print(
        "HGB model found under key: 'model'"
    )

else:

    hgb_model = saved_object

    print(
        "HGB model loaded directly."
    )


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
    # 2. COLLABORATIVE FILTERING
    # ========================================================

    print(
        "Calculating collaborative filtering..."
    )

    # --------------------------------------------------------
    # Interaction weights
    # --------------------------------------------------------

    interaction_weights = {
        "cooked": 5,
        "liked": 4,
        "saved": 3,
        "skipped": 1,
        "disliked": 0,
    }

    interactions["preference"] = (
        interactions["interaction"]
        .map(interaction_weights)
    )

    # Use rating when interaction is unknown
    interactions["preference"] = (
        interactions["preference"]
        .fillna(
            pd.to_numeric(
                interactions["rating"],
                errors="coerce"
            )
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # User-recipe preference matrix
    # --------------------------------------------------------

    matrix = interactions.pivot_table(
        index="user_id",
        columns="recipe_id",
        values="preference",
        aggfunc="mean",
        fill_value=0
    )

    collaborative = {}

    if target_user in matrix.index:

        # ----------------------------------------------------
        # User similarity
        # ----------------------------------------------------

        user_similarity = cosine_similarity(
            matrix
        )

        similarity_df = pd.DataFrame(
            user_similarity,
            index=matrix.index,
            columns=matrix.index
        )

        neighbors = (
            similarity_df.loc[target_user]
            .drop(target_user)
            .sort_values(
                ascending=False
            )
        )

        # ----------------------------------------------------
        # Generate recommendations from similar users
        # ----------------------------------------------------

        target_history = set(
            interactions[
                interactions["user_id"] == target_user
            ]["recipe_id"]
        )

        for neighbor, similarity in neighbors.items():

            if similarity <= 0:
                continue

            neighbor_preferences = matrix.loc[
                neighbor
            ]

            for recipe_id, preference in (
                neighbor_preferences.items()
            ):

                # Skip recipes already used
                # by target user
                if recipe_id in target_history:
                    continue

                if preference <= 0:
                    continue

                score = (
                    similarity * preference
                )

                collaborative[recipe_id] = (
                    collaborative.get(
                        recipe_id,
                        0
                    )
                    + score
                )

    # --------------------------------------------------------
    # Normalize collaborative scores
    # --------------------------------------------------------

    max_collaborative = max(
        collaborative.values(),
        default=0
    )

    if max_collaborative > 0:

        collaborative = {
            recipe_id:
            score / max_collaborative

            for recipe_id, score
            in collaborative.items()
        }

    # --------------------------------------------------------
    # Map CF scores to master dataset
    # --------------------------------------------------------

    df["collaborative_score"] = (
        df["recipe_id"]
        .map(collaborative)
        .fillna(0)
    )

    # ========================================================
    # 3. HYBRID CONTENT + COLLABORATIVE
    # ========================================================

    df["hybrid_score"] = (
        0.60 * df["content_score"]
        +
        0.40 * df["collaborative_score"]
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
    diet_text = df["diet_type"].fillna("").astype(str).str.lower().str.strip()
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
    # 7. HISTGRADIENTBOOSTING SUITABILITY
    # ========================================================

    print(
        "\nRunning HistGradientBoosting "
        "suitability model..."
    )

    # --------------------------------------------------------
    # Prepare HGB input
    # --------------------------------------------------------

    hgb_input = pd.DataFrame(
        index=df.index
    )

    for feature in HGB_FEATURES:

        if feature in df.columns:

            hgb_input[feature] = (
                pd.to_numeric(
                    df[feature],
                    errors="coerce"
                )
                .fillna(0)
            )

        else:

            print(
                f"WARNING: Missing feature "
                f"'{feature}'. Using 0."
            )

            hgb_input[feature] = 0.0

    # Ensure exact feature order
    hgb_input = hgb_input[
        HGB_FEATURES
    ]

    print(
        "HGB input shape:",
        hgb_input.shape
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    hgb_probability = (
        hgb_model
        .predict_proba(
            hgb_input
        )[:, 1]
    )

    df["hgb_suitability_score"] = (
        hgb_probability
    )

    print(
        "HGB suitability prediction completed."
    )

    # ========================================================
    # 8. FINAL WEIGHTED RANKING
    # ========================================================

    # Context is deliberately stronger now so that changing
    # meal type, diet, cuisine or time changes the ranking.
    df["final_score"] = (
        0.25 * df["hybrid_score"]
        + 0.20 * df["pantry_score"]
        + 0.10 * df["expiry_score"]
        + 0.20 * df["context_score"]
        + 0.05 * df["time_match"]
        + 0.15 * df["hgb_suitability_score"]
        + 0.03 * df["weather_score"]
        + 0.02 * df["festival_score"]
    )

    # ========================================================
    # CONTEXT-AWARE CANDIDATE FILTER
    # ========================================================

    if "recipe_scope" in df.columns:
        result = df[
            df["recipe_scope"].fillna("").str.lower().eq("indian")
            & (df.index != ref_idx)
        ].copy()
    else:
        result = df[df.index != ref_idx].copy()

    # Apply hard constraints only when they produce candidates.
    # This prevents obviously incompatible recipes from dominating
    # while still allowing graceful fallback for sparse metadata.

    def apply_constraint(current, mask, label):
        filtered = current[mask.loc[current.index]]
        if len(filtered) >= max(top_k, 5):
            print(f"Hard filter applied: {label} -> {len(filtered)} candidates")
            return filtered
        print(f"Hard filter skipped: {label} (only {len(filtered)} candidates)")
        return current

    # Meal type is highly relevant to the user request.
    if requested_meal:
        result = apply_constraint(
            result,
            df["meal_match"].eq(1),
            "meal type"
        )

    # Diet should be respected strongly.
    if requested_diet:
        result = apply_constraint(
            result,
            df["diet_match"].eq(1),
            "diet"
        )

    # Cuisine should influence the pool when enough recipes exist.
    if requested_cuisine:
        result = apply_constraint(
            result,
            df["cuisine_match"].eq(1),
            "cuisine"
        )

    # Maximum cooking time is a real user constraint.
    result = apply_constraint(
        result,
        df["time_match"].eq(1),
        "maximum cooking time"
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            "final_score",
            ascending=False
        )
        .head(top_k)
    )

    # ========================================================
    # AUTOMATIC MIXED-DIET DIVERSIFICATION
    # ========================================================
    if requested_diet in {"mixed", "mixed diet", "all"} and top_k >= 2:
        nonveg_pattern = (
            r"non.?vegetarian|nonveg|chicken|mutton|lamb|beef|pork|"
            r"fish|prawn|shrimp|crab|meat|egg"
        )
        candidates = df.copy().sort_values("final_score", ascending=False)
        diet_text_for_mix = candidates.get(
            "diet_type", pd.Series("", index=candidates.index)
        ).fillna("").astype(str)
        name_text_for_mix = candidates.get(
            "recipe_name", pd.Series("", index=candidates.index)
        ).fillna("").astype(str)
        is_nonveg = (
            diet_text_for_mix.str.contains(nonveg_pattern, case=False, regex=True, na=False)
            | name_text_for_mix.str.contains(nonveg_pattern, case=False, regex=True, na=False)
        )
        veg = candidates[~is_nonveg]
        nonveg = candidates[is_nonveg]
        selected = []
        vi = ni = 0
        for i in range(top_k):
            if i % 2 == 1 and ni < len(nonveg):
                selected.append(nonveg.iloc[ni])
                ni += 1
            elif vi < len(veg):
                selected.append(veg.iloc[vi])
                vi += 1
            elif ni < len(nonveg):
                selected.append(nonveg.iloc[ni])
                ni += 1
        if selected:
            result = pd.DataFrame(selected)

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return result[
        [
            "recipe_id",
            "recipe_name",
            "content_score",
            "collaborative_score",
            "hybrid_score",
            "pantry_score",
            "missing_ingredient_count",
            "expiry_score",
            "context_score",
            "time_match",
            "weather_score",
            "festival_score",
            "location_score",
            "hgb_suitability_score",
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
        "2. Collaborative Filtering\n"
        "3. Hybrid Content + Collaborative Score\n"
        "4. Pantry Compatibility\n"
        "5. Expiry / Food-Waste Score\n"
        "6. Context Score\n"
        "7. HistGradientBoosting Suitability Prediction\n"
        "8. Final Weighted Ranking"
    )

    print(
        "\n"
        + "=" * 100
    )