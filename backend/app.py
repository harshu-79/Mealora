from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import re
import json
import ast
import joblib
import pandas as pd
import numpy as np
import urllib.request
import urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo

# Gemini is used only for multilingual recipe presentation.
# The ML recommendation engine remains the source of recipe selection.

sys.path.append(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "notebooks"
))
from final_ranker import recommend

RECIPE_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "datasets", "processed", "recipe_master_dataset.csv"
)

# Cached recipe master dataset.
# The weekly planner calls this endpoint frequently, so avoid re-reading
# the 18,568-row CSV on every request.
_RECIPE_DF_CACHE = None
_RECIPE_DF_MTIME = None

def get_recipe_df():
    global _RECIPE_DF_CACHE, _RECIPE_DF_MTIME

    if not os.path.exists(RECIPE_DATASET):
        raise FileNotFoundError(
            f"Recipe dataset not found: {RECIPE_DATASET}"
        )

    mtime = os.path.getmtime(RECIPE_DATASET)

    if (
        _RECIPE_DF_CACHE is None
        or _RECIPE_DF_MTIME != mtime
    ):
        print("Loading Mealora recipe master dataset...")
        _RECIPE_DF_CACHE = pd.read_csv(
            RECIPE_DATASET,
            low_memory=False
        )
        _RECIPE_DF_MTIME = mtime
        print(
            f"Recipe master loaded: {len(_RECIPE_DF_CACHE):,} recipes"
        )

    return _RECIPE_DF_CACHE


# ============================================================
# WEEKLY PLANNER CACHE + ML MODEL
# ============================================================
_WEEKLY_RECORD_CACHE = None
_WEEKLY_RECORD_MTIME = None
_WEEKLY_ML_MODEL = None
_WEEKLY_ML_MODEL_NAME = "None"


def _load_weekly_ml_model():
    """Load Mealora's trained ranking model once and reuse it."""
    global _WEEKLY_ML_MODEL, _WEEKLY_ML_MODEL_NAME

    if _WEEKLY_ML_MODEL is not None:
        return _WEEKLY_ML_MODEL

    base = os.path.dirname(os.path.abspath(__file__))
    model_paths = [
        (
            os.path.join(
                base, "..", "datasets", "processed", "mealora_rf_model.joblib"
            ),
            "Random Forest",
        ),
        (
            os.path.join(
                base, "..", "datasets", "processed", "mealora_hgb_model.joblib"
            ),
            "HistGradientBoosting",
        ),
    ]

    for path, name in model_paths:
        if os.path.exists(path):
            try:
                _WEEKLY_ML_MODEL = joblib.load(path)
                _WEEKLY_ML_MODEL_NAME = name
                print(f"Loaded weekly planner ML model: {name}")
                return _WEEKLY_ML_MODEL
            except Exception as exc:
                print(f"Weekly ML model load warning ({name}): {exc}")

    print("Weekly planner ML model: no trained model found; using neutral ML prior.")
    return None


def _weekly_first_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        key = str(candidate).lower()
        if key in lower:
            return lower[key]
    return None


def _weekly_clean_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _weekly_norm(value):
    s = _weekly_clean_text(value).lower()
    s = s.replace("&", " and ")
    s = s.replace("chilli", "chili")
    s = s.replace("capsicum", "bell pepper")
    s = s.replace("coriander leaves", "cilantro")
    s = s.replace("coriander leaf", "cilantro")
    s = s.replace("curd", "yogurt")
    s = s.replace("dahi", "yogurt")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _weekly_tokens(value):
    stop = {
        "the", "and", "with", "for", "from", "into", "this", "that",
        "cup", "cups", "tsp", "tbsp", "tablespoon", "tablespoons",
        "teaspoon", "teaspoons", "recipe", "fresh", "dried", "dry",
        "whole", "ground", "chopped", "sliced", "small", "large",
        "medium", "finely", "roughly",
    }
    out = set()
    for token in _weekly_norm(value).split():
        if len(token) < 3 or token in stop:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("es") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        out.add(token)
    return out


def _weekly_parse_ingredients(value):
    if isinstance(value, list):
        return [x for x in (_weekly_clean_text(v) for v in value) if x]

    s = _weekly_clean_text(value)
    if not s or s.lower() in {"nan", "none", "null"}:
        return []

    if s.startswith("[") and s.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(s)
                if isinstance(parsed, list):
                    return [x for x in (_weekly_clean_text(v) for v in parsed) if x]
            except Exception:
                pass

    parts = re.split(r"\n|;|\|", s)
    if len(parts) == 1:
        parts = [x.strip() for x in s.split(",")]

    return [x for x in (_weekly_clean_text(v) for v in parts) if x]


_WEEKLY_ROLE_TERMS = {
    "breakfast": (
        "idli", "dosa", "dosai", "uttapam", "upma", "poha", "pongal",
        "pesarattu", "adai", "appam", "puttu", "paratha", "parantha",
        "puri", "poori", "vada", "omelette", "omelet", "toast",
        "pancake", "thepla", "dhokla", "ven pongal",
    ),
    "chutney": (
        "chutney", "thogayal", "thuvaiyal", "pachadi", "dip",
    ),
    "accompaniment": (
        "sambar", "sambhar", "gothsu", "kurma", "korma", "stew",
        "potato masala", "aloo masala", "saagu", "vegetable curry",
        "veg curry", "dal", "dhal", "pappu", "rasam", "kadhi",
    ),
    "staple": (
        "white rice", "plain rice", "steamed rice", "rice",
        "jeera rice", "brown rice", "curd rice", "lemon rice",
        "coconut rice", "tomato rice", "pulao", "pulav",
        "fried rice", "biryani", "biriyani", "khichdi", "khichadi",
        "bisibele", "bisi bele", "roti", "chapati", "phulka",
        "naan", "kulcha", "paratha", "parantha",
    ),
    "dal_gravy": (
        "dal", "dhal", "pappu", "sambar", "sambhar", "rasam", "kadhi",
        "curry", "gravy", "korma", "kurma", "masala", "stew",
    ),
    "vegetable": (
        "poriyal", "palya", "thoran", "kootu", "avial", "sabzi",
        "subzi", "bhaji", "vegetable fry", "veg fry", "stir fry",
        "beans fry", "potato fry", "cabbage", "carrot", "beetroot",
        "okra", "bhindi", "brinjal", "eggplant", "cauliflower", "peas",
    ),
    "protein": (
        "chicken", "mutton", "lamb", "fish", "prawn", "prawns",
        "shrimp", "egg curry", "paneer", "kofta", "meat", "seafood",
    ),
    "light": (
        "raita", "salad", "curd", "yogurt", "pickle", "papad", "kosambari",
    ),
    "snack": (
        "tea", "coffee", "pakora", "bajji", "bonda", "samosa",
        "sandwich", "cutlet", "snack", "murukku", "mixture",
    ),
}


def _weekly_has(name, role):
    n = _weekly_norm(name)
    return any(term in n for term in _WEEKLY_ROLE_TERMS[role])


def _weekly_role(name, raw_meal_type=""):
    raw = _weekly_norm(raw_meal_type)
    name_n = _weekly_norm(name)

    if "breakfast" in raw or "morning" in raw:
        return "breakfast"
    if "snack" in raw or "starter" in raw or "appetizer" in raw:
        return "snack"
    if "lunch" in raw:
        return "main"
    if "dinner" in raw or "supper" in raw:
        return "main"

    if _weekly_has(name_n, "breakfast"):
        return "breakfast"
    if _weekly_has(name_n, "snack"):
        return "snack"
    return "main"


def _build_weekly_recipe_cache():
    """Preprocess the 18,568-row master once; requests then score candidates only."""
    global _WEEKLY_RECORD_CACHE, _WEEKLY_RECORD_MTIME

    mtime = os.path.getmtime(RECIPE_DATASET)
    if _WEEKLY_RECORD_CACHE is not None and _WEEKLY_RECORD_MTIME == mtime:
        return _WEEKLY_RECORD_CACHE

    df = get_recipe_df()
    name_col = _weekly_first_col(
        df, ["recipe_name", "name", "title", "recipe", "dish_name"]
    )
    id_col = _weekly_first_col(df, ["recipe_id", "id", "ID", "recipeid"])
    ing_col = _weekly_first_col(
        df, [
            "ingredients", "ingredient_names", "cleaned_ingredients",
            "Cleaned-Ingredients", "TranslatedIngredients", "ingredient_list",
            "Ingredients",
        ]
    )
    meal_col = _weekly_first_col(
        df, ["meal_type", "meal", "mealtype", "meal_type_name"]
    )
    cuisine_col = _weekly_first_col(df, ["cuisine", "cuisine_type"])
    state_col = _weekly_first_col(df, ["state", "state_name", "region_state"])
    region_col = _weekly_first_col(
        df, ["region", "regional", "regional_cuisine", "food_region"]
    )
    diet_col = _weekly_first_col(
        df, ["diet", "diet_type", "diet_category", "diet_type"]
    )
    time_col = _weekly_first_col(
        df, [
            "cooking_time_minutes", "cooking_time", "cook_time",
            "prep_time_minutes", "total_time_minutes", "total_time",
        ]
    )
    # The actual Mealora master uses `nutrition` plus macro columns
    # (calories, protein, fat, fibre, sodium). Older versions looked only
    # for nutrition_score/health_score, which caused a misleading 0%.
    nutrition_col = _weekly_first_col(
        df, ["nutrition_score", "nutrition_match_score", "health_score", "nutrition"]
    )
    protein_col = _weekly_first_col(df, ["protein"])
    fibre_col = _weekly_first_col(df, ["fibre", "fiber"])
    calories_col = _weekly_first_col(df, ["calories", "calorie"])
    fat_col = _weekly_first_col(df, ["fat", "total_fat"])
    sodium_col = _weekly_first_col(df, ["sodium"])
    context_col = _weekly_first_col(
        df, ["context_score", "context_match_score", "context_suitability_score"]
    )

    if not name_col or not ing_col:
        raise RuntimeError(
            "Mealora could not identify recipe_name/ingredients columns "
            f"in {RECIPE_DATASET}."
        )

    records = []
    token_index = {}

    for idx, row in df.iterrows():
        name = _weekly_clean_text(row.get(name_col))
        ingredients = _weekly_parse_ingredients(row.get(ing_col))
        if not name or not ingredients:
            continue

        ingredient_norms = [_weekly_norm(x) for x in ingredients]
        ingredient_tokens = [_weekly_tokens(x) for x in ingredients]
        all_tokens = set().union(*ingredient_tokens) if ingredient_tokens else set()

        recipe_id = (
            _weekly_clean_text(row.get(id_col))
            if id_col else str(idx)
        )
        if not recipe_id:
            recipe_id = str(idx)

        raw_time = row.get(time_col) if time_col else None
        try:
            cook_time = float(raw_time)
            if not np.isfinite(cook_time):
                cook_time = None
        except Exception:
            cook_time = None

        def numeric(col, default=0.5):
            if not col:
                return default
            try:
                value = float(row.get(col))
                if np.isfinite(value):
                    return float(np.clip(value, 0, 1))
            except Exception:
                pass
            return default

        record = {
            "recipe_id": recipe_id,
            "recipe_name": name,
            "ingredients": ingredients,
            "_ingredient_norms": ingredient_norms,
            "_ingredient_tokens": ingredient_tokens,
            "_all_tokens": all_tokens,
            "meal_type": _weekly_clean_text(row.get(meal_col)) if meal_col else "",
            "cuisine": _weekly_clean_text(row.get(cuisine_col)) if cuisine_col else "",
            "state": _weekly_clean_text(row.get(state_col)) if state_col else "",
            "region": _weekly_clean_text(row.get(region_col)) if region_col else "",
            "diet": _weekly_clean_text(row.get(diet_col)) if diet_col else "",
            "cook_time": cook_time,
            "_nutrition_raw": numeric(nutrition_col, 0.5),
            "_context_raw": numeric(context_col, 0.5),
            "_protein": numeric(protein_col, None),
            "_fibre": numeric(fibre_col, None),
            "_calories": numeric(calories_col, None),
            "_fat": numeric(fat_col, None),
            "_sodium": numeric(sodium_col, None),
        }
        record["_role"] = _weekly_role(name, record["meal_type"])
        # Quality prior: prefer ordinary dish names over verbose/generated variants.
        nn = _weekly_norm(name)
        quality = 0.0
        if len(nn.split()) <= 7:
            quality += 0.05
        if any(t in nn for t in ("using", "pressure cooker", "instant pot", "leftover", "fusion", "cheese toast")):
            quality -= 0.15
        if any(t in nn for t in ("recipe with", "recipe using")):
            quality -= 0.10
        record["_quality"] = quality
        records.append(record)

        for tok in all_tokens:
            token_index.setdefault(tok, set()).add(len(records) - 1)

    # A bounded fallback list of ordinary/common-looking Indian dishes.
    quality_indices = sorted(
        range(len(records)),
        key=lambda i: (records[i].get("_quality", 0.0), -len(records[i]["recipe_name"].split())),
        reverse=True,
    )
    _WEEKLY_RECORD_CACHE = {
        "records": records,
        "token_index": token_index,
        "quality_indices": quality_indices,
    }
    _WEEKLY_RECORD_MTIME = mtime
    print(f"Weekly planner index ready: {len(records):,} recipes")
    return _WEEKLY_RECORD_CACHE

def _load_backend_env():
    """Load simple KEY=VALUE pairs from backend/.env without exposing them to React."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
    except Exception as e:
        print("Backend .env load warning:", e)


_load_backend_env()

# Gemini 3.6 Flash is the current stable model.
# Allow backend/.env to override it, but never fall back to the retired 2.5 model.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

app = Flask(__name__)
CORS(app)


def _json_safe(value):
    """Convert pandas/NumPy missing values to JSON-safe null values."""
    if value is None:
        return None

    if value is pd.NA:
        return None

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, np.bool_):
        return bool(value)

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value

FESTIVALS = {
    # 2026 India major festivals / occasions.
    # Dates verified against Timeanddate India 2026 and Drik Panchang.
    "2026-01-01": "New Year's Day",
    "2026-01-13": "Lohri",
    "2026-01-14": "Makar Sankranti / Pongal",
    "2026-01-23": "Vasant Panchami",
    "2026-01-26": "Republic Day",
    "2026-02-15": "Maha Shivaratri",
    "2026-03-03": "Holika Dahan",
    "2026-03-04": "Holi",
    "2026-03-19": "Ugadi / Gudi Padwa / Chaitra Navratri",
    "2026-03-21": "Eid / Ramzan Id",
    "2026-03-26": "Rama Navami",
    "2026-03-31": "Mahavir Jayanti",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Vaisakhi / Ambedkar Jayanti",
    "2026-04-15": "Bohag Bihu",
    "2026-05-01": "Buddha Purnima",
    "2026-05-28": "Bakrid / Eid al-Adha",
    "2026-06-26": "Muharram / Ashura",
    "2026-07-16": "Rath Yatra",
    "2026-07-29": "Guru Purnima",
    "2026-08-15": "Independence Day",
    "2026-08-26": "Onam / Milad-un-Nabi",
    "2026-08-28": "Raksha Bandhan",
    "2026-09-04": "Krishna Janmashtami",
    "2026-09-14": "Ganesh Chaturthi",
    "2026-10-02": "Gandhi Jayanti",
    "2026-10-11": "Sharad Navratri Begins",
    "2026-10-17": "Durga Puja Begins",
    "2026-10-19": "Durga Ashtami / Maha Navami",
    "2026-10-20": "Dussehra / Vijayadashami",
    "2026-10-29": "Karwa Chauth",
    "2026-11-08": "Diwali / Deepavali / Naraka Chaturdashi",
    "2026-11-09": "Govardhan Puja",
    "2026-11-11": "Bhai Dooj",
    "2026-11-15": "Chhath Puja",
    "2026-11-24": "Guru Nanak Jayanti",
    "2026-12-25": "Christmas",
}

def now_india():
    return datetime.now(ZoneInfo("Asia/Kolkata"))

def meal_from_time(value=None):
    if value:
        try:
            hour = int(str(value).split(":")[0])
        except Exception:
            hour = now_india().hour
    else:
        hour = now_india().hour
    if 5 <= hour < 11: return "Breakfast"
    if 11 <= hour < 16: return "Lunch"
    if 16 <= hour < 19: return "Snack"
    return "Dinner"

def festival_for_date(date_str=None):
    d = date_str or now_india().strftime("%Y-%m-%d")
    return {"name": FESTIVALS.get(d, ""), "date": d}

def geocode_city(city):
    if not city:
        return None, None
    try:
        q = urllib.parse.urlencode({
            "name": city, "count": 1, "language": "en",
            "format": "json", "countryCode": "IN"
        })
        req = urllib.request.Request(
            "https://geocoding-api.open-meteo.com/v1/search?" + q,
            headers={"User-Agent": "Mealora/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
        results = data.get("results") or []
        if not results:
            return None, None
        return results[0].get("latitude"), results[0].get("longitude")
    except Exception as e:
        print("Geocoding failed:", e)
        return None, None

def reverse_geocode(lat, lon):
    """Return a human-readable city/region for browser coordinates."""
    if lat is None or lon is None:
        return ""
    try:
        q = urllib.parse.urlencode({
            "lat": float(lat),
            "lon": float(lon),
            "format": "json",
            "zoom": 10,
            "addressdetails": 1,
        })
        req = urllib.request.Request(
            "https://nominatim.openstreetmap.org/reverse?" + q,
            headers={"User-Agent": "Mealora/1.0 (educational project)"}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read().decode())
        address = data.get("address", {})
        return (
            address.get("city")
            or address.get("town")
            or address.get("municipality")
            or address.get("village")
            or address.get("county")
            or ""
        )
    except Exception as e:
        print("Reverse geocoding failed:", e)
        return ""

def weather_for_coords(lat, lon):
    if lat is None or lon is None:
        return {"available": False, "condition": "Unknown", "temperature": None}
    try:
        q = urllib.parse.urlencode({
            "latitude": float(lat), "longitude": float(lon),
            "current": "temperature_2m,weather_code", "timezone": "auto"
        })
        req = urllib.request.Request(
            "https://api.open-meteo.com/v1/forecast?" + q,
            headers={"User-Agent": "Mealora/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            current = json.loads(r.read().decode()).get("current", {})
        code = int(current.get("weather_code", -1))
        if code in [51,53,55,56,57]: condition = "Drizzle"
        elif code in [61,63,65,66,67,80,81,82]: condition = "Rain"
        elif code in [95,96,99]: condition = "Thunderstorm"
        elif code in [71,73,75,77,85,86]: condition = "Snow"
        elif code == 0: condition = "Clear"
        elif code in [1,2,3]: condition = "Cloudy"
        else: condition = "Unknown"
        return {
            "available": True,
            "condition": condition,
            "temperature": current.get("temperature_2m"),
            "weather_code": code
        }
    except Exception as e:
        print("Weather failed:", e)
        return {"available": False, "condition": "Unknown", "temperature": None}

def cuisine_hint(lat, lon):
    """Return a location-specific regional hint for automatic recommendations."""
    try:
        lat, lon = float(lat), float(lon)
        if 16 <= lat < 20 and 77 <= lon <= 84:
            return "Andhra"
        if 12 <= lat < 14.5 and 79 <= lon <= 80.8:
            return "Tamil Nadu"
        if 8 <= lat < 13 and 74 <= lon <= 77.5:
            return "Kerala"
        if 12 <= lat < 18 and 74 <= lon < 78.5:
            return "Karnataka"
        if 18 <= lat < 23 and 72 <= lon < 76.5:
            return "Maharashtra"
        if 20 <= lat < 24.5 and 87 <= lon < 89.5:
            return "West Bengal"
        if 27.5 <= lat < 30.5 and 76 <= lon < 78.5:
            return "Delhi"
        if 20 <= lat < 29 and 76 <= lon < 89:
            return "North Indian"
        if 21 <= lat < 28 and 85 <= lon < 94:
            return "East Indian"
    except Exception:
        pass
    return ""

def build_context(data):
    custom = str(data.get("mode", "")).lower() == "custom"

    current = now_india()
    date_value = data.get("date") if custom and data.get("date") else current.strftime("%Y-%m-%d")
    time_value = data.get("time") if custom and data.get("time") else current.strftime("%H:%M")

    # Location: custom city wins; otherwise browser coordinates.
    city = (data.get("location_city") or data.get("location")) if custom else None
    lat = data.get("latitude")
    lon = data.get("longitude")

    if city:
        geo_lat, geo_lon = geocode_city(city)
        if geo_lat is not None:
            lat, lon = geo_lat, geo_lon
    else:
        city = reverse_geocode(lat, lon)

    weather = weather_for_coords(lat, lon)
    hint = cuisine_hint(lat, lon)

    # Festival is resolved from the selected date in Custom mode
    # and today's date in Automatic mode.
    occasion = festival_for_date(date_value)

    meal = data.get("meal_type")
    if not meal or str(meal).lower() in ["auto", "automatic", "current"]:
        meal = meal_from_time(time_value)

    diet = data.get("diet") or "Vegetarian"
    if str(diet).lower() in ["auto", "automatic"]:
        diet = "Mixed"

    cuisine = data.get("cuisine")
    if not cuisine or str(cuisine).lower() in ["auto", "automatic"]:
        cuisine = hint or "Indian"

    return {
        "date": date_value,
        "time": time_value,
        "day": datetime.strptime(date_value, "%Y-%m-%d").strftime("%A"),
        "meal_type": meal,
        "diet": diet,
        "cuisine": cuisine,
        "max_time": float(data.get("max_time", 30)),
        "location_city": city or None,
        "location_name": city or None,
        "latitude": lat,
        "longitude": lon,
        "weather": weather,
        "festival": occasion,
        "mode": "custom" if custom else "automatic",
        "festival_source": "Mealora 2026 festival calendar",
    }


# ============================================================
# GEMINI RECIPE PRESENTATION LAYER
# ============================================================
# Gemini does NOT select recipes. The Mealora ML ranker selects
# the recipe first; this endpoint only translates/presents it.
# ============================================================

LANGUAGE_NAMES = {
    "English": "English",
    "தமிழ்": "Tamil",
    "తెలుగు": "Telugu",
    "ಕನ್ನಡ": "Kannada",
    "മലയാളം": "Malayalam",
    "हिन्दी": "Hindi",
    "मराठी": "Marathi",
    "বাংলা": "Bengali",
}

def _recipe_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None

def _clean_recipe_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    text = str(value).strip()

    # Many recipe datasets store lists as JSON/Python-list strings.
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return "\n".join(str(x).strip() for x in parsed if str(x).strip())
        except Exception:
            pass

    return text

def _get_recipe_for_gemini(recipe_id=None, recipe_name=None):
    if not os.path.exists(RECIPE_DATASET):
        raise FileNotFoundError(
            f"Recipe dataset not found: {RECIPE_DATASET}"
        )

    import pandas as pd

    df = get_recipe_df()

    id_col = _recipe_column(df, ["recipe_id", "id"])
    name_col = _recipe_column(df, ["recipe_name", "name", "title"])
    ingredients_col = _recipe_column(
        df, ["ingredients", "ingredient_names", "ingredient_list"]
    )
    instructions_col = _recipe_column(
        df, ["instructions", "directions", "steps", "method"]
    )

    row = None

    if recipe_id is not None and id_col:
        matches = df[df[id_col].astype(str) == str(recipe_id)]
        if not matches.empty:
            row = matches.iloc[0]

    if row is None and recipe_name and name_col:
        exact = df[
            df[name_col].astype(str).str.strip().str.lower()
            == str(recipe_name).strip().lower()
        ]
        if not exact.empty:
            row = exact.iloc[0]

    if row is None:
        raise ValueError("Selected recipe was not found in the Mealora dataset.")

    return {
        "recipe_id": str(row[id_col]) if id_col else str(recipe_id or ""),
        "recipe_name": _clean_recipe_value(row[name_col]) if name_col else str(recipe_name or ""),
        "ingredients": _clean_recipe_value(row[ingredients_col]) if ingredients_col else "",
        "instructions": _clean_recipe_value(row[instructions_col]) if instructions_col else "",
    }


# ============================================================
# FESTIVAL SPECIAL RECIPE LAYER
# ============================================================
# This is a small Mealora-owned catalogue used only to highlight a
# traditional dish on a festival day. It does NOT replace ML ranking.
# The selected recipe is still looked up from Mealora's recipe dataset.

FESTIVAL_SPECIALS = {
    "Krishna Janmashtami": [
        "Gopalkala", "Gopalkala Recipe", "Dahi Poha", "Aval Payasam",
        "Panchamrit", "Panchamrut", "Seedai", "Sweet Aval"
    ],
    "Ganesh Chaturthi": [
        "Modak", "Ukadiche Modak", "Kozhukattai", "Steamed Modak"
    ],
    "Diwali / Deepavali / Naraka Chaturdashi": [
        "Mysore Pak", "Kaju Katli", "Gulab Jamun", "Badusha", "Adhirasam"
    ],
    "Makar Sankranti / Pongal": [
        "Sakkarai Pongal", "Sweet Pongal", "Pongal", "Til Ladoo", "Sesame Ladoo"
    ],
    "Ugadi / Gudi Padwa / Chaitra Navratri": [
        "Ugadi Pachadi", "Puran Poli", "Obbattu", "Holige"
    ],
    "Holi": [
        "Gujiya", "Thandai", "Dahi Vada", "Malpua"
    ],
    "Onam / Milad-un-Nabi": [
        "Avial", "Ada Pradhaman", "Payasam", "Onam Sadya"
    ],
    "Rath Yatra": [
        "Mahaprasad", "Khichdi", "Dalma", "Poda Pitha"
    ],
    "Raksha Bandhan": [
        "Besan Ladoo", "Kaju Katli", "Rasmalai", "Peda"
    ],
    "Navratri": [
        "Sabudana Khichdi", "Kuttu Ki Roti", "Singhara Halwa", "Rajgira Ladoo"
    ],
    "Dussehra / Vijayadashami": [
        "Payasam", "Puran Poli", "Mysore Pak", "Kheer"
    ],
    "Christmas": [
        "Christmas Cake", "Plum Cake", "Fruit Cake", "Kulkul"
    ],
    "Eid / Ramzan Id": [
        "Sheer Khurma", "Vermicelli Kheer", "Biryani", "Haleem"
    ],
}


def _normalize_lookup_text(value):
    import re
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _split_recipe_lines(value):
    """Turn dataset list/JSON/newline/semicolon recipe fields into a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    # Most Mealora datasets use newline-separated values after cleaning.
    lines = [x.strip(" •\t") for x in text.replace("\r", "\n").split("\n")]
    lines = [x for x in lines if x]
    if len(lines) > 1:
        return lines

    # Fall back to semicolon separation when a row stores a single-line list.
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    return [text]


def _get_festival_special(festival_name):
    """Find a festival-special recipe already present in Mealora's dataset."""
    if not festival_name or not os.path.exists(RECIPE_DATASET):
        return None

    candidates = FESTIVAL_SPECIALS.get(festival_name, [])
    if not candidates:
        # Handles combined calendar labels such as 'Sharad Navratri Begins'.
        if "Navratri" in festival_name:
            candidates = FESTIVAL_SPECIALS.get("Navratri", [])
        elif "Diwali" in festival_name or "Deepavali" in festival_name:
            candidates = FESTIVAL_SPECIALS.get("Diwali / Deepavali / Naraka Chaturdashi", [])
        elif "Eid" in festival_name:
            candidates = FESTIVAL_SPECIALS.get("Eid / Ramzan Id", [])

    if not candidates:
        return None

    import pandas as pd
    df = get_recipe_df()
    id_col = _recipe_column(df, ["recipe_id", "id"])
    name_col = _recipe_column(df, ["recipe_name", "name", "title"])
    ingredients_col = _recipe_column(df, ["ingredients", "ingredient_names", "ingredient_list"])
    instructions_col = _recipe_column(df, ["instructions", "directions", "steps", "method"])

    if not name_col:
        return None

    names = df[name_col].fillna("").astype(str)
    normalized_names = names.map(_normalize_lookup_text)

    # Prefer exact title matches, then safe whole-title substring matches.
    for candidate in candidates:
        target = _normalize_lookup_text(candidate)
        if not target:
            continue
        exact_mask = normalized_names == target
        matches = df[exact_mask]
        if matches.empty:
            contains_mask = normalized_names.str.contains(target, regex=False, na=False)
            matches = df[contains_mask]
        if matches.empty:
            continue

        row = matches.iloc[0]
        return {
            "recipe_id": str(row[id_col]) if id_col else "",
            "recipe_name": _clean_recipe_value(row[name_col]),
            "ingredients": _split_recipe_lines(_clean_recipe_value(row[ingredients_col]) if ingredients_col else ""),
            "instructions": _split_recipe_lines(_clean_recipe_value(row[instructions_col]) if instructions_col else ""),
            "festival": festival_name,
            "special_reason": f"Traditional {festival_name} festival recipe",
        }

    return None

def _gemini_translate_recipe(recipe, language):
    """Translate ONLY the selected Mealora recipe into the requested language."""
    language = language if language in LANGUAGE_NAMES else "English"
    language_name = LANGUAGE_NAMES[language]

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to backend/.env and restart Flask."
        )

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "Gemini SDK is not installed. Run: pip install google-genai"
        )

    client = genai.Client(api_key=api_key)

    if language == "English":
        language_rules = """
- Keep the recipe in clear natural English.
- Do not change the recipe facts, quantities, ingredients, or steps.
"""
    else:
        language_rules = f"""
- Translate EVERY user-visible word into {language_name} wherever a natural translation exists.
- The recipe name MUST be in {language_name}.
- EVERY ingredient line MUST be in {language_name}.
- EVERY instruction MUST be a complete natural cooking sentence in {language_name}.
- Do NOT produce English sentences with a few {language_name} words inserted.
- Do NOT produce transliterated English sentences such as '3 tomatoes diced into large pieces'.
- Do NOT leave English phrases such as 'diced into large pieces', 'as required', 'mix well',
  'heat oil', etc. inside a {language_name} sentence.
- Food names that are commonly used in India may be retained, but write them in the target
  script where practical (for example, Telugu should use Telugu script rather than English).
- For Telugu, Tamil, Kannada, Malayalam, Hindi, Marathi, and Bengali, write the response
  using that language's native script. Do NOT write Telugu/Tamil/etc. as English transliteration.
- Preserve quantities and units accurately. Translate the surrounding words naturally.
- Make the instructions sound like a person explaining the recipe aloud in {language_name}.
- Do NOT invent, remove, or reorder recipe facts.
"""

    prompt = f"""
You are Mealora's multilingual recipe presentation layer.

Mealora's own recommendation/ML engine has ALREADY selected this exact recipe.
Your ONLY job is to present that exact recipe in {language_name}.
You are NOT the recipe recommender and must NOT select another recipe.

STRICT LANGUAGE REQUIREMENT:
{language_rules}

IMPORTANT OUTPUT REQUIREMENTS:
1. Return JSON matching the schema exactly.
2. 'language' must be exactly '{language}'.
3. 'recipe_name', every item in 'ingredients', and every item in 'instructions'
   must be written in the requested language.
4. Do not mix English and the requested language in the same sentence unless a food
   name genuinely has no useful translation. Even then, prefer the target script.
5. Keep the meaning, quantities, and cooking sequence from the source exactly.
6. Do not add nutrition values, health claims, substitutions, or new ingredients.
7. The instructions should be natural spoken cooking language, suitable for Read Aloud.

SOURCE RECIPE FROM MEALORA DATASET:
Recipe ID: {recipe["recipe_id"]}
Recipe Name: {recipe["recipe_name"]}

Ingredients:
{recipe["ingredients"]}

Instructions:
{recipe["instructions"]}
"""

    schema = {
        "type": "object",
        "properties": {
            "recipe_name": {"type": "string"},
            "language": {"type": "string"},
            "ingredients": {"type": "array", "items": {"type": "string"}},
            "instructions": {"type": "array", "items": {"type": "string"}},
            "note": {"type": "string"},
        },
        "required": ["recipe_name", "language", "ingredients", "instructions", "note"],
    }

    # Gemini 3.6 Flash no longer accepts the old temperature/top_p/top_k parameters.
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    text = response.text
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Gemini returned invalid structured output: {e}")

    # Never trust a model response to silently change the requested language.
    returned_language = str(result.get("language", "")).strip()
    if returned_language != language:
        raise RuntimeError(
            f"Gemini returned language '{returned_language}' instead of requested '{language}'."
        )

    for key in ("recipe_name", "ingredients", "instructions"):
        if key not in result:
            raise RuntimeError(f"Gemini response is missing '{key}'.")

    if not isinstance(result["ingredients"], list) or not isinstance(result["instructions"], list):
        raise RuntimeError("Gemini returned an invalid ingredient/instruction structure.")

    result["recipe_id"] = recipe["recipe_id"]
    result["source"] = "Mealora recipe dataset + Gemini presentation"
    return result

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Mealora ML Recommendation API is running",
        "current_time": now_india().isoformat(),
        "current_meal_type": meal_from_time(),
        "festival": festival_for_date()
    })

@app.route("/api/context", methods=["POST"])
def context():
    try:
        return jsonify({"success": True, "context": build_context(request.get_json() or {})})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500















# ===== MEALORA HOUSEHOLD WEEKLY PLANNER START =====
# Household-first planner:
# pantry -> role-compatible candidates -> regional/context scoring
# -> practical meal composition -> compact NSGA-II optimization.


# ===== MEALORA WEEKLY PLANNER V3 START =====

@app.route("/api/weekly-plan", methods=["POST"])
def weekly_plan():
    """Mealora's practical 7-day household planner.

    The planner deliberately separates:
      1. pantry-compatible candidate retrieval,
      2. ML suitability scoring,
      3. regional/context re-ranking,
      4. simple Indian meal composition.

    It does NOT optimize for a large number of unique recipes. A normal
    household can repeat a dish after a short cooldown; the goal is realistic
    meals, not artificial uniqueness.
    """
    try:
        data = request.get_json() or {}
        household = data.get("household") or {}
        adults = max(int(household.get("adults", 0) or 0), 0)
        children = max(int(household.get("children", 0) or 0), 0)
        seniors = max(int(household.get("seniors", 0) or 0), 0)
        if adults + children + seniors <= 0:
            return jsonify({"success": False, "error": "Add at least one household member."}), 400

        mode = str(data.get("mode", "automatic") or "automatic").lower()
        diet = str(data.get("diet", "Mixed") or "Mixed")
        cuisine = str(data.get("cuisine", "Auto") or "Auto")
        max_time = float(data.get("max_time", 60) or 60)

        ctx = build_context({
            "mode": mode,
            "date": data.get("date"),
            "location_city": data.get("location_city"),
            "location": data.get("location_city"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "meal_type": "Automatic",
            "diet": diet,
            "cuisine": cuisine,
            "max_time": max_time,
        })
        if diet.lower() in {"auto", "automatic"}:
            diet = ctx.get("diet", "Mixed")
        if cuisine.lower() in {"auto", "automatic"}:
            cuisine = ctx.get("cuisine", "Indian")

        # ---------- pantry ----------
        today = now_india().date()
        pantry_names, pantry_expiry_names = [], []
        for item in data.get("pantry") or []:
            if not isinstance(item, dict):
                continue
            name = item.get("normalized_name") or item.get("ingredient_name") or item.get("name")
            name = _weekly_norm(name)
            if not name:
                continue
            pantry_names.append(name)
            expiry = item.get("expiry_date")
            if expiry:
                try:
                    days = (datetime.strptime(str(expiry)[:10], "%Y-%m-%d").date() - today).days
                    if days <= 7:
                        pantry_expiry_names.append(name)
                except Exception:
                    pass
        pantry_names = list(dict.fromkeys(pantry_names))
        pantry_expiry_names = list(dict.fromkeys(pantry_expiry_names))
        if not pantry_names:
            return jsonify({"success": False, "error": "Your pantry is empty. Add ingredients before generating the weekly plan."}), 400

        pantry_tokens = set().union(*(_weekly_tokens(x) for x in pantry_names)) if pantry_names else set()
        expiry_tokens = set().union(*(_weekly_tokens(x) for x in pantry_expiry_names)) if pantry_expiry_names else set()
        pantry_exact = {_weekly_norm(x): x for x in pantry_names}
        pantry_token_to_names = {}
        for p in pantry_names:
            for token in _weekly_tokens(p):
                pantry_token_to_names.setdefault(token, []).append(p)

        staples = {
            "salt", "sugar", "water", "oil", "cooking oil", "sunflower oil",
            "groundnut oil", "coconut oil", "mustard oil", "sesame oil",
            "olive oil", "ghee", "butter", "black pepper", "turmeric",
            "turmeric powder", "red chili", "red chili powder", "coriander powder",
            "cumin powder", "garam masala", "hing", "asafoetida", "ginger garlic paste"
        }
        staple_norms = {_weekly_norm(x) for x in staples}

        # ---------- cached 18,568-row recipe representation ----------
        cache = _build_weekly_recipe_cache()
        records = cache["records"]
        token_index = cache["token_index"]
        quality_indices = cache.get("quality_indices", list(range(len(records))))
        if not records:
            return jsonify({"success": False, "error": "Mealora's recipe dataset contains no usable recipes."}), 500

        # Retrieve recipes sharing pantry tokens. Add only a bounded set of
        # high-quality fallback recipes so the request stays fast.
        candidate_ids = set()
        for token in pantry_tokens:
            candidate_ids.update(token_index.get(token, ()))

        if len(candidate_ids) < 900:
            candidate_ids.update(quality_indices[:1200])

        # Rank candidates cheaply before doing detailed pantry scoring.
        def retrieval_score(i):
            r = records[i]
            overlap = len(r["_all_tokens"] & pantry_tokens)
            return (overlap, r.get("_quality", 0.0), -len(r["ingredients"]))

        candidate_ids = sorted(candidate_ids, key=retrieval_score, reverse=True)[:3200]

        # ---------- context / regional scoring ----------
        city = _weekly_norm(ctx.get("location_city"))
        requested_cuisine = _weekly_norm(cuisine)
        city_terms = {
            "hyderabad": ["telangana", "andhra pradesh", "andhra", "telugu", "hyderabadi", "deccan"],
            "bengaluru": ["karnataka", "kannada", "udupi"],
            "bangalore": ["karnataka", "kannada", "udupi"],
            "chennai": ["tamil nadu", "tamil", "chettinad"],
            "vijayawada": ["andhra pradesh", "andhra", "telugu"],
            "visakhapatnam": ["andhra pradesh", "andhra", "telugu"],
            "kochi": ["kerala", "malabar"],
            "ernakulam": ["kerala", "malabar"],
            "thiruvananthapuram": ["kerala"],
            "mumbai": ["maharashtra", "marathi", "konkani"],
            "pune": ["maharashtra", "marathi"],
            "ahmedabad": ["gujarat", "gujarati"],
            "surat": ["gujarat", "gujarati"],
            "jaipur": ["rajasthan", "rajasthani"],
            "lucknow": ["uttar pradesh", "awadhi"],
            "delhi": ["delhi", "punjab", "punjabi", "mughlai"],
            "new delhi": ["delhi", "punjab", "punjabi", "mughlai"],
            "kolkata": ["west bengal", "bengali"],
            "bhubaneswar": ["odisha", "odia"],
            "patna": ["bihar", "bihari"],
            "goa": ["goa", "konkani"],
        }
        cuisine_aliases = {
            "south indian": ["south indian", "tamil", "tamil nadu", "telugu", "telangana", "andhra", "andhra pradesh", "kannada", "karnataka", "kerala", "malabar", "udupi", "chettinad"],
            "north indian": ["north indian", "punjabi", "rajasthani", "rajasthan", "awadhi", "uttar pradesh", "delhi", "mughlai"],
            "west indian": ["west indian", "maharashtra", "marathi", "gujarati", "gujarat", "goan", "konkani"],
            "east indian": ["east indian", "bengali", "west bengal", "odia", "odisha", "bihari", "bihar"],
        }

        def region_score(r):
            text = " ".join([
                r.get("state", ""), r.get("region", ""),
                r.get("cuisine", ""), r.get("recipe_name", "")
            ]).lower()
            if not text:
                return 0.45
            city_match = any(t in text for t in city_terms.get(city, []))
            cuisine_terms = cuisine_aliases.get(requested_cuisine, [])
            cuisine_match = bool(cuisine_terms and any(t in text for t in cuisine_terms))
            if city_match and cuisine_match:
                return 1.0
            if city_match:
                return 0.95
            if cuisine_match:
                return 0.85
            if requested_cuisine == "south indian" and any(t in text for t in (
                "idli", "dosa", "dosai", "sambar", "sambhar", "pongal",
                "upma", "uttapam", "appam", "puttu", "pesarattu", "adai",
                "poriyal", "thoran", "rasam", "pappu", "chutney"
            )):
                return 0.85
            # Keep an unknown recipe neutral rather than calling it 0%.
            return 0.45

        def diet_score(r):
            req = _weekly_norm(diet)
            text = _weekly_norm(r["recipe_name"] + " " + " ".join(r["ingredients"]))
            nonveg = any(x in text for x in ("chicken", "mutton", "lamb", "fish", "prawn", "prawns", "shrimp", "meat", "seafood", "egg", "keema"))
            if req in {"", "mixed", "any", "auto", "automatic"}:
                return 0.75
            if req in {"vegetarian", "veg", "pure vegetarian"}:
                return 0.0 if nonveg else 1.0
            if req == "vegan":
                dairy = any(x in text for x in ("milk", "curd", "yogurt", "ghee", "butter", "paneer", "cheese", "cream", "egg"))
                return 0.0 if (nonveg or dairy) else 1.0
            if req in {"non vegetarian", "non-vegetarian", "non veg", "non-veg", "nonvegetarian"}:
                return 1.0 if nonveg else 0.45
            return 0.65

        def cuisine_score(r):
            req = requested_cuisine
            if req in {"", "auto", "automatic", "mixed", "any", "indian"}:
                return 0.75
            text = " ".join([r.get("cuisine", ""), r.get("state", ""), r.get("region", "")]).lower()
            terms = cuisine_aliases.get(req, [req])
            return 1.0 if any(t in text for t in terms) else 0.40

        def pantry_score(r):
            ingredients = r["ingredients"]
            matched, missing, missing_core, expiry_matched = [], [], [], []
            core_total = core_hits = 0
            for ing, ing_norm, ing_tokens in zip(ingredients, r["_ingredient_norms"], r["_ingredient_tokens"]):
                is_staple = ing_norm in staple_norms
                if not is_staple:
                    core_total += 1
                match = pantry_exact.get(ing_norm)
                if match is None and ing_tokens:
                    possible = []
                    for token in ing_tokens & pantry_tokens:
                        possible.extend(pantry_token_to_names.get(token, []))
                    if possible:
                        match = max(set(possible), key=lambda p: len(ing_tokens & _weekly_tokens(p)))
                if match:
                    matched.append(match)
                    if not is_staple:
                        core_hits += 1
                    if ing_tokens & expiry_tokens:
                        expiry_matched.append(match)
                else:
                    missing.append(ing)
                    if not is_staple:
                        missing_core.append(ing)
            all_cov = len(matched) / max(len(ingredients), 1)
            core_cov = core_hits / core_total if core_total else all_cov
            expiry_cov = len(set(expiry_matched)) / max(len(pantry_expiry_names), 1) if pantry_expiry_names else 0.0
            return float(np.clip(0.75 * core_cov + 0.25 * all_cov, 0, 1)), list(dict.fromkeys(matched)), missing, missing_core, expiry_cov

        def nutrition_fit(r):
            # Prefer an explicit normalized nutrition score when the dataset
            # provides one. Otherwise derive a conservative score from the
            # available macro fields. If nutrition data is genuinely absent,
            # use a neutral 0.50 rather than falsely reporting 0%.
            explicit = r.get("_nutrition_raw")
            if explicit is not None and np.isfinite(explicit):
                return float(np.clip(explicit, 0, 1))

            parts = []
            protein = r.get("_protein")
            fibre = r.get("_fibre")
            calories = r.get("_calories")
            fat = r.get("_fat")
            sodium = r.get("_sodium")

            if protein is not None and np.isfinite(protein):
                parts.append(float(np.clip(protein / 25.0, 0, 1)))
            if fibre is not None and np.isfinite(fibre):
                parts.append(float(np.clip(fibre / 8.0, 0, 1)))
            if calories is not None and np.isfinite(calories):
                # A broad per-serving middle range is preferable to rewarding
                # either extremely high or extremely low calorie values.
                parts.append(float(np.clip(1.0 - abs(calories - 450.0) / 450.0, 0, 1)))
            if fat is not None and np.isfinite(fat):
                parts.append(float(np.clip(1.0 - max(fat - 15.0, 0) / 35.0, 0, 1)))
            if sodium is not None and np.isfinite(sodium):
                parts.append(float(np.clip(1.0 - max(sodium - 500.0, 0) / 1500.0, 0, 1)))

            return float(np.clip(np.mean(parts), 0, 1)) if parts else 0.50

        scored = []
        for idx in candidate_ids:
            r = records[idx]
            p, matched, missing, missing_core, expiry = pantry_score(r)
            if p <= 0:
                continue

            cook = r.get("cook_time")
            time_score = (
                1.0
                if cook is None
                else float(np.clip(
                    1.0 - max(cook - max_time, 0) / max(max_time, 1),
                    0, 1
                ))
            )

            region = region_score(r)
            diet_fit = diet_score(r)
            cuisine_fit = cuisine_score(r)

            # This is Mealora's ACTUAL weekly context score. The old code
            # displayed the dataset's raw context_score (often 0.05/0.10),
            # which is why the UI showed ~9%. Here context means the current
            # user's region + diet + cuisine + time + learned recipe context.
            raw_context = float(np.clip(r.get("_context_raw", 0.5), 0, 1))
            dynamic_context = float(np.clip(
                0.35 * region +
                0.20 * diet_fit +
                0.15 * cuisine_fit +
                0.15 * time_score +
                0.15 * raw_context,
                0, 1
            ))

            scored.append({
                **r,
                "pantry": p,
                "core_pantry": p,
                "expiry": expiry,
                "matched_pantry": matched,
                "missing": missing,
                "missing_core": missing_core,
                "region_score": region,
                "diet_score": diet_fit,
                "cuisine_score": cuisine_fit,
                "time_score": time_score,
                "context_score": dynamic_context,
                "nutrition_score": nutrition_fit(r),
            })

        if not scored:
            return jsonify({"success": False, "error": "Mealora could not find pantry-compatible recipes. Check the pantry ingredient names."}), 400

        # Prefer regional candidates when enough exist. This makes regional
        # context a real constraint instead of a decorative percentage.
        regional_candidates = [r for r in scored if r["region_score"] >= 0.85]
        planning_pool = regional_candidates if len(regional_candidates) >= 40 else scored

        # ---------- real trained ML suitability ----------
        model = _load_weekly_ml_model()
        feature_names = ["pantry_match_pct", "missing_ingredient_count", "expiry_score", "meal_match", "diet_match", "cuisine_match", "time_match", "context_score", "ingredient_count", "cooking_time_minutes"]
        X_rows = []
        for r in planning_pool:
            X_rows.append([r["core_pantry"], len(r["missing_core"]), r["expiry"], 0.70, r["diet_score"], r["cuisine_score"], r["time_score"], r["context_score"], len(r["ingredients"]), r.get("cook_time") or 30])
        if model is not None:
            try:
                X = pd.DataFrame(X_rows, columns=feature_names)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X)
                    ml_values = probs[:, 1] if getattr(probs, "ndim", 1) == 2 and probs.shape[1] >= 2 else probs.reshape(-1)
                else:
                    ml_values = model.predict(X)
                ml_values = np.clip(np.nan_to_num(np.asarray(ml_values, dtype=float), nan=0.5), 0, 1)
            except Exception as exc:
                print("Weekly ML prediction warning:", exc)
                ml_values = np.array([0.5 + 0.3*r["core_pantry"] + 0.1*r["region_score"] + 0.1*r["diet_score"] for r in planning_pool])
        else:
            ml_values = np.array([0.5 + 0.3*r["core_pantry"] + 0.1*r["region_score"] + 0.1*r["diet_score"] for r in planning_pool])

        for r, ml in zip(planning_pool, ml_values):
            r["ml"] = float(np.clip(ml, 0, 1))
            r["_base_score"] = (
                0.28 * r["core_pantry"] +
                0.28 * r["ml"] +
                0.18 * r["region_score"] +
                0.10 * r["expiry"] +
                0.07 * r["diet_score"] +
                0.04 * r["cuisine_score"] +
                0.03 * r["context_score"] +
                0.02 * r["time_score"]
            )

        # ---------- semantic meal roles ----------
        def n(r):
            return _weekly_norm(r["recipe_name"])

        def has_any(text, terms):
            return any(t in text for t in terms)

        classic_breakfast = ("idli", "dosa", "dosai", "upma", "pongal", "poha", "uttapam", "vada", "appam", "puttu", "pesarattu", "adai", "paratha", "parantha", "puri", "poori", "thepla", "dhokla")
        chutneys = ("chutney", "thogayal", "thuvaiyal", "pachadi")
        breakfast_sides = ("sambar", "sambhar", "gothsu", "saagu", "kurma", "korma", "potato masala", "aloo masala")
        rice = ("plain rice", "white rice", "steamed rice", "cooked rice", "rice")
        roti = ("chapati", "roti", "phulka")
        dals = ("dal", "dhal", "pappu", "sambar", "sambhar", "rasam", "kadhi", "kootu")
        vegetables = ("poriyal", "palya", "thoran", "kootu", "avial", "sabzi", "subzi", "bhaji", "vegetable fry", "veg fry", "beans fry", "potato fry", "cabbage", "carrot", "beetroot", "okra", "bhindi", "brinjal", "eggplant", "cauliflower", "peas")
        proteins = ("chicken", "mutton", "lamb", "fish", "prawn", "prawns", "shrimp", "egg curry", "paneer", "kofta", "meat", "seafood")
        snacks = ("snack", "pakora", "bajji", "bonda", "samosa", "sundal", "punugulu", "garelu", "murukku", "mixture", "cutlet", "tea", "coffee", "banana chips", "unniyappam")

        def clean_display(name):
            s = str(name).strip()
            s = re.sub(r"\s+recipe\s*$", "", s, flags=re.I)
            return s.strip(" -") or str(name)

        def family(name):
            x = _weekly_norm(name)
            x = re.sub(r"\brecipe\b", "", x)
            x = re.sub(r"\bwith\b.*$", "", x).strip()
            x = re.sub(r"\busing\b.*$", "", x).strip()
            x = re.sub(r"\bstyle\b", "", x).strip()
            return x

        canonical = {
            "idli", "masala dosa", "dosa", "plain dosa", "rava dosa", "uttapam", "pongal", "ven pongal",
            "upma", "poha", "pesarattu", "adai", "appam", "puttu", "chapati", "roti", "phulka",
            "plain rice", "steamed rice", "white rice", "dal", "toor dal", "moong dal", "sambar",
            "coconut chutney", "tomato chutney", "peanut chutney", "mint chutney", "potato fry", "beans poriyal",
            "vegetable poriyal", "curd rice", "lemon rice"
        }

        def quality_bonus(r):
            x = n(r)
            b = 0.0
            if x in canonical:
                b += 0.32
            elif family(x) in canonical:
                b += 0.20
            if len(x.split()) <= 7:
                b += 0.05
            if any(t in x for t in ("using", "pressure cooker", "instant pot", "leftover", "fusion", "cheese toast")):
                b -= 0.15
            return b

        def role_match(r, role):
            x = n(r)
            meal = _weekly_norm(r.get("meal_type", ""))
            if role == "breakfast":
                return ("breakfast" in meal or has_any(x, classic_breakfast))
            if role == "chutney":
                return has_any(x, chutneys)
            if role == "breakfast_side":
                return has_any(x, breakfast_sides)
            if role == "lunch_main":
                return has_any(x, rice + roti) and not has_any(x, proteins + ("pulao", "pulav", "biryani", "fried rice"))
            if role == "lunch_side":
                return has_any(x, dals + vegetables) and not has_any(x, proteins)
            if role == "dinner_main":
                return has_any(x, roti + rice) and not has_any(x, proteins + ("pulao", "pulav", "biryani", "fried rice"))
            if role == "dinner_side":
                return has_any(x, proteins + dals + ("paneer", "gravy", "curry", "korma", "kurma"))
            if role == "snack":
                return ("snack" in meal or "starter" in meal or "appetizer" in meal or has_any(x, snacks))
            return False

        def rank_role(role, preferred=()):
            vals = [r for r in planning_pool if role_match(r, role)]
            if not vals and role != "snack":
                vals = list(planning_pool)
            for r in vals:
                r["_role_rank"] = r["_base_score"] + quality_bonus(r)
                if preferred and has_any(n(r), preferred):
                    r["_role_rank"] += 0.16
            vals.sort(key=lambda r: (r["_role_rank"], r["region_score"], r["pantry"]), reverse=True)
            return vals[:250]

        pools = {
            "breakfast": rank_role("breakfast", classic_breakfast),
            "chutney": rank_role("chutney", ("coconut chutney", "peanut chutney", "tomato chutney")),
            "breakfast_side": rank_role("breakfast_side", ("sambar", "sambhar", "gothsu")),
            "lunch_main": rank_role("lunch_main", rice if any("rice" in p for p in pantry_names) else roti),
            "lunch_side": rank_role("lunch_side", dals + vegetables),
            "dinner_main": rank_role("dinner_main", roti if any(any(t in p for t in ("atta", "wheat flour", "chapati")) for p in pantry_names) else rice),
            "dinner_side": rank_role("dinner_side", proteins if _weekly_norm(diet) in {"non vegetarian", "non-vegetarian", "non veg", "non-veg", "nonvegetarian"} else ("paneer", "dal", "gravy", "curry")),
            "snack": rank_role("snack"),
        }

        # Regional snack vocabulary. These are preferences, not hardcoded outputs.
        regional_snacks = {
            "hyderabad": ("punugulu", "mirchi bajji", "garelu", "bonda"),
            "vijayawada": ("punugulu", "garelu", "bajji"),
            "visakhapatnam": ("punugulu", "garelu", "bajji"),
            "chennai": ("sundal", "murukku", "bajji", "bonda"),
            "bengaluru": ("bonda", "bajji", "sundal", "murukku"),
            "bangalore": ("bonda", "bajji", "sundal", "murukku"),
            "kochi": ("banana chips", "unniyappam", "bonda"),
        }
        snack_pref = regional_snacks.get(city, ("punugulu", "bajji", "bonda", "sundal", "murukku"))
        snack_pool = rank_role("snack", snack_pref)

        # Realistic variety: do NOT require 21 unique recipes, but also do
        # NOT let the same breakfast repeat every day. A dish family gets a
        # short three-day cooldown. After that, normal household repetition
        # is allowed.
        recent_day_ids = []
        recent_role_families = {
            "breakfast": [],
            "chutney": [],
            "lunch_main": [],
            "lunch_side": [],
            "dinner_main": [],
            "dinner_side": [],
            "snack": [],
        }

        def choose(pool, excluded_ids=(), preferred=(), role=None, day_index=0):
            excluded = {str(x) for x in excluded_ids}
            if not pool:
                return None

            vals = [r for r in pool if str(r["recipe_id"]) not in excluded]
            if not vals:
                vals = list(pool)

            # Preferred terms are a bonus, not a hard filter. Hard filtering
            # was one reason the same Masala Dosa kept winning every morning.
            recent_families = set(recent_role_families.get(role, [])[-3:]) if role else set()

            def candidate_score(r):
                pref_bonus = 0.12 if preferred and has_any(n(r), preferred) else 0.0
                return (
                    r["_role_rank"]
                    + quality_bonus(r)
                    + pref_bonus,
                    r["region_score"],
                    r["pantry"],
                )

            fresh = [
                r for r in vals
                if (family(clean_display(r["recipe_name"])) not in recent_families)
            ]

            if fresh:
                vals = fresh

            vals = sorted(vals, key=candidate_score, reverse=True)

            # Rotate through the strongest candidates instead of always taking
            # the single maximum. This creates genuine day-to-day variety while
            # preserving the learned/pantry/regional ranking.
            window = vals[:min(8, len(vals))]
            return window[day_index % len(window)]

        def serialize(r, role):
            return {
                "recipe_id": str(r["recipe_id"]),
                "recipe_name": clean_display(r["recipe_name"]),
                "suitability_score": round(float(r["ml"]), 4),
                "pantry_score": round(float(r["pantry"]), 4),
                "expiry_score": round(float(r["expiry"]), 4),
                "context_score": round(float(r["context_score"]), 4),
                "region_score": round(float(r["region_score"]), 4),
                "regional_score": round(float(r["region_score"]), 4),
                "region": r.get("state") or r.get("region") or "",
                "nutrition_score": round(float(r["nutrition_score"]), 4),
                "matched_pantry": list(r["matched_pantry"])[:10],
                "missing_ingredients": list(r["missing"])[:8],
                "ingredients": list(r["ingredients"]),
                "component_role": role,
            }

        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        plan = []
        for day_index, day in enumerate(days):
            day_used = set().union(*(recent_day_ids[-2:] or [set()]))

            b = choose(
                pools["breakfast"], day_used, classic_breakfast,
                role="breakfast", day_index=day_index
            )
            b_side = choose(
                pools["chutney"],
                day_used | {str(b["recipe_id"])},
                ("coconut chutney", "peanut chutney", "tomato chutney"),
                role="chutney", day_index=day_index
            )
            l = choose(
                pools["lunch_main"],
                day_used | {str(b["recipe_id"]), str(b_side["recipe_id"])},
                rice if any("rice" in p for p in pantry_names) else roti,
                role="lunch_main", day_index=day_index
            )
            l_side = choose(
                pools["lunch_side"],
                day_used | {str(b["recipe_id"]), str(b_side["recipe_id"]), str(l["recipe_id"])},
                dals + vegetables,
                role="lunch_side", day_index=day_index
            )
            d = choose(
                pools["dinner_main"],
                day_used | {str(x["recipe_id"]) for x in (b, b_side, l, l_side)},
                roti if any("atta" in p or "wheat flour" in p for p in pantry_names) else rice,
                role="dinner_main", day_index=day_index
            )
            d_side = choose(
                pools["dinner_side"],
                day_used | {str(x["recipe_id"]) for x in (b, b_side, l, l_side, d)},
                proteins if _weekly_norm(diet) in {"non vegetarian", "non-vegetarian", "non veg", "non-veg", "nonvegetarian"} else ("paneer", "dal", "gravy", "curry"),
                role="dinner_side", day_index=day_index
            )

            snack = choose(
                pools["snack"],
                day_used | {str(x["recipe_id"]) for x in (b, b_side, l, l_side, d, d_side)},
                snack_pref,
                role="snack", day_index=day_index
            )

            chosen = [b, b_side, l, l_side, d, d_side] + ([snack] if snack else [])
            recent_day_ids.append({str(r["recipe_id"]) for r in chosen})

            for role_name, recipe in [
                ("breakfast", b),
                ("chutney", b_side),
                ("lunch_main", l),
                ("lunch_side", l_side),
                ("dinner_main", d),
                ("dinner_side", d_side),
                ("snack", snack),
            ]:
                if recipe is not None:
                    recent_role_families[role_name].append(
                        family(clean_display(recipe["recipe_name"]))
                    )

            plan.append({
                "day": day,
                "theme": "Pantry-first regional household meal plan",
                "breakfast_main": serialize(b, "Breakfast Main"),
                "breakfast_side": serialize(b_side, "Breakfast Side"),
                "lunch_main": serialize(l, "Lunch Main"),
                "lunch_side": serialize(l_side, "Lunch Side"),
                "dinner_main": serialize(d, "Dinner Main"),
                "dinner_side": serialize(d_side, "Dinner Side"),
                "snack": serialize(snack, "Regional Snack"),
            })

        meals = [x[k] for x in plan for k in ("breakfast_main", "breakfast_side", "lunch_main", "lunch_side", "dinner_main", "dinner_side") if x.get(k)]
        def avg(key, default=0.5):
            vals = [float(x.get(key, default)) for x in meals]
            return float(np.mean(vals)) if vals else default

        return jsonify({
            "success": True,
            "context": ctx,
            "plan": plan,
            "summary": {
                "meals_planned": 21,
                "pantry_utilization": round(avg("pantry_score") * 100, 1),
                "expiry_utilization": round(avg("expiry_score", 0) * 100, 1),
                "nutrition_score": round(avg("nutrition_score") * 100, 1),
                "context_score": round(avg("context_score") * 100, 1),
                "regional_score": round(avg("region_score", 0.45) * 100, 1),
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recipe-assistant", methods=["POST"])
def recipe_assistant():
    try:
        data = request.get_json() or {}

        recipe = _get_recipe_for_gemini(
            recipe_id=data.get("recipe_id"),
            recipe_name=data.get("recipe_name"),
        )

        language = data.get("language", "English")

        # English NEVER calls Gemini.
        if str(language).strip().lower() == "english":
            result = {
                "recipe_id": recipe.get("recipe_id"),
                "recipe_name": recipe.get("recipe_name"),
                "language": "English",
                "ingredients": recipe.get("ingredients", []),
                "instructions": recipe.get("instructions", []),
                "note": "Recipe presented directly from Mealora's recipe dataset.",
                "source": "Mealora recipe dataset",
            }

            return jsonify({
                "success": True,
                "assistant_role": "recipe_presentation",
                "recipe": result,
            })

        # Non-English: Gemini with one retry.
        last_error = None

        for attempt in range(2):
            try:
                result = _gemini_translate_recipe(recipe, language)

                return jsonify({
                    "success": True,
                    "assistant_role": "translation_and_recipe_presentation_only",
                    "recipe": result,
                })

            except Exception as exc:
                last_error = exc
                print(
                    f"Gemini recipe presentation attempt "
                    f"{attempt + 1}/2 failed: {exc}"
                )

                if attempt == 0:
                    import time
                    time.sleep(1.0)

        # Gemini unavailable: return original English recipe.
        fallback = {
            "recipe_id": recipe.get("recipe_id"),
            "recipe_name": recipe.get("recipe_name"),
            "language": "English",
            "ingredients": recipe.get("ingredients", []),
            "instructions": recipe.get("instructions", []),
            "note": (
                "The selected language service is temporarily unavailable. "
                "Mealora is showing the original recipe in English."
            ),
            "source": "Mealora recipe dataset (English fallback)",
            "translation_error": str(last_error) if last_error else "",
        }

        return jsonify({
            "success": True,
            "assistant_role": "recipe_presentation_with_translation_fallback",
            "recipe": fallback,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500


@app.route("/api/recommendations", methods=["POST"])
def recommendations():
    try:
        data = request.get_json() or {}
        ctx = build_context(data)

        result = recommend(
            reference_recipe=data.get("reference_recipe", "Balu shahi"),
            target_user=int(data.get("target_user", 1)),
            top_k=int(data.get("top_k", 10)),
            meal_type=ctx["meal_type"],
            diet=ctx["diet"],
            cuisine=ctx["cuisine"],
            max_time=ctx["max_time"],
            weather=ctx["weather"],
            festival=ctx["festival"],
            pantry_items=data.get("pantry") or [],
            location={
                "city": ctx["location_city"],
                "location_city": ctx["location_city"],
                "latitude": ctx["latitude"],
                "longitude": ctx["longitude"],
                "cuisine_hint": cuisine_hint(ctx["latitude"], ctx["longitude"])
            }
        )

        festival_special = _get_festival_special(ctx["festival"].get("name"))

        return jsonify({
            "success": True,
            "mode": ctx["mode"],
            "reference_recipe": data.get("reference_recipe", "Balu shahi"),
            "target_user": int(data.get("target_user", 1)),
            "context": ctx,
            "festival_special": festival_special,
            "recommendations": _json_safe(
                result.to_dict(orient="records")
            )
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# LEFTOVER INTELLIGENCE
# ============================================================
# Reuse-first logic: search the same Mealora recipe dataset for
# recipes whose ingredient list contains the reported leftover.
# This is a lightweight ingredient-compatibility algorithm, not
# a second recipe recommender that replaces the main ML engine.
# ============================================================

def _leftover_tokens(value):
    import re
    stop = {
        "leftover", "leftovers", "cooked", "cooking", "food", "some",
        "the", "and", "with", "of", "a", "an", "to", "from"
    }
    text = _normalize_lookup_text(value)
    return {x for x in text.split() if len(x) >= 3 and x not in stop}


def _ingredient_column(df):
    return _recipe_column(df, [
        "ingredients", "ingredient_names", "ingredient_list",
        "Cleaned-Ingredients", "TranslatedIngredients"
    ])


def _recipe_name_column(df):
    return _recipe_column(df, ["recipe_name", "name", "title", "final_food_name"])


def _leftover_recommendations(leftover, top_k=6):
    import pandas as pd
    import re

    if not os.path.exists(RECIPE_DATASET):
        raise FileNotFoundError(f"Recipe dataset not found: {RECIPE_DATASET}")

    df = get_recipe_df()
    name_col = _recipe_name_column(df)
    ing_col = _ingredient_column(df)
    id_col = _recipe_column(df, ["recipe_id", "id"])

    if not name_col or not ing_col:
        raise RuntimeError("Recipe dataset does not contain recipe name and ingredient columns.")

    leftover_tokens = _leftover_tokens(leftover)
    if not leftover_tokens:
        return []

    results = []
    for idx, row in df[[name_col, ing_col] + ([id_col] if id_col else [])].iterrows():
        raw = str(row[ing_col]) if row[ing_col] == row[ing_col] else ""
        if not raw:
            continue

        ingredient_text = _normalize_lookup_text(raw)
        ingredient_tokens = set(ingredient_text.split())
        if not ingredient_tokens:
            continue

        # Whole leftover phrase gets the strongest signal.
        normalized_leftover = _normalize_lookup_text(leftover)
        phrase_match = normalized_leftover and normalized_leftover in ingredient_text
        token_matches = leftover_tokens.intersection(ingredient_tokens)

        # Also allow singular/plural and simple Indian ingredient wording.
        expanded_matches = set(token_matches)
        for token in leftover_tokens:
            if token.endswith("s") and token[:-1] in ingredient_tokens:
                expanded_matches.add(token[:-1])
            elif f"{token}s" in ingredient_tokens:
                expanded_matches.add(f"{token}s")

        if not phrase_match and not expanded_matches:
            continue

        token_score = len(expanded_matches) / max(len(leftover_tokens), 1)
        score = 0.75 * token_score + (0.25 if phrase_match else 0.0)
        if score <= 0:
            continue

        name = str(row[name_col]).strip()
        if not name:
            continue

        # Prefer recipes where the leftover appears as an ingredient phrase,
        # then by the number of matched leftover terms.
        results.append({
            "recipe_id": str(row[id_col]) if id_col else str(idx),
            "recipe_name": name,
            "match_score": round(min(score, 1.0), 4),
            "matched_ingredients": sorted(expanded_matches),
            "cuisine": str(row["cuisine"]).strip() if "cuisine" in df.columns and row["cuisine"] == row["cuisine"] else "",
            "meal_type": str(row["meal_type"]).strip() if "meal_type" in df.columns and row["meal_type"] == row["meal_type"] else "",
        })

    results.sort(key=lambda x: (-x["match_score"], x["recipe_name"].lower()))

    # Remove duplicate recipe names caused by multiple source records.
    unique = []
    seen = set()
    for item in results:
        key = _normalize_lookup_text(item["recipe_name"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= max(1, min(int(top_k), 20)):
            break

    return unique


@app.route("/api/leftover-recommendations", methods=["POST"])
def leftover_recommendations():
    """
    Leftover-specific ML recommendation.

    1. TF-IDF + cosine similarity finds recipes semantically
       related to the leftover.
    2. Ingredient overlap measures actual leftover coverage.
    3. Mealora's trained suitability model (HGB/RF when
       available) provides an additional ML suitability signal.
    """
    try:
        import pandas as pd
        import numpy as np

        data = request.get_json() or {}

        leftover = str(
            data.get("leftover", "")
        ).strip()

        top_k = int(
            data.get("top_k", 6)
        )

        if not leftover:
            return jsonify({
                "success": False,
                "error": "Leftover ingredient is required."
            }), 400

        if not os.path.exists(RECIPE_DATASET):
            raise FileNotFoundError(
                f"Recipe dataset not found: {RECIPE_DATASET}"
            )

        df = pd.read_csv(
            RECIPE_DATASET,
            low_memory=False
        )

        def col(candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        name_col = col(
            ["recipe_name", "name", "title"]
        )

        ingredients_col = col(
            [
                "ingredients",
                "ingredient_names",
                "ingredient_list"
            ]
        )

        instructions_col = col(
            [
                "instructions",
                "directions",
                "steps",
                "method"
            ]
        )

        cuisine_col = col(
            ["cuisine", "cuisine_type"]
        )

        meal_col = col(
            ["meal_type", "meal"]
        )

        id_col = col(
            ["recipe_id", "id"]
        )

        time_col = col(
            [
                "cooking_time_minutes",
                "cooking_time",
                "prep_time"
            ]
        )

        if not name_col or not ingredients_col:
            raise ValueError(
                "Required recipe columns are missing."
            )

        def clean(value):
            if value is None:
                return ""

            try:
                if pd.isna(value):
                    return ""
            except Exception:
                pass

            return str(value).strip()

        names = df[name_col].fillna("").astype(str)
        ingredient_text = (
            df[ingredients_col]
            .fillna("")
            .astype(str)
        )

        # ====================================================
        # TF-IDF + COSINE SIMILARITY
        # ====================================================

        corpus = (
            names + " " + ingredient_text
        ).tolist()

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )

        matrix = vectorizer.fit_transform(
            corpus + [leftover]
        )

        tfidf_scores = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        ).flatten()

        # ====================================================
        # ACTUAL INGREDIENT TOKEN OVERLAP
        # ====================================================

        query_tokens = set(
            re.findall(
                r"[a-zA-Z]+",
                leftover.lower()
            )
        )

        overlap_scores = []

        for text_value in ingredient_text:
            recipe_tokens = set(
                re.findall(
                    r"[a-zA-Z]+",
                    str(text_value).lower()
                )
            )

            if not query_tokens:
                overlap_scores.append(0.0)
                continue

            overlap_scores.append(
                len(query_tokens & recipe_tokens)
                / len(query_tokens)
            )

        overlap_scores = np.asarray(
            overlap_scores,
            dtype=float
        )

        # ====================================================
        # ML SUITABILITY MODEL
        # ====================================================

        ml_scores = np.zeros(
            len(df),
            dtype=float
        )

        model = None

        for model_path in [
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "..",
                "datasets",
                "processed",
                "mealora_hgb_model.joblib"
            ),
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                "..",
                "datasets",
                "processed",
                "mealora_rf_model.joblib"
            ),
        ]:

            if os.path.exists(model_path):
                try:
                    model = joblib.load(
                        model_path
                    )
                    break
                except Exception:
                    model = None

        if model is not None:

            feature_names = [
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

            features = pd.DataFrame(
                0.0,
                index=df.index,
                columns=feature_names
            )

            features[
                "pantry_match_pct"
            ] = np.clip(
                overlap_scores,
                0,
                1
            )

            features[
                "missing_ingredient_count"
            ] = (
                1.0 -
                features["pantry_match_pct"]
            )

            features["expiry_score"] = 0.0
            features["meal_match"] = 0.5
            features["diet_match"] = 0.5
            features["cuisine_match"] = 0.5
            features["time_match"] = 0.5
            features["context_score"] = 0.5

            features[
                "ingredient_count"
            ] = ingredient_text.apply(
                lambda x: max(
                    1,
                    len(
                        re.findall(
                            r",|;|\n",
                            str(x)
                        )
                    ) + 1
                )
            )

            if time_col:
                features[
                    "cooking_time_minutes"
                ] = pd.to_numeric(
                    df[time_col],
                    errors="coerce"
                ).fillna(30)

            try:
                if hasattr(
                    model,
                    "predict_proba"
                ):
                    ml_scores = model.predict_proba(
                        features
                    )[:, 1]
                else:
                    ml_scores = model.predict(
                        features
                    ).astype(float)

                ml_scores = np.nan_to_num(
                    ml_scores,
                    nan=0.0
                )

                ml_scores = np.clip(
                    ml_scores,
                    0,
                    1
                )

            except Exception as model_error:
                print(
                    "Leftover ML model warning:",
                    model_error
                )

        # ====================================================
        # FINAL LEFTOVER SCORE
        # ====================================================

        final_scores = (
            0.55 * tfidf_scores
            + 0.25 * overlap_scores
            + 0.20 * ml_scores
        )

        # Strongly prioritize recipes that actually contain
        # the leftover token.
        final_scores += (
            0.10 *
            (overlap_scores > 0).astype(float)
        )

        df["_leftover_score"] = final_scores
        df["_tfidf_score"] = tfidf_scores
        df["_overlap_score"] = overlap_scores
        df["_ml_score"] = ml_scores

        df = df.sort_values(
            "_leftover_score",
            ascending=False
        )

        # Avoid completely unrelated zero-match recipes.
        df = df[
            (df["_overlap_score"] > 0)
            | (df["_tfidf_score"] > 0)
        ].head(top_k)

        recommendations = []

        for _, row in df.iterrows():

            raw_ingredients = (
                clean(
                    row[ingredients_col]
                )
                if ingredients_col
                else ""
            )

            raw_instructions = (
                clean(
                    row[instructions_col]
                )
                if instructions_col
                else ""
            )

            def split_lines(value):
                if not value:
                    return []

                value = value.replace(
                    "\r",
                    "\n"
                )

                if ";" in value and "\n" not in value:
                    parts = value.split(";")
                else:
                    parts = value.split("\n")

                return [
                    x.strip(" •\t-")
                    for x in parts
                    if x.strip(" •\t-")
                ]

            recipe_id = (
                clean(row[id_col])
                if id_col
                else ""
            )

            matched = []

            for token in query_tokens:
                if token in raw_ingredients.lower():
                    matched.append(token)

            recommendations.append({
                "recipe_id": recipe_id,
                "recipe_name": clean(
                    row[name_col]
                ),
                "match_score": float(
                    np.clip(
                        row["_leftover_score"],
                        0,
                        1
                    )
                ),
                "tfidf_score": float(
                    np.clip(
                        row["_tfidf_score"],
                        0,
                        1
                    )
                ),
                "ml_score": float(
                    np.clip(
                        row["_ml_score"],
                        0,
                        1
                    )
                ),
                "matched_ingredients": matched,
                "ingredients": split_lines(
                    raw_ingredients
                ),
                "instructions": split_lines(
                    raw_instructions
                ),
                "cuisine": (
                    clean(row[cuisine_col])
                    if cuisine_col
                    else ""
                ),
                "meal_type": (
                    clean(row[meal_col])
                    if meal_col
                    else ""
                ),
            })

        return jsonify({
            "success": True,
            "algorithm": (
                "TF-IDF + Cosine Similarity + "
                "Mealora ML Suitability"
            ),
            "recommendations": recommendations
        })

    except Exception as e:
        import traceback
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
if __name__ == "__main__":
    print("=" * 60)
    print("MEALORA BACKEND API")
    print("http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True)
