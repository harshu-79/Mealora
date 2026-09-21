from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import json
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
    try:
        lat, lon = float(lat), float(lon)
        if 8 <= lat < 16 and 74 <= lon < 80: return "South Indian"
        if 16 <= lat < 24 and 72 <= lon < 82: return "West Indian"
        if 20 <= lat < 29 and 76 <= lon < 89: return "North Indian"
        if 21 <= lat < 28 and 85 <= lon < 94: return "East Indian"
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

    df = pd.read_csv(RECIPE_DATASET, low_memory=False)

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
    df = pd.read_csv(RECIPE_DATASET, low_memory=False)
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


@app.route("/api/recipe-assistant", methods=["POST"])
def recipe_assistant():
    """
    Presentation-only endpoint.
    The selected recipe comes from Mealora's ML ranker; Gemini translates
    and formats that exact recipe in the requested language.
    """
    try:
        data = request.get_json() or {}
        recipe = _get_recipe_for_gemini(
            recipe_id=data.get("recipe_id"),
            recipe_name=data.get("recipe_name"),
        )
        language = data.get("language", "English")
        result = _gemini_translate_recipe(recipe, language)

        return jsonify({
            "success": True,
            "assistant_role": "translation_and_recipe_presentation_only",
            "recipe": result,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
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
            "recommendations": result.to_dict(orient="records")
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("MEALORA BACKEND API")
    print("http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True)
