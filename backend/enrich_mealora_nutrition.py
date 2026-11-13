#!/usr/bin/env python3
"""
Mealora batch nutrition enrichment.
- Preserves existing nutrition values.
- Uses Gemini to estimate missing per-serving nutrition from recipe ingredients/instructions.
- Processes in batches, checkpoints after every batch, and can resume safely.
- Writes a NEW CSV; never overwrites the source dataset.
- Adds nutrition_source / nutrition_status / nutrition_estimated_at columns.
"""

import os
import re
import json
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

NUTRIENTS = [
    "calories", "protein", "carbohydrates", "fat", "fibre",
    "sodium", "calcium", "iron", "vitamin_c", "folate"
]
UNITS = {
    "calories": "kcal",
    "protein": "g",
    "carbohydrates": "g",
    "fat": "g",
    "fibre": "g",
    "sodium": "mg",
    "calcium": "mg",
    "iron": "mg",
    "vitamin_c": "mg",
    "folate": "mcg",
}

def load_env_file(path):
    if not path or not Path(path).exists():
        return
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

def is_missing(v):
    return pd.isna(v) or str(v).strip() == ""

def numeric(v):
    try:
        x = float(v)
        if x < 0 or x != x or x == float("inf"):
            return None
        return x
    except (TypeError, ValueError):
        return None

def row_has_complete_nutrition(row):
    # Complete means all requested values are numeric. Existing partial values
    # are retained and only missing fields are requested from Gemini.
    return all(numeric(row.get(k)) is not None for k in NUTRIENTS)

def compact(value, max_chars=4500):
    if is_missing(value):
        return ""
    text = str(value).strip()
    return text[:max_chars]

def first_available(row, names):
    for name in names:
        value = row.get(name)
        if not is_missing(value):
            return value
    return ""

def parse_json_response(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise

def make_prompt(batch):
    nutrient_schema = ", ".join(
        f'"{n}": number|null' for n in NUTRIENTS
    )
    records = []
    for item in batch:
        records.append({
            "recipe_id": str(item["recipe_id"]),
            "recipe_name": item["recipe_name"],
            "ingredients": item["ingredients"],
            "instructions": item["instructions"],
            "serving_info": item["serving_info"],
            "existing_nutrition_to_preserve": item["existing"],
            "fields_to_estimate": item["missing"],
        })
    return f"""
You are estimating nutrition for a recipe database. Return ONLY valid JSON.
Estimate nutrition PER ONE SERVING, not the whole batch.
Use ingredient quantities and cooking method when supplied. If quantities or
serving count are absent, use a reasonable typical Indian home-cooking portion
and clearly reflect uncertainty in confidence and notes. Do not claim these
are laboratory measurements. Do not invent ingredients absent from the recipe.
Use realistic non-negative values and these units:
calories=kcal, protein/carbohydrates/fat/fibre=g,
sodium/calcium/iron/vitamin_c=mg, folate=mcg.

For each recipe estimate ONLY the fields listed in fields_to_estimate.
For fields already present in existing_nutrition_to_preserve, do not change them.
If a nutrient genuinely cannot be estimated, use null; do not put zero merely
because information is missing. We still want a best-effort estimate where
the recipe gives enough information.

Return this exact shape:
{{
  "recipes": [
    {{
      "recipe_id": "same id as input",
      "nutrition": {{{nutrient_schema}}},
      "confidence": "low|medium|high",
      "notes": "short reason/assumptions"
    }}
  ]
}}

Recipes:
{json.dumps(records, ensure_ascii=False)}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../datasets/processed/recipe_master_dataset.csv")
    parser.add_argument("--output", default="../datasets/processed/recipe_master_dataset_nutrition_COMPLETE.csv")
    parser.add_argument("--env", default=".env", help="Path to backend .env")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pause", type=float, default=1.2)
    parser.add_argument("--max-recipes", type=int, default=0, help="0 means all missing recipes")
    args = parser.parse_args()

    backend_dir = Path(__file__).resolve().parent
    input_path = (backend_dir / args.input).resolve()
    output_path = (backend_dir / args.output).resolve()
    env_path = (backend_dir / args.env).resolve()
    load_env_file(env_path)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(f"GEMINI_API_KEY not found. Checked environment and {env_path}")

    try:
        from google import genai
    except ImportError:
        raise SystemExit("Gemini SDK missing. Activate backend/venv and run: pip install google-genai")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    client = genai.Client(api_key=api_key)

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Resume from output if it already exists; otherwise begin from original.
    source_path = output_path if output_path.exists() else input_path
    df = pd.read_csv(source_path, low_memory=False)
    required = {"recipe_id", "recipe_name"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Dataset must contain columns: {sorted(required)}")

    # Ensure expected nutrition columns exist without deleting any existing fields.
    for col in NUTRIENTS:
        if col not in df.columns:
            df[col] = pd.NA

    for col, default in [
        ("nutrition_source", ""),
        ("nutrition_status", ""),
        ("nutrition_confidence", ""),
        ("nutrition_notes", ""),
        ("nutrition_estimated_at", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    # A row is pending if at least one of the core nutrition values is absent.
    pending = []
    for idx, row in df.iterrows():
        missing = [n for n in NUTRIENTS if numeric(row.get(n)) is None]
        if missing:
            pending.append((idx, missing))
    if args.max_recipes > 0:
        pending = pending[:args.max_recipes]

    print(f"Loaded {len(df):,} recipes from {source_path}")
    print(f"Recipes needing at least one nutrition value: {len(pending):,}")
    print(f"Gemini model: {model_name}")
    print(f"Output checkpoint: {output_path}")

    completed = 0
    failed = 0
    for offset in range(0, len(pending), max(1, args.batch_size)):
        chunk = pending[offset:offset + max(1, args.batch_size)]
        payload = []
        for idx, missing in chunk:
            row = df.loc[idx]
            payload.append({
                "recipe_id": str(row.get("recipe_id")),
                "recipe_name": compact(row.get("recipe_name"), 300),
                "ingredients": compact(first_available(row, ["ingredients", "ingredient_list", "ingredients_clean", "canonical_ingredients"]), 5000),
                "instructions": compact(row.get("instructions"), 5000),
                "serving_info": compact(first_available(row, ["servings", "serving_size", "yield"]), 300),
                "existing": {n: numeric(row.get(n)) for n in NUTRIENTS if numeric(row.get(n)) is not None},
                "missing": missing,
            })

        success = False
        last_error = None
        for attempt in range(4):
            try:
                prompt = make_prompt(payload)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                    },
                )
                result = parse_json_response(response.text)
                returned = result.get("recipes", [])
                by_id = {str(x.get("recipe_id")): x for x in returned if isinstance(x, dict)}
                for (idx, missing), item in zip(chunk, payload):
                    generated = by_id.get(item["recipe_id"])
                    if not generated:
                        failed += 1
                        continue
                    nutrition = generated.get("nutrition") or {}
                    for nutrient in missing:
                        value = numeric(nutrition.get(nutrient))
                        if value is not None:
                            df.at[idx, nutrient] = value
                    df.at[idx, "nutrition_source"] = (
                        "Dataset + Gemini AI estimate" if item["existing"] else "Gemini AI estimate"
                    )
                    df.at[idx, "nutrition_status"] = "AI-estimated"
                    df.at[idx, "nutrition_confidence"] = str(generated.get("confidence", "low"))
                    df.at[idx, "nutrition_notes"] = str(generated.get("notes", ""))[:1000]
                    df.at[idx, "nutrition_estimated_at"] = datetime.now(timezone.utc).isoformat()
                    completed += 1
                success = True
                break
            except Exception as exc:
                last_error = exc
                wait = min(30, 2 ** attempt * 2)
                print(f"Batch {offset // args.batch_size + 1}: attempt {attempt + 1} failed: {exc}")
                time.sleep(wait)

        # Save after every batch so interruption does not lose prior work.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(
            f"Progress: {min(offset + len(chunk), len(pending)):,}/{len(pending):,} "
            f"pending processed | updated rows: {completed:,} | failed rows: {failed:,}"
        )
        if not success:
            print(f"Batch skipped after retries: {last_error}")
        if args.pause > 0:
            time.sleep(args.pause)

    # Final audit: never claim 100% if Gemini left nulls.
    for col in NUTRIENTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    complete_count = sum(row_has_complete_nutrition(row) for _, row in df.iterrows())
    print("\nFINISHED")
    print(f"Recipes with all {len(NUTRIENTS)} nutrition fields: {complete_count:,}/{len(df):,}")
    print(f"Coverage: {complete_count / max(1, len(df)) * 100:.2f}%")
    print(f"Saved: {output_path}")
    print("Existing nutrition values were preserved. AI estimates are explicitly labelled.")

if __name__ == "__main__":
    main()
