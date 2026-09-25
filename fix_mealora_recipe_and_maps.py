#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import datetime

ROOT = Path("/Users/harshitha/mealora")
BACKEND = ROOT / "backend" / "app.py"

if not BACKEND.exists():
    raise SystemExit(f"Backend not found: {BACKEND}")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup = BACKEND.with_name(f"app.py.before_recipe_fix_{stamp}")
shutil.copy2(BACKEND, backup)
print("Backup:", backup)

text = BACKEND.read_text(encoding="utf-8")

pattern = re.compile(
    r'@app\.route\("/api/recipe-assistant",\s*methods=\["POST"\]\)\s*'
    r'def recipe_assistant\(\):.*?'
    r'(?=\n@app\.route\()',
    re.S,
)

replacement = r"""@app.route("/api/recipe-assistant", methods=["POST"])
def recipe_assistant():
    try:
        data = request.get_json() or {}

        recipe = _get_recipe_for_gemini(
            recipe_id=data.get("recipe_id"),
            recipe_name=data.get("recipe_name"),
        )

        language = data.get("language", "English")

        # English never depends on Gemini.
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

        # Non-English: retry Gemini once after a short delay.
        last_error = None

        for attempt in range(2):
            try:
                result = _gemini_translate_recipe(recipe, language)

                return jsonify({
                    "success": True,
                    "assistant_role": "translation_and_recipe_presentation_only",
                    "recipe": result,
                })

            except Exception as e:
                last_error = e
                print(
                    f"Gemini recipe presentation attempt {attempt + 1}/2 failed: {e}"
                )

                if attempt == 0:
                    import time
                    time.sleep(1.2)

        # Final fallback: never leave the frontend blank.
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

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500

"""

match = pattern.search(text)
if not match:
    raise SystemExit(
        "Could not find /api/recipe-assistant in backend/app.py. "
        "The backup was created, but the backend was not changed."
    )

text = text[:match.start()] + replacement + text[match.end():]
BACKEND.write_text(text, encoding="utf-8")

print("Patched:", BACKEND)
