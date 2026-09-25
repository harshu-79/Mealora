#!/usr/bin/env python3
from pathlib import Path
import re
import shutil
import datetime

ROOT = Path("/Users/harshitha/mealora")
APP = ROOT / "frontend" / "src" / "App.jsx"
BACKEND = ROOT / "backend" / "app.py"

if not APP.exists():
    raise SystemExit(f"App.jsx not found: {APP}")
if not BACKEND.exists():
    raise SystemExit(f"backend/app.py not found: {BACKEND}")

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

app_backup = APP.with_name(f"App.jsx.before_final_safe_fix_{stamp}")
backend_backup = BACKEND.with_name(f"app.py.before_final_safe_fix_{stamp}")
shutil.copy2(APP, app_backup)
shutil.copy2(BACKEND, backend_backup)

print("BACKUP App.jsx:", app_backup)
print("BACKUP app.py :", backend_backup)

# ---------- FRONTEND: recipe opening ----------
app = APP.read_text(encoding="utf-8")

start = app.find("  async function openRecipeAssistant(")
if start == -1:
    raise SystemExit("Could not find openRecipeAssistant() in App.jsx.")

end = app.find("  async function handleAssistantLanguageChange(", start)
if end == -1:
    raise SystemExit("Could not find handleAssistantLanguageChange() in App.jsx.")

new_function = """  async function openRecipeAssistant(
    recipe,
    languageOverride = null,
    rememberSource = true
  ) {
    const requestedLanguage = languageOverride || assistantLanguage;

    if (!recipe?.recipe_id) {
      setAssistantError("The selected recipe does not have a valid recipe ID.");
      return;
    }

    const sourceRecipe = {
      recipe_id: recipe.recipe_id,
      recipe_name: recipe.recipe_name,
    };

    if (rememberSource) {
      setAssistantSourceRecipe(sourceRecipe);
    }

    const requestId = ++assistantRequestRef.current;
    setAssistantLoading(true);
    setAssistantError("");
    setAssistantRecipe(null);

    async function requestRecipe(language) {
      const response = await fetch(RECIPE_ASSISTANT_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recipe_id: sourceRecipe.recipe_id,
          recipe_name: sourceRecipe.recipe_name,
          language,
        }),
      });

      let data = null;

      try {
        data = await response.json();
      } catch {
        data = null;
      }

      if (!response.ok || !data?.success) {
        throw new Error(
          data?.error ||
            `Recipe assistant request failed (${response.status}).`
        );
      }

      if (!data.recipe) {
        throw new Error("Recipe assistant returned no recipe data.");
      }

      return data.recipe;
    }

    try {
      let result;

      try {
        result = await requestRecipe(requestedLanguage);
      } catch (firstError) {
        console.warn("Recipe presentation failed:", firstError);

        if (String(requestedLanguage).trim().toLowerCase() !== "english") {
          await new Promise((resolve) => setTimeout(resolve, 1000));

          try {
            result = await requestRecipe(requestedLanguage);
          } catch (translationError) {
            console.warn(
              "Translation unavailable; falling back to English:",
              translationError
            );

            result = await requestRecipe("English");
            result = {
              ...result,
              note:
                "The selected language service is temporarily unavailable. " +
                "Mealora is showing the original recipe in English.",
            };
          }
        } else {
          await new Promise((resolve) => setTimeout(resolve, 500));
          result = await requestRecipe("English");
        }
      }

      if (requestId !== assistantRequestRef.current) return;

      setAssistantRecipe(result);
    } catch (error) {
      console.error("Recipe assistant error:", error);

      setAssistantRecipe({
        recipe_id: sourceRecipe.recipe_id,
        recipe_name: sourceRecipe.recipe_name,
        language: "English",
        ingredients: [],
        instructions: [],
        note:
          "The recipe service is temporarily unavailable. " +
          "Please restart the Mealora backend and try again.",
      });

      setAssistantError(
        error?.message || "Unable to prepare the selected recipe."
      );
    } finally {
      if (requestId === assistantRequestRef.current) {
        setAssistantLoading(false);
      }
    }
  }

"""

app = app[:start] + new_function + app[end:]

# ---------- FRONTEND: Google Maps only ----------
marker = '              {/* GET RECOMMENDATIONS BUTTON - CENTERED BELOW MODES */}'

if "Open Location in Google Maps" not in app and marker in app:
    maps_code = """            {contextInfo && (
              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                  marginTop: "14px",
                }}
              >
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => {
                    const latitude = contextInfo.latitude;
                    const longitude = contextInfo.longitude;

                    const query =
                      latitude !== null &&
                      latitude !== undefined &&
                      longitude !== null &&
                      longitude !== undefined
                        ? `${latitude},${longitude}`
                        : contextInfo.location_name ||
                          contextInfo.location_city ||
                          "India";

                    const mapsUrl =
                      "https://www.google.com/maps/search/?api=1&query=" +
                      encodeURIComponent(query);

                    window.open(
                      mapsUrl,
                      "_blank",
                      "noopener,noreferrer"
                    );
                  }}
                  style={{
                    padding: "10px 18px",
                    borderRadius: "10px",
                    fontWeight: "700",
                    cursor: "pointer",
                  }}
                >
                  📍 Open Location in Google Maps
                </button>
              </div>
            )}

"""
    app = app.replace(marker, maps_code + marker, 1)

# ---------- FRONTEND: loading text ----------
old_loading = """              <p style={{ color: "#64748b", margin: 0 }}>
                Gemini is translating the recipe selected by Mealora.
              </p>"""

new_loading = """              <p style={{ color: "#64748b", margin: 0 }}>
                {String(assistantLanguage).trim().toLowerCase() === "english"
                  ? "Opening the recipe selected by Mealora."
                  : "Gemini is translating the recipe selected by Mealora."}
              </p>"""

if old_loading in app:
    app = app.replace(old_loading, new_loading, 1)

APP.write_text(app, encoding="utf-8")
print("FRONTEND PATCHED:", APP)

# ---------- BACKEND: recipe assistant only ----------
backend = BACKEND.read_text(encoding="utf-8")

route_pattern = re.compile(
    r'@app\.route\("/api/recipe-assistant",\s*methods=\["POST"\]\)\s*'
    r'def recipe_assistant\(\):.*?'
    r'(?=\n@app\.route\()',
    re.S,
)

replacement = """@app.route("/api/recipe-assistant", methods=["POST"])
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

"""

match = route_pattern.search(backend)

if not match:
    print("WARNING: /api/recipe-assistant route not found.")
    print("Backend backup exists; backend was NOT changed.")
else:
    backend = backend[:match.start()] + replacement + backend[match.end():]
    BACKEND.write_text(backend, encoding="utf-8")
    print("BACKEND PATCHED:", BACKEND)

print("")
print("FINAL SAFE PATCH COMPLETE.")
print("Weekly Planner was NOT touched.")
print("Leftover Intelligence was NOT touched.")
print("Recommendation UI was NOT replaced.")
print("Only recipe opening, Gemini fallback, and Google Maps were patched.")
