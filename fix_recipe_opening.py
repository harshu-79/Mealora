from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path("/Users/harshitha/mealora")
APP = ROOT / "frontend/src/App.jsx"
WEEKLY = ROOT / "frontend/src/WeeklyMealPlanner.jsx"

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- App.jsx ----------
s = APP.read_text(encoding="utf-8")
shutil.copy2(APP, APP.with_name(f"App.jsx.before_recipe_open_fix_{stamp}"))

# 1. Make recipe fields always arrays
helper = r'''
function normalizeRecipeLines(value) {
  if (Array.isArray(value)) {
    return value.map((x) => String(x).trim()).filter(Boolean);
  }

  if (value === null || value === undefined) {
    return [];
  }

  if (typeof value === "object") {
    return Object.values(value)
      .flat()
      .map((x) => String(x).trim())
      .filter(Boolean);
  }

  const text = String(value).trim();
  if (!text) return [];

  try {
    const parsed = JSON.parse(text);
    if (Array.isArray(parsed)) {
      return parsed.map((x) => String(x).trim()).filter(Boolean);
    }
  } catch (_) {}

  return text
    .replace(/\r/g, "\n")
    .split(/\n|;/)
    .map((x) => x.replace(/^[-•*]\s*/, "").trim())
    .filter(Boolean);
}
'''

if "function normalizeRecipeLines" not in s:
    marker = "function App()"
    s = s.replace(marker, helper + "\n" + marker, 1)

# 2. Normalize Gemini/backend response before storing it
old = "setAssistantRecipe(data.recipe);"
new = """setAssistantRecipe({
        ...data.recipe,
        ingredients: normalizeRecipeLines(data.recipe?.ingredients),
        instructions: normalizeRecipeLines(data.recipe?.instructions),
      });"""
s = s.replace(old, new, 1)

# 3. Do not require recipe_id if recipe_name exists
s = s.replace(
    'if (!recipe?.recipe_id) {',
    'if (!recipe?.recipe_id && !recipe?.recipe_name) {',
    1
)

# 4. Preserve recipe name/id for weekly planner too
old_source = '''const sourceRecipe = {
      recipe_id: recipe.recipe_id,
      recipe_name: recipe.recipe_name,
    };'''
new_source = '''const sourceRecipe = {
      recipe_id: recipe.recipe_id || "",
      recipe_name: recipe.recipe_name || "",
    };'''
s = s.replace(old_source, new_source, 1)

# 5. Language switching should work even when only recipe_name exists
s = s.replace(
    'if (assistantSourceRecipe?.recipe_id) {',
    'if (assistantSourceRecipe?.recipe_id || assistantSourceRecipe?.recipe_name) {',
    1
)

# 6. Safe rendering even if backend sends unexpected data
s = s.replace(
    '(assistantRecipe.ingredients || []).map',
    'normalizeRecipeLines(assistantRecipe.ingredients).map'
)
s = s.replace(
    '(assistantRecipe.instructions || []).map',
    'normalizeRecipeLines(assistantRecipe.instructions).map'
)

# 7. Weekly planner gets the recipe-opening callback
if "<WeeklyMealPlanner" in s and "onOpenRecipe={openRecipeAssistant}" not in s:
    s = s.replace(
        "<WeeklyMealPlanner",
        "<WeeklyMealPlanner\n              onOpenRecipe={openRecipeAssistant}",
        1
    )

APP.write_text(s, encoding="utf-8")

# ---------- WeeklyMealPlanner.jsx ----------
w = WEEKLY.read_text(encoding="utf-8")
shutil.copy2(
    WEEKLY,
    WEEKLY.with_name(f"WeeklyMealPlanner.jsx.before_recipe_open_fix_{stamp}")
)

# Accept callback
w = w.replace(
    'export default function WeeklyMealPlanner({ pantryItems = [] })',
    'export default function WeeklyMealPlanner({ pantryItems = [], onOpenRecipe })',
    1
)

# Add Open Recipe button after recipe name
needle = '''<strong style={{ display: "block", marginTop: 5 }}>{meal.recipe_name}</strong>'''

button = '''<strong style={{ display: "block", marginTop: 5 }}>{meal.recipe_name}</strong>

                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() =>
                        onOpenRecipe?.({
                          recipe_id: meal.recipe_id,
                          recipe_name: meal.recipe_name,
                        })
                      }
                      style={{
                        marginTop: 10,
                        width: "100%",
                        padding: "8px 10px",
                        borderRadius: 9,
                        cursor: "pointer",
                      }}
                    >
                      🌐 Open Recipe in English
                    </button>'''

if "🌐 Open Recipe in English" not in w:
    w = w.replace(needle, button, 1)

WEEKLY.write_text(w, encoding="utf-8")

print("✅ Recipe opening fix applied.")
print("✅ App.jsx patched.")
print("✅ WeeklyMealPlanner.jsx patched.")
print("✅ Backups created.")
