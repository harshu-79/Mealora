from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path("/Users/harshitha/mealora")
APP = ROOT / "frontend/src/App.jsx"

if not APP.exists():
    raise SystemExit(f"Could not find {APP}")

# ------------------------------------------------------------
# BACKUP FIRST
# ------------------------------------------------------------

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = ROOT / f"mealora_backup_before_explainability_{stamp}"
backup.mkdir(parents=True, exist_ok=True)

shutil.copy2(APP, backup / "App.jsx")

app = APP.read_text(encoding="utf-8")

# ------------------------------------------------------------
# EXPLAINABILITY HELPER
# ------------------------------------------------------------

helper = r'''
function getRecommendationExplanation(recipe, contextInfo) {
  const pantry = Number(recipe.pantry_score || 0);
  const expiry = Number(recipe.expiry_score || 0);
  const content = Number(recipe.content_score || 0);
  const collaborative = Number(recipe.collaborative_score || 0);
  const context = Number(recipe.context_score || 0);
  const time = Number(recipe.time_match || 0);
  const weather = Number(recipe.weather_score || 0);
  const festival = Number(recipe.festival_score || 0);
  const location = Number(recipe.location_score || 0);
  const hgb = Number(recipe.hgb_suitability_score || 0);
  const missing = Number(recipe.missing_ingredient_count || 0);

  const reasons = [];

  if (pantry >= 0.80) {
    reasons.push(
      `🧺 Strong pantry match (${Math.round(pantry * 100)}%)`
    );
  } else if (pantry >= 0.50) {
    reasons.push(
      `🧺 Good pantry match (${Math.round(pantry * 100)}%)`
    );
  }

  if (missing === 0) {
    reasons.push("✅ All required ingredients appear available");
  } else if (missing <= 2) {
    reasons.push(
      `🛒 Only ${missing} ingredient${missing === 1 ? "" : "s"} may be missing`
    );
  }

  if (expiry >= 0.80) {
    reasons.push("♻️ Helps use ingredients approaching expiry");
  }

  if (time >= 0.99) {
    if (contextInfo?.max_time) {
      reasons.push(
        `⏱️ Fits your ${contextInfo.max_time}-minute time limit`
      );
    } else {
      reasons.push("⏱️ Fits the selected cooking-time preference");
    }
  }

  if (context >= 0.70) {
    reasons.push("🎯 Strong match to your current meal preferences");
  } else if (context >= 0.50) {
    reasons.push("🎯 Matches your current meal context");
  }

  if (weather >= 0.50) {
    reasons.push("🌦️ Suitable for the current weather");
  }

  if (location >= 0.50) {
    reasons.push("📍 Matches your regional/location context");
  }

  if (festival >= 0.50) {
    reasons.push("🎉 Matches today's festival or occasion");
  }

  if (hgb >= 0.90) {
    reasons.push("🤖 High ML suitability score");
  }

  if (content >= 0.60) {
    reasons.push("🔎 Strong recipe-content similarity");
  }

  if (collaborative > 0.05) {
    reasons.push("👥 Supported by user-preference signals");
  }

  return reasons.slice(0, 4);
}

'''

# Insert helper only once.
if "function getRecommendationExplanation(" not in app:

    marker = "\nfunction App() {"

    if marker not in app:
        raise SystemExit(
            "Could not find 'function App() {' in App.jsx. "
            "Nothing was changed."
        )

    app = app.replace(
        marker,
        "\n" + helper + marker,
        1
    )

# ------------------------------------------------------------
# FIND EXISTING "WHY RECOMMENDED" SECTION
# ------------------------------------------------------------

start_marker = '<strong>Why recommended?</strong>'

if start_marker not in app:
    raise SystemExit(
        "Could not find the existing 'Why recommended?' section. "
        "Nothing was changed."
    )

start = app.rfind("<div", 0, app.find(start_marker))

if start == -1:
    raise SystemExit(
        "Could not locate the parent explanation <div>. "
        "Nothing was changed."
    )

# Find the matching closing div using a simple JSX div counter.
depth = 0
i = start

while i < len(app):
    open_match = app.find("<div", i)
    close_match = app.find("</div>", i)

    if close_match == -1:
        raise SystemExit(
            "Could not find the end of the explanation block. "
            "Nothing was changed."
        )

    if open_match != -1 and open_match < close_match:
        depth += 1
        i = open_match + 4
    else:
        depth -= 1
        i = close_match + 6

        if depth == 0:
            end = i
            break

new_block = r'''<div
                        style={{
                          marginTop: "14px",
                          padding: "14px",
                          borderRadius: "12px",
                          background:
                            "linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%)",
                          border: "1px solid #dbeafe",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            gap: "10px",
                            marginBottom: "9px",
                          }}
                        >
                          <strong>
                            💡 Why Mealora recommended this
                          </strong>

                          <span
                            style={{
                              fontSize: "10px",
                              fontWeight: "700",
                              color: "#64748b",
                              background: "#fff",
                              border: "1px solid #e2e8f0",
                              borderRadius: "999px",
                              padding: "3px 7px",
                            }}
                          >
                            ML EXPLANATION
                          </span>
                        </div>

                        <ul
                          style={{
                            margin: 0,
                            paddingLeft: "20px",
                            color: "#475569",
                            fontSize: "13px",
                            lineHeight: 1.65,
                          }}
                        >
                          {getRecommendationExplanation(
                            recipe,
                            contextInfo
                          ).map((reason, reasonIndex) => (
                            <li key={reasonIndex}>
                              {reason}
                            </li>
                          ))}
                        </ul>

                        <div
                          style={{
                            marginTop: "10px",
                            paddingTop: "9px",
                            borderTop: "1px solid #dbeafe",
                            fontSize: "11px",
                            color: "#64748b",
                          }}
                        >
                          Based on Mealora's hybrid ranking signals:
                          pantry, expiry, context, time, weather,
                          festival, location and ML suitability.
                        </div>
                      </div>'''

app = app[:start] + new_block + app[end:]

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

APP.write_text(app, encoding="utf-8")

print()
print("==============================================")
print("MEALORA EXPLAINABILITY PATCH SUCCESSFUL")
print("==============================================")
print()
print("Updated:")
print(APP)
print()
print("Backup created:")
print(backup)
print()
print("Backend was NOT modified.")
print("Gemini was NOT modified.")
print("Festival logic was NOT modified.")
print("Pantry deduction was NOT modified.")
print("ML ranking was NOT modified.")
print()
print("Only the recommendation explanation UI was added.")