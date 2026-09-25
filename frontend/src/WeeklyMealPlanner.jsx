import { useState } from "react";

const API = "http://127.0.0.1:5000/api/weekly-plan";

export default function WeeklyMealPlanner({
  pantryItems = [],
  onOpenRecipe,
}) {
  const [adults, setAdults] = useState(2);
  const [children, setChildren] = useState(0);
  const [seniors, setSeniors] = useState(0);
  const [diet, setDiet] = useState("Mixed");
  const [maxTime, setMaxTime] = useState(45);
  const [plan, setPlan] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function generatePlan(event) {
    event?.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pantry: pantryItems,
          household: {
            adults: Number(adults),
            children: Number(children),
            seniors: Number(seniors),
          },
          diet,
          max_time: Number(maxTime),
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data.error || "Unable to generate weekly plan."
        );
      }

      setPlan(Array.isArray(data.plan) ? data.plan : []);
      setSummary(data.summary || {});
    } catch (err) {
      console.error("Weekly planner error:", err);
      setError(
        err.message || "Unable to generate weekly plan."
      );
    } finally {
      setLoading(false);
    }
  }

  const allMeals = plan.flatMap((day) =>
    Array.isArray(day?.meals) ? day.meals : []
  );

  const differentRecipes = new Set(
    allMeals
      .map((meal) => String(meal?.recipe_name || "").trim())
      .filter(Boolean)
  ).size;

  function averageMetric(field) {
    const values = allMeals
      .map((meal) => Number(meal?.[field]))
      .filter((value) => Number.isFinite(value));

    if (!values.length) return 0;

    return Math.round(
      (values.reduce((a, b) => a + b, 0) / values.length) * 100
    );
  }

  const pantryUtilization =
    summary?.pantry_utilization !== undefined &&
    summary?.pantry_utilization !== null &&
    summary?.pantry_utilization !== ""
      ? Number(summary.pantry_utilization)
      : averageMetric("pantry_score");

  const expiryReduction =
    summary?.expiry_focus !== undefined &&
    summary?.expiry_focus !== null &&
    summary?.expiry_focus !== ""
      ? Number(summary.expiry_focus)
      : averageMetric("expiry_score");

  function openRecipe(meal) {
    if (!onOpenRecipe) {
      console.warn(
        "WeeklyMealPlanner: onOpenRecipe was not supplied by App.jsx"
      );
      return;
    }

    onOpenRecipe({
      recipe_id: meal?.recipe_id,
      recipe_name: meal?.recipe_name,
    });
  }

  return (
    <div>
      <div style={{ marginBottom: 22 }}>
        <h2 style={{ margin: 0 }}>📅 Weekly Meal Planner</h2>

        <p style={{ marginTop: 8, color: "#64748b" }}>
          Mealora builds a 7-day meal plan by balancing pantry usage,
          expiry, nutrition, cooking time, preferences and different recipes.
        </p>
      </div>

      <form
        onSubmit={generatePlan}
        style={{
          padding: 20,
          borderRadius: 16,
          background: "#f8fafc",
          border: "1px solid #e2e8f0",
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(150px, 1fr))",
            gap: 14,
          }}
        >
          <label>
            Adults
            <input
              type="number"
              min="0"
              value={adults}
              onChange={(e) => setAdults(e.target.value)}
              style={{ display: "block", width: "100%" }}
            />
          </label>

          <label>
            Children
            <input
              type="number"
              min="0"
              value={children}
              onChange={(e) => setChildren(e.target.value)}
              style={{ display: "block", width: "100%" }}
            />
          </label>

          <label>
            Seniors
            <input
              type="number"
              min="0"
              value={seniors}
              onChange={(e) => setSeniors(e.target.value)}
              style={{ display: "block", width: "100%" }}
            />
          </label>

          <label>
            Diet
            <select
              value={diet}
              onChange={(e) => setDiet(e.target.value)}
              style={{ display: "block", width: "100%" }}
            >
              <option value="Mixed">Mixed</option>
              <option value="Vegetarian">Vegetarian</option>
              <option value="Non-Vegetarian">
                Non-Vegetarian
              </option>
            </select>
          </label>

          <label>
            Max cooking time
            <select
              value={maxTime}
              onChange={(e) => setMaxTime(e.target.value)}
              style={{ display: "block", width: "100%" }}
            >
              <option value="30">30 min</option>
              <option value="45">45 min</option>
              <option value="60">60 min</option>
              <option value="90">90 min</option>
            </select>
          </label>
        </div>

        <button
          type="submit"
          className="primary-button"
          disabled={loading}
          style={{
            marginTop: 18,
            width: "100%",
          }}
        >
          {loading
            ? "⏳ Optimizing 7-day plan..."
            : "✨ Generate Weekly Plan"}
        </button>
      </form>

      {error && (
        <div
          style={{
            marginTop: 16,
            padding: 14,
            borderRadius: 12,
            background: "#fff1f2",
            color: "#b91c1c",
            border: "1px solid #fecdd3",
          }}
        >
          {error}
        </div>
      )}

      {plan.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(160px, 1fr))",
            gap: 12,
            marginTop: 18,
          }}
        >
          <div
            style={{
              padding: 18,
              borderRadius: 14,
              border: "1px solid #e2e8f0",
              background: "#fff",
            }}
          >
            <div style={{ color: "#334155" }}>
              Pantry utilization
            </div>
            <strong style={{ display: "block", fontSize: 28 }}>
              {Number.isFinite(pantryUtilization)
                ? pantryUtilization
                : 0}
              %
            </strong>
          </div>

          <div
            style={{
              padding: 18,
              borderRadius: 14,
              border: "1px solid #e2e8f0",
              background: "#fff",
            }}
          >
            <div style={{ color: "#334155" }}>
              Expiry reduction
            </div>
            <strong style={{ display: "block", fontSize: 28 }}>
              {Number.isFinite(expiryReduction)
                ? expiryReduction
                : 0}
              %
            </strong>
          </div>

          <div
            style={{
              padding: 18,
              borderRadius: 14,
              border: "1px solid #e2e8f0",
              background: "#fff",
            }}
          >
            <div style={{ color: "#334155" }}>
              Different recipes
            </div>
            <strong style={{ display: "block", fontSize: 28 }}>
              {differentRecipes}
            </strong>
          </div>

          <div
            style={{
              padding: 18,
              borderRadius: 14,
              border: "1px solid #e2e8f0",
              background: "#fff",
            }}
          >
            <div style={{ color: "#334155" }}>
              Meals planned
            </div>
            <strong style={{ display: "block", fontSize: 28 }}>
              {allMeals.length}
            </strong>
          </div>
        </div>
      )}

      {plan.length > 0 && (
        <div
          style={{
            marginTop: 22,
            display: "grid",
            gap: 14,
          }}
        >
          {plan.map((day, dayIndex) => (
            <div
              key={day?.day || dayIndex}
              style={{
                padding: 18,
                borderRadius: 16,
                border: "1px solid #e2e8f0",
                background: "#fff",
              }}
            >
              <h3 style={{ margin: "0 0 14px" }}>
                {day?.day || `Day ${dayIndex + 1}`}
              </h3>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns:
                    "repeat(auto-fit, minmax(220px, 1fr))",
                  gap: 12,
                }}
              >
                {(Array.isArray(day?.meals)
                  ? day.meals
                  : []
                ).map((meal, mealIndex) => (
                  <div
                    key={
                      `${meal?.meal_type || "meal"}-${meal?.recipe_id || mealIndex}`
                    }
                    role="button"
                    tabIndex={0}
                    onClick={() => openRecipe(meal)}
                    onKeyDown={(event) => {
                      if (
                        event.key === "Enter" ||
                        event.key === " "
                      ) {
                        event.preventDefault();
                        openRecipe(meal);
                      }
                    }}
                    style={{
                      padding: 16,
                      borderRadius: 14,
                      background: "#f8fafc",
                      border: "1px solid #e2e8f0",
                      cursor: "pointer",
                      transition: "transform 0.15s ease",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 12,
                        color: "#64748b",
                        fontWeight: 700,
                      }}
                    >
                      {String(
                        meal?.meal_type || "Meal"
                      ).toUpperCase()}
                    </div>

                    <strong
                      style={{
                        display: "block",
                        marginTop: 7,
                        fontSize: 17,
                        color: "#1f2937",
                      }}
                    >
                      {meal?.recipe_name ||
                        "Recipe unavailable"}
                    </strong>

                    <div
                      style={{
                        marginTop: 8,
                        fontSize: 12,
                        fontWeight: 700,
                        color: "#475569",
                      }}
                    >
                      🌐 Open Recipe in English
                    </div>

                    {meal?.cooking_time && (
                      <div
                        style={{
                          marginTop: 8,
                          fontSize: 13,
                          color: "#64748b",
                        }}
                      >
                        ⏱ {meal.cooking_time} min
                      </div>
                    )}

                    {meal?.reason && (
                      <div
                        style={{
                          marginTop: 8,
                          fontSize: 13,
                          color: "#475569",
                        }}
                      >
                        {meal.reason}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
