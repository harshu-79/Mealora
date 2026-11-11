import { useEffect, useMemo, useState } from "react";

const API = "http://127.0.0.1:5000/api/weekly-plan";
const CONTEXT_API = "http://127.0.0.1:5000/api/context";

function todayISO() {
  const d = new Date();
  const local = new Date(d.getTime() - d.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 10);
}

function prettyDate(value) {
  if (!value) return "";
  const d = new Date(`${value}T00:00:00`);
  return d.toLocaleDateString("en-IN", {
    weekday: "long",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function recipeButtonData(recipe) {
  if (!recipe) return null;
  return {
    recipe_id: recipe.recipe_id || "",
    recipe_name: recipe.recipe_name || "",
  };
}

function pct(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

export default function WeeklyMealPlanner({
  pantryItems = [],
  onOpenRecipe,
  onResetPantry,
}) {
  const [adults, setAdults] = useState(2);
  const [children, setChildren] = useState(0);
  const [seniors, setSeniors] = useState(0);

  const [mode, setMode] = useState("automatic");
  const [date, setDate] = useState(todayISO());
  const [locationCity, setLocationCity] = useState("");
  const [latitude, setLatitude] = useState(null);
  const [longitude, setLongitude] = useState(null);
  const [diet, setDiet] = useState("Mixed");
  const [cuisine, setCuisine] = useState("Auto");
  const [maxTime, setMaxTime] = useState(45);

  const [context, setContext] = useState(null);
  const [contextLoading, setContextLoading] = useState(false);
  const [locationLoading, setLocationLoading] = useState(false);
  const [plan, setPlan] = useState([]);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const householdSize =
    Number(adults || 0) + Number(children || 0) + Number(seniors || 0);

  const pantrySummary = useMemo(() => {
    const total = pantryItems.length;
    const expiring = pantryItems.filter((item) => {
      if (!item?.expiry_date) return false;
      const expiry = new Date(`${item.expiry_date}T23:59:59`);
      const days = Math.ceil((expiry - new Date()) / 86400000);
      return days <= 7;
    }).length;
    return { total, expiring };
  }, [pantryItems]);

  async function loadContext(custom = mode === "custom", coords = null) {
    setContextLoading(true);
    setError("");
    try {
      const lat = coords?.latitude ?? latitude;
      const lon = coords?.longitude ?? longitude;
      const response = await fetch(CONTEXT_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: custom ? "custom" : "automatic",
          date: custom ? date : undefined,
          location_city: custom ? locationCity : undefined,
          latitude: lat,
          longitude: lon,
          meal_type: "Automatic",
          diet,
          cuisine: custom && cuisine !== "Auto" ? cuisine : "Automatic",
          max_time: Number(maxTime),
        }),
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Unable to load live context.");
      }
      setContext(data.context || null);
      if (data.context?.location_city && !locationCity) {
        setLocationCity(data.context.location_city);
      }
    } catch (err) {
      console.error("Weekly context error:", err);
      setError(err.message || "Unable to load live context.");
    } finally {
      setContextLoading(false);
    }
  }

  function fetchMyLocation({ automatic = false } = {}) {
    if (!navigator.geolocation) {
      if (!automatic) setError("Location is not supported by this browser.");
      return;
    }

    setLocationLoading(true);
    if (!automatic) setError("");

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const lat = position.coords.latitude;
        const lon = position.coords.longitude;
        setLatitude(lat);
        setLongitude(lon);
        await loadContext(mode === "custom", { latitude: lat, longitude: lon });
        setLocationLoading(false);
      },
      async () => {
        setLocationLoading(false);
        if (!automatic) {
          setError("Location permission was not granted. You can enter a city manually.");
        }
        await loadContext(false);
      },
      {
        enableHighAccuracy: false,
        timeout: 10000,
        maximumAge: 300000,
      }
    );
  }

  // Smart Automatic: request browser location on first load so the backend
  // can reverse-geocode the city and fetch current weather automatically.
  useEffect(() => {
    fetchMyLocation({ automatic: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (mode === "custom") {
      loadContext(true);
    } else if (latitude != null && longitude != null) {
      loadContext(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  async function generatePlan(event) {
    event?.preventDefault();
    if (householdSize <= 0) {
      setError("Add at least one adult, child or senior.");
      return;
    }

    setLoading(true);
    setError("");
    setPlan([]);
    setSummary(null);

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
          mode,
          date,
          location_city: locationCity,
          latitude,
          longitude,
          meal_type: "Automatic",
          diet,
          cuisine,
          max_time: Number(maxTime),
        }),
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Unable to generate weekly plan.");
      }

      setPlan(data.plan || []);
      setSummary(data.summary || null);
      setContext(data.context || context);
    } catch (err) {
      console.error("Weekly planner error:", err);
      setError(err.message || "Unable to generate weekly plan.");
    } finally {
      setLoading(false);
    }
  }

  function openRecipe(recipe) {
    const payload = recipeButtonData(recipe);
    if (payload && onOpenRecipe) onOpenRecipe(payload);
  }

  function resetPlannerPantry() {
    if (!onResetPantry) {
      setError("Pantry reset is not connected in App.jsx.");
      return;
    }
    if (
      window.confirm(
        "Start a new weekly pantry cycle? This will remove the current pantry items from Mealora."
      )
    ) {
      onResetPantry();
    }
  }

  const inputStyle = {
    width: "100%",
    padding: "12px",
    borderRadius: 10,
    border: "1px solid #cbd5e1",
    boxSizing: "border-box",
    background: "#fff",
    fontSize: 15,
  };

  const cardStyle = {
    border: "1px solid #e2e8f0",
    borderRadius: 14,
    padding: 16,
    background: "#fff",
  };

  function RecipeCard({ label, recipe }) {
    if (!recipe) return null;
    return (
      <div style={cardStyle}>
        <div style={{ fontSize: 12, color: "#64748b", fontWeight: 800, marginBottom: 7 }}>
          {label}
        </div>
        <div style={{ fontWeight: 800, fontSize: 17 }}>{recipe.recipe_name}</div>
        <div style={{ marginTop: 8, display: "flex", gap: 6, flexWrap: "wrap", fontSize: 11 }}>
          <span style={{ padding: "4px 7px", borderRadius: 999, background: "#f0fdf4" }}>
            Suitability {pct(recipe.suitability_score)}
          </span>
          <span style={{ padding: "4px 7px", borderRadius: 999, background: "#eff6ff" }}>
            Pantry {pct(recipe.pantry_score)}
          </span>
          <span style={{ padding: "4px 7px", borderRadius: 999, background: "#faf5ff" }}>
            Regional {pct(recipe.region_score)}
          </span>
        </div>
        {recipe.matched_pantry?.length > 0 && (
          <div style={{ marginTop: 9, fontSize: 12, color: "#475569" }}>
            <strong>Uses:</strong> {recipe.matched_pantry.slice(0, 6).join(", ")}
          </div>
        )}
        <button
          type="button"
          className="secondary-button"
          onClick={() => openRecipe(recipe)}
          style={{ marginTop: 12 }}
        >
          📖 Open Recipe
        </button>
      </div>
    );
  }

  return (
    <div>
      <div style={{ marginBottom: 22 }}>
        <h2 style={{ margin: 0 }}>📅 Weekly Meal Planner</h2>
        <p style={{ marginTop: 8, color: "#64748b", lineHeight: 1.6 }}>
          Mealora builds a practical 7-day Indian household plan from your pantry,
          learned recipe suitability, regional context, diet, cooking time and live weather.
        </p>
      </div>

      <form
        onSubmit={generatePlan}
        style={{ padding: 20, borderRadius: 16, background: "#f8fafc", border: "1px solid #e2e8f0" }}
      >
        <h3 style={{ marginTop: 0 }}>👨‍👩‍👧 Household & Planning Settings</h3>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 12 }}>
          <label>
            Adults
            <input style={inputStyle} type="number" min="0" value={adults} onChange={(e) => setAdults(e.target.value)} />
          </label>
          <label>
            Children
            <input style={inputStyle} type="number" min="0" value={children} onChange={(e) => setChildren(e.target.value)} />
          </label>
          <label>
            Seniors
            <input style={inputStyle} type="number" min="0" value={seniors} onChange={(e) => setSeniors(e.target.value)} />
          </label>
          <label>
            Diet
            <select style={inputStyle} value={diet} onChange={(e) => setDiet(e.target.value)}>
              <option>Mixed</option>
              <option>Vegetarian</option>
              <option>Non-Vegetarian</option>
              <option>Vegan</option>
            </select>
          </label>
          <label>
            Maximum cooking time
            <select style={inputStyle} value={maxTime} onChange={(e) => setMaxTime(e.target.value)}>
              <option value="30">30 minutes</option>
              <option value="45">45 minutes</option>
              <option value="60">60 minutes</option>
              <option value="90">90 minutes</option>
            </select>
          </label>
        </div>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 18 }}>
          <button type="button" className={mode === "automatic" ? "primary-button" : "secondary-button"} onClick={() => setMode("automatic")}>
            🤖 Smart Automatic
          </button>
          <button type="button" className={mode === "custom" ? "primary-button" : "secondary-button"} onClick={() => setMode("custom")}>
            🎯 Custom
          </button>
        </div>

        {mode === "custom" && (
          <div style={{ marginTop: 18, padding: 18, borderRadius: 14, background: "#fff", border: "1px solid #e2e8f0" }}>
            <h3 style={{ marginTop: 0 }}>🎯 Custom Preferences</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
              <label>
                Week starting date
                <input style={inputStyle} type="date" value={date} onChange={(e) => setDate(e.target.value)} />
              </label>
              <label>
                Location
                <input style={inputStyle} value={locationCity} onChange={(e) => setLocationCity(e.target.value)} placeholder="e.g. Hyderabad" />
              </label>
              <label>
                Cuisine
                <select style={inputStyle} value={cuisine} onChange={(e) => setCuisine(e.target.value)}>
                  <option>Auto</option>
                  <option>South Indian</option>
                  <option>North Indian</option>
                  <option>West Indian</option>
                  <option>East Indian</option>
                  <option>Indian</option>
                </select>
              </label>
              <div style={{ display: "flex", alignItems: "end" }}>
                <button type="button" className="secondary-button" onClick={() => fetchMyLocation()} disabled={locationLoading} style={{ width: "100%" }}>
                  {locationLoading ? "📍 Fetching location..." : "📍 Use My Current Location"}
                </button>
              </div>
            </div>
            <button type="button" className="secondary-button" onClick={() => loadContext(true)} disabled={contextLoading} style={{ marginTop: 14 }}>
              {contextLoading ? "Refreshing context..." : "🔄 Refresh Live Context"}
            </button>
            <div style={{ marginTop: 10, color: "#64748b", fontSize: 13 }}>
              Snacks are selected automatically from the regional context; no manual snack switch is required.
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            width: "100%",
            marginTop: 18,
            padding: "16px 20px",
            borderRadius: 12,
            border: "none",
            background: "#1f2937",
            color: "white",
            fontSize: 18,
            fontWeight: 800,
            cursor: loading ? "wait" : "pointer",
          }}
        >
          {loading ? "⏳ Building your 7-day household plan..." : "✨ Generate 7-Day Complete Meal Plan"}
        </button>
      </form>

      {error && (
        <div style={{ marginTop: 16, padding: 14, borderRadius: 12, background: "#fff1f2", border: "1px solid #fecdd3", color: "#be123c" }}>
          {error}
        </div>
      )}

      <section style={{ marginTop: 24 }}>
        <h3>📡 Live Context Used by Mealora</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
          <div style={cardStyle}>
            <strong>📅 Date</strong>
            <div style={{ marginTop: 8, fontWeight: 800 }}>{context ? prettyDate(context.date) : "Loading..."}</div>
          </div>
          <div style={cardStyle}>
            <strong>🍽️ Planning Context</strong>
            <div style={{ marginTop: 8, fontWeight: 800 }}>7-day household plan</div>
          </div>
          <div style={cardStyle}>
            <strong>📍 Location</strong>
            <div style={{ marginTop: 8, fontWeight: 800 }}>{context?.location_city || "Fetching..."}</div>
          </div>
          <div style={cardStyle}>
            <strong>🌤️ Weather</strong>
            <div style={{ marginTop: 8, fontWeight: 800 }}>
              {context?.weather?.available
                ? `${context.weather.condition} · ${context.weather.temperature ?? "—"}°C`
                : "Fetching..."}
            </div>
          </div>
          <div style={cardStyle}>
            <strong>🎉 Occasion</strong>
            <div style={{ marginTop: 8, fontWeight: 800 }}>{context?.festival?.name || "No major festival"}</div>
          </div>
          <div style={cardStyle}>
            <strong>🥗 Diet & Cuisine</strong>
            <div style={{ marginTop: 8, fontWeight: 800 }}>{diet} · {context?.cuisine || cuisine}</div>
          </div>
        </div>
      </section>

      <section style={{ marginTop: 24, padding: 18, borderRadius: 16, background: "#eff6ff", border: "1px solid #dbeafe" }}>
        <h3 style={{ marginTop: 0 }}>🧺 Pantry status for this weekly plan</h3>
        <div style={{ color: "#475569" }}>{pantrySummary.total} pantry items · {pantrySummary.expiring} items expiring within 7 days</div>
        <button type="button" className="secondary-button" onClick={resetPlannerPantry} style={{ marginTop: 12 }}>
          🧹 Reset Pantry for New Week
        </button>
      </section>

      {summary && (
        <section style={{ marginTop: 24 }}>
          <h3>🧠 Weekly Optimization Summary</h3>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 12 }}>
            {[
              ["Meals planned", summary.meals_planned, ""],
              ["Pantry utilization", summary.pantry_utilization, "%"],
              ["Nutrition score", summary.nutrition_score, "%"],
              ["Context fit", summary.context_score, "%"],
              ["Regional fit", summary.regional_score, "%"],
            ].map(([label, value, suffix]) => (
              <div key={label} style={cardStyle}>
                <div style={{ color: "#64748b", fontSize: 14 }}>{label}</div>
                <div style={{ marginTop: 8, fontWeight: 900, fontSize: 28 }}>{value}{suffix}</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {plan.length > 0 && (
        <section style={{ marginTop: 28 }}>
          <h3>🍽️ Your 7-Day Complete Meal Plan</h3>
          <p style={{ color: "#64748b", lineHeight: 1.6 }}>
            Each day keeps the household structure simple: one main + one suitable side for breakfast, lunch and dinner. A regional snack is recommended automatically for the selected location.
          </p>

          <div style={{ display: "grid", gap: 18 }}>
            {plan.map((day) => (
              <div key={day.day} style={{ border: "1px solid #dbeafe", borderRadius: 18, padding: 20, background: "#fff" }}>
                <h3 style={{ margin: 0 }}>{day.day}</h3>
                <div style={{ color: "#64748b", marginTop: 4, marginBottom: 14 }}>{day.theme}</div>

                <div style={{ display: "grid", gap: 12 }}>
                  <div style={{ padding: 14, borderRadius: 16, background: "#fffaf0", border: "1px solid #fde68a" }}>
                    <div style={{ fontWeight: 900, color: "#92400e", marginBottom: 10 }}>🌅 Breakfast — Main + Side</div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: 10 }}>
                      <RecipeCard label="🍽️ Main" recipe={day.breakfast_main} />
                      <RecipeCard label="🥥 Side" recipe={day.breakfast_side} />
                    </div>
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: 12 }}>
                    <RecipeCard label="🍛 Lunch Main" recipe={day.lunch_main} />
                    <RecipeCard label="🥗 Lunch Side" recipe={day.lunch_side} />
                    <RecipeCard label="🌙 Dinner Main" recipe={day.dinner_main} />
                    <RecipeCard label="🥘 Dinner Side" recipe={day.dinner_side} />
                  </div>

                  {day.snack && (
                    <div style={{ padding: 14, borderRadius: 14, background: "#f8fafc", border: "1px solid #e2e8f0" }}>
                      <div style={{ fontWeight: 900, marginBottom: 10 }}>☕ Recommended Regional Snack</div>
                      <RecipeCard label="Regional Snack" recipe={day.snack} />
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
