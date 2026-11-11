import { useState } from "react";

const API = "http://127.0.0.1:5000/api/leftover-recommendations";

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export default function LeftoverDonation({ pantryItems = [] }) {
  const [leftover, setLeftover] = useState("");
  const [quantity, setQuantity] = useState("");
  const [unit, setUnit] = useState("serving");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);

  const pantryMatches = pantryItems.filter((item) => {
    const a = normalizeText(leftover);
    const b = normalizeText(item.normalized_name || item.ingredient_name);
    return a && b && (a === b || a.includes(b) || b.includes(a));
  });

  async function findReuseRecipes(event) {
    event.preventDefault();
    setError("");
    setResults([]);
    setSearched(false);

    const value = leftover.trim();
    if (!value) {
      setError("Enter the leftover ingredient or food name.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          leftover: value,
          quantity: quantity || null,
          unit,
          top_k: 6,
        }),
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || "Unable to find reuse recipes.");
      }

      setResults(data.recommendations || []);
      setSearched(true);
    } catch (err) {
      console.error("Leftover recommendation error:", err);
      setError(err.message || "Unable to find reuse recipes.");
    } finally {
      setLoading(false);
    }
  }

  function clearAll() {
    setLeftover("");
    setQuantity("");
    setUnit("serving");
    setResults([]);
    setError("");
    setSearched(false);
  }

  return (
    <div>
      <div style={{ marginBottom: "22px" }}>
        <h2 style={{ margin: 0 }}>♻️ Leftover Intelligence</h2>
        <p style={{ marginTop: "8px", color: "#64748b" }}>
          Mealora checks whether a leftover can be reused in another recipe before suggesting donation.
        </p>
      </div>

      <form
        onSubmit={findReuseRecipes}
        style={{
          padding: "20px",
          borderRadius: "16px",
          background: "#f8fafc",
          border: "1px solid #e2e8f0",
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(220px, 2fr) minmax(120px, 1fr) minmax(130px, 1fr)",
            gap: "12px",
          }}
        >
          <label>
            <span style={{ display: "block", fontSize: "13px", fontWeight: 700, marginBottom: 6 }}>
              Leftover food / ingredient
            </span>
            <input
              value={leftover}
              onChange={(e) => setLeftover(e.target.value)}
              placeholder="e.g. cooked rice, paneer, potatoes"
              style={{ width: "100%", padding: "11px", borderRadius: "10px", border: "1px solid #cbd5e1", boxSizing: "border-box" }}
            />
          </label>

          <label>
            <span style={{ display: "block", fontSize: "13px", fontWeight: 700, marginBottom: 6 }}>
              Quantity
            </span>
            <input
              type="number"
              min="0"
              step="any"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              placeholder="Optional"
              style={{ width: "100%", padding: "11px", borderRadius: "10px", border: "1px solid #cbd5e1", boxSizing: "border-box" }}
            />
          </label>

          <label>
            <span style={{ display: "block", fontSize: "13px", fontWeight: 700, marginBottom: 6 }}>
              Unit
            </span>
            <select
              value={unit}
              onChange={(e) => setUnit(e.target.value)}
              style={{ width: "100%", padding: "11px", borderRadius: "10px", border: "1px solid #cbd5e1", boxSizing: "border-box" }}
            >
              <option value="serving">Serving</option>
              <option value="piece">Piece</option>
              <option value="cup">Cup</option>
              <option value="g">g</option>
              <option value="kg">kg</option>
            </select>
          </label>
        </div>

        {pantryMatches.length > 0 && (
          <div style={{ marginTop: 12, padding: "10px 12px", borderRadius: 10, background: "#ecfdf5", border: "1px solid #bbf7d0", fontSize: 13 }}>
            🧺 Pantry match: {pantryMatches.map((x) => `${x.ingredient_name} (${x.quantity} ${x.unit})`).join(", ")}
          </div>
        )}

        <div style={{ display: "flex", gap: 10, marginTop: 16, flexWrap: "wrap" }}>
          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? "Finding reuse recipes..." : "🔎 Find Reuse Recipes"}
          </button>
          <button className="secondary-button" type="button" onClick={clearAll}>
            Clear
          </button>
        </div>
      </form>

      {error && (
        <div style={{ marginTop: 18, padding: "13px 15px", borderRadius: 12, background: "#fff1f2", border: "1px solid #fecdd3", color: "#9f1239" }}>
          {error}
        </div>
      )}

      {results.length > 0 && (
        <div style={{ marginTop: 22 }}>
          <h3>🍳 Reuse Suggestions</h3>
          <p style={{ color: "#64748b", fontSize: 13 }}>
            These recipes contain or closely match the leftover ingredient.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16 }}>
            {results.map((recipe, index) => (
              <div key={recipe.recipe_id || `${recipe.recipe_name}-${index}`} style={{ border: "1px solid #e2e8f0", borderRadius: 14, padding: 16, background: "#fff" }}>
                <div style={{ fontSize: 11, fontWeight: 800, color: "#64748b" }}>REUSE MATCH #{index + 1}</div>
                <h3 style={{ margin: "7px 0" }}>{recipe.recipe_name}</h3>
                <div style={{ padding: "9px 10px", borderRadius: 9, background: "#f0fdf4", fontSize: 13 }}>
                  <strong>{Math.round(Number(recipe.match_score || 0) * 100)}% ingredient match</strong>
                </div>
                {recipe.matched_ingredients?.length > 0 && (
                  <p style={{ fontSize: 13, color: "#475569", lineHeight: 1.5 }}>
                    Matched: {recipe.matched_ingredients.join(", ")}
                  </p>
                )}
                <p style={{ fontSize: 12, color: "#64748b", marginBottom: 0 }}>
                  {recipe.cuisine || "Indian cuisine"} {recipe.meal_type ? `• ${recipe.meal_type}` : ""}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {searched && results.length === 0 && (
        <div style={{ marginTop: 22, padding: 18, borderRadius: 14, background: "#fff7ed", border: "1px solid #fed7aa" }}>
          <h3 style={{ marginTop: 0 }}>🤝 Donation Suggestion</h3>
          <p style={{ color: "#475569", lineHeight: 1.6 }}>
            Mealora could not find a strong recipe reuse match for this leftover. If the food is still fresh, safely stored, and eligible under your local food-sharing rules, consider donating it to a nearby community food organization.
          </p>
          <p style={{ fontSize: 12, color: "#78716c", marginBottom: 0 }}>
            Do not donate spoiled, expired, or improperly stored food.
          </p>
        </div>
      )}

      {!searched && (
        <div style={{ marginTop: 20, padding: 16, borderRadius: 14, background: "#eff6ff", border: "1px solid #dbeafe", color: "#475569", fontSize: 13 }}>
          <strong>Example:</strong> Enter <em>cooked rice</em> and Mealora can look for recipes such as lemon rice, fried rice, or other recipes containing rice.
        </div>
      )}
    </div>
  );
}
