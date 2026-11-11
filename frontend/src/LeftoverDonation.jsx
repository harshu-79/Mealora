import { useState } from "react";

const API = "http://127.0.0.1:5000/api/leftover-recommendations";

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function mapsSearchUrl(lat, lon) {
  return `https://www.google.com/maps/search/?api=1&query=NGOs+food+donation+near+${lat},${lon}`;
}

function mapsDirectionsUrl(destination) {
  return `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(destination)}`;
}

export default function LeftoverDonation({
  pantryItems = [],
  onOpenRecipe,
}) {
  const [leftover, setLeftover] = useState("");
  const [quantity, setQuantity] = useState("");
  const [unit, setUnit] = useState("serving");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);

  const [location, setLocation] = useState(null);
  const [locationLoading, setLocationLoading] = useState(false);

  const pantryMatches = pantryItems.filter((item) => {
    const a = normalizeText(leftover);
    const b = normalizeText(
      item.normalized_name || item.ingredient_name
    );
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
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          leftover: value,
          quantity: quantity || null,
          unit,
          top_k: 6,
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data.error || "Unable to find reuse recipes."
        );
      }

      setResults(data.recommendations || []);
      setSearched(true);
    } catch (err) {
      console.error("Leftover recommendation error:", err);
      setError(
        err.message || "Unable to find reuse recipes."
      );
    } finally {
      setLoading(false);
    }
  }

  function findNearbyNGOs() {
    if (!navigator.geolocation) {
      window.open(
        "https://www.google.com/maps/search/?api=1&query=NGOs+food+donation+near+me",
        "_blank"
      );
      return;
    }

    setLocationLoading(true);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const value = {
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        };

        setLocation(value);
        setLocationLoading(false);

        window.open(
          mapsSearchUrl(value.latitude, value.longitude),
          "_blank"
        );
      },
      () => {
        setLocationLoading(false);

        window.open(
          "https://www.google.com/maps/search/?api=1&query=NGOs+food+donation+near+me",
          "_blank"
        );
      },
      {
        enableHighAccuracy: false,
        timeout: 8000,
        maximumAge: 300000,
      }
    );
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
        <h2 style={{ margin: 0 }}>
          ♻️ Leftover Intelligence
        </h2>

        <p
          style={{
            marginTop: "8px",
            color: "#64748b",
          }}
        >
          Mealora uses ML-based ingredient similarity and
          suitability scoring to find recipes that can reuse
          your leftover before suggesting donation.
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
            gridTemplateColumns:
              "minmax(220px, 2fr) minmax(120px, 1fr) minmax(130px, 1fr)",
            gap: "12px",
          }}
        >

          <label>
            <span
              style={{
                display: "block",
                fontSize: "13px",
                fontWeight: 700,
                marginBottom: 6,
              }}
            >
              Leftover food / ingredient
            </span>

            <input
              value={leftover}
              onChange={(e) =>
                setLeftover(e.target.value)
              }
              placeholder="e.g. cooked rice, paneer, roti"
              style={{
                width: "100%",
                padding: "11px",
                borderRadius: "10px",
                border: "1px solid #cbd5e1",
                boxSizing: "border-box",
              }}
            />
          </label>

          <label>
            <span
              style={{
                display: "block",
                fontSize: "13px",
                fontWeight: 700,
                marginBottom: 6,
              }}
            >
              Quantity
            </span>

            <input
              type="number"
              min="0"
              step="any"
              value={quantity}
              onChange={(e) =>
                setQuantity(e.target.value)
              }
              placeholder="Optional"
              style={{
                width: "100%",
                padding: "11px",
                borderRadius: "10px",
                border: "1px solid #cbd5e1",
                boxSizing: "border-box",
              }}
            />
          </label>

          <label>
            <span
              style={{
                display: "block",
                fontSize: "13px",
                fontWeight: 700,
                marginBottom: 6,
              }}
            >
              Unit
            </span>

            <select
              value={unit}
              onChange={(e) =>
                setUnit(e.target.value)
              }
              style={{
                width: "100%",
                padding: "11px",
                borderRadius: "10px",
                border: "1px solid #cbd5e1",
                boxSizing: "border-box",
              }}
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
          <div
            style={{
              marginTop: 12,
              padding: "10px 12px",
              borderRadius: 10,
              background: "#ecfdf5",
              border: "1px solid #bbf7d0",
              fontSize: 13,
            }}
          >
            🧺 Pantry match:{" "}
            {pantryMatches
              .map(
                (x) =>
                  `${x.ingredient_name} (${x.quantity} ${x.unit})`
              )
              .join(", ")}
          </div>
        )}

        <div
          style={{
            display: "flex",
            gap: 10,
            marginTop: 16,
            flexWrap: "wrap",
          }}
        >
          <button
            className="primary-button"
            type="submit"
            disabled={loading}
          >
            {loading
              ? "Finding ML reuse matches..."
              : "🔎 Find Reuse Recipes"}
          </button>

          <button
            className="secondary-button"
            type="button"
            onClick={clearAll}
          >
            Clear
          </button>
        </div>
      </form>

      {error && (
        <div
          style={{
            marginTop: 18,
            padding: "13px 15px",
            borderRadius: 12,
            background: "#fff1f2",
            border: "1px solid #fecdd3",
            color: "#9f1239",
          }}
        >
          {error}
        </div>
      )}

      {results.length > 0 && (
        <div style={{ marginTop: 26 }}>

          <h3>🍳 ML-Powered Reuse Suggestions</h3>

          <p
            style={{
              color: "#64748b",
              fontSize: 13,
            }}
          >
            Mealora combines TF-IDF cosine similarity with
            ML suitability scoring to rank recipes for the
            entered leftover.
          </p>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(auto-fit, minmax(300px, 1fr))",
              gap: 16,
            }}
          >

            {results.map((recipe, index) => (

              <div
                key={
                  recipe.recipe_id ||
                  `${recipe.recipe_name}-${index}`
                }
                style={{
                  border: "1px solid #e2e8f0",
                  borderRadius: 14,
                  padding: 18,
                  background: "#fff",
                }}
              >

                <div
                  style={{
                    fontSize: 11,
                    fontWeight: 800,
                    color: "#64748b",
                  }}
                >
                  REUSE MATCH #{index + 1}
                </div>

                <h3 style={{ margin: "7px 0" }}>
                  {recipe.recipe_name}
                </h3>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns:
                      "repeat(2, minmax(0, 1fr))",
                    gap: 8,
                    marginBottom: 12,
                  }}
                >

                  <div
                    style={{
                      padding: 9,
                      borderRadius: 9,
                      background: "#f0fdf4",
                      fontSize: 12,
                    }}
                  >
                    <strong>
                      {Math.round(
                        Number(
                          recipe.match_score || 0
                        ) * 100
                      )}
                      %
                    </strong>
                    <br />
                    Ingredient match
                  </div>

                  <div
                    style={{
                      padding: 9,
                      borderRadius: 9,
                      background: "#eff6ff",
                      fontSize: 12,
                    }}
                  >
                    <strong>
                      {Math.round(
                        Number(
                          recipe.ml_score || 0
                        ) * 100
                      )}
                      %
                    </strong>
                    <br />
                    ML suitability
                  </div>

                </div>

                {recipe.matched_ingredients?.length > 0 && (
                  <p
                    style={{
                      fontSize: 13,
                      color: "#475569",
                      lineHeight: 1.5,
                    }}
                  >
                    <strong>Matched:</strong>{" "}
                    {recipe.matched_ingredients.join(", ")}
                  </p>
                )}

                {recipe.ingredients?.length > 0 && (
                  <details style={{ marginTop: 12 }}>
                    <summary
                      style={{
                        cursor: "pointer",
                        fontWeight: 700,
                      }}
                    >
                      🧺 Ingredients
                    </summary>

                    <ul
                      style={{
                        lineHeight: 1.7,
                        paddingLeft: 20,
                        fontSize: 13,
                      }}
                    >
                      {recipe.ingredients.map(
                        (item, i) => (
                          <li key={i}>{item}</li>
                        )
                      )}
                    </ul>
                  </details>
                )}

                {recipe.instructions?.length > 0 && (
                  <details style={{ marginTop: 10 }}>
                    <summary
                      style={{
                        cursor: "pointer",
                        fontWeight: 700,
                      }}
                    >
                      ��‍🍳 Cooking Procedure
                    </summary>

                    <ol
                      style={{
                        lineHeight: 1.7,
                        paddingLeft: 20,
                        fontSize: 13,
                      }}
                    >
                      {recipe.instructions.map(
                        (step, i) => (
                          <li
                            key={i}
                            style={{
                              marginBottom: 7,
                            }}
                          >
                            {step}
                          </li>
                        )
                      )}
                    </ol>
                  </details>
                )}

                <div
                  style={{
                    display: "flex",
                    gap: 8,
                    flexWrap: "wrap",
                    marginTop: 14,
                  }}
                >

                  <button
                    type="button"
                    className="primary-button"
                    onClick={() => {
                      if (!onOpenRecipe) {
                        console.error(
                          "LeftoverDonation: onOpenRecipe is not connected."
                        );
                        return;
                      }

                      if (!recipe?.recipe_id && !recipe?.recipe_name) {
                        console.error(
                          "LeftoverDonation: selected recipe has no recipe_id or recipe_name.",
                          recipe
                        );
                        return;
                      }

                      onOpenRecipe({
                        recipe_id: recipe.recipe_id || "",
                        recipe_name: recipe.recipe_name || "",
                      });
                    }}
                  >
                    📖 Open Full Recipe
                  </button>

                </div>

                <p
                  style={{
                    fontSize: 12,
                    color: "#64748b",
                    marginBottom: 0,
                    marginTop: 10,
                  }}
                >
                  {recipe.cuisine ||
                    "Indian cuisine"}
                  {recipe.meal_type
                    ? ` • ${recipe.meal_type}`
                    : ""}
                </p>

              </div>
            ))}
          </div>
        </div>
      )}

      {/* =====================================================
          DONATION / GOOGLE MAPS
         ===================================================== */}

      <div
        style={{
          marginTop: 30,
          padding: 22,
          borderRadius: 16,
          background: "#f0fdf4",
          border: "1px solid #bbf7d0",
        }}
      >

        <h3 style={{ marginTop: 0 }}>
          🤝 Can't Reuse It? Donate It.
        </h3>

        <p
          style={{
            color: "#475569",
            lineHeight: 1.6,
          }}
        >
          If the leftover is fresh, safely stored and
          suitable for donation, find nearby food-support
          organizations or use a verified donation channel.
        </p>

        <div
          style={{
            display: "flex",
            gap: 10,
            flexWrap: "wrap",
            marginBottom: 18,
          }}
        >

          <button
            type="button"
            className="primary-button"
            onClick={findNearbyNGOs}
            disabled={locationLoading}
          >
            {locationLoading
              ? "Finding your location..."
              : "📍 Find Nearby NGOs on Google Maps"}
          </button>

          <a
            href="https://www.google.com/maps/search/?api=1&query=NGOs+food+donation+near+me"
            target="_blank"
            rel="noreferrer"
            className="secondary-button"
            style={{
              textDecoration: "none",
              display: "inline-flex",
              alignItems: "center",
            }}
          >
            🗺️ Open Google Maps
          </a>

        </div>

        {location && (
          <div
            style={{
              background: "#fff",
              borderRadius: 12,
              overflow: "hidden",
              border: "1px solid #d1fae5",
            }}
          >

            <iframe
              title="Nearby food donation NGOs"
              src={`https://www.google.com/maps?q=NGOs+food+donation+near+${location.latitude},${location.longitude}&output=embed`}
              width="100%"
              height="330"
              style={{
                border: 0,
                display: "block",
              }}
              loading="lazy"
            />

            <div style={{ padding: 14 }}>

              <strong>
                📍 Nearby NGO search
              </strong>

              <p
                style={{
                  fontSize: 12,
                  color: "#64748b",
                  marginBottom: 12,
                }}
              >
                Open a result in Google Maps to see the
                exact distance, travel time and directions
                from your current location.
              </p>

              <a
                href={mapsSearchUrl(
                  location.latitude,
                  location.longitude
                )}
                target="_blank"
                rel="noreferrer"
                className="primary-button"
                style={{
                  textDecoration: "none",
                  display: "inline-flex",
                }}
              >
                🧭 View Nearby NGOs + Distance
              </a>

            </div>
          </div>
        )}

        <h4 style={{ marginTop: 22 }}>
          📦 Food Donation Channels
        </h4>

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(240px, 1fr))",
            gap: 12,
          }}
        >

          <a
            href="https://www.feedingindia.org/"
            target="_blank"
            rel="noreferrer"
            style={{
              padding: 16,
              borderRadius: 12,
              background: "#fff",
              border: "1px solid #d1d5db",
              textDecoration: "none",
              color: "#1f2937",
            }}
          >
            <strong>🍚 Feeding India</strong>
            <p
              style={{
                fontSize: 12,
                color: "#64748b",
                marginBottom: 0,
              }}
            >
              Feeding India donation channel; its FAQ
              also lists Zomato and Blinkit as donation
              channels.
            </p>
          </a>

          <a
            href="https://www.robinhoodarmy.com/"
            target="_blank"
            rel="noreferrer"
            style={{
              padding: 16,
              borderRadius: 12,
              background: "#fff",
              border: "1px solid #d1d5db",
              textDecoration: "none",
              color: "#1f2937",
            }}
          >
            <strong>🥘 Robin Hood Army</strong>
            <p
              style={{
                fontSize: 12,
                color: "#64748b",
                marginBottom: 0,
              }}
            >
              Volunteer-based organization that connects
              surplus food with people who need it.
            </p>
          </a>

        </div>

        <p
          style={{
            fontSize: 11,
            color: "#78716c",
            marginTop: 14,
            marginBottom: 0,
          }}
        >
          Only donate food that is safe and suitable for
          consumption. Do not donate spoiled, expired or
          improperly stored food.
        </p>

      </div>

      {searched && results.length === 0 && (
        <div
          style={{
            marginTop: 22,
            padding: 18,
            borderRadius: 14,
            background: "#fff7ed",
            border: "1px solid #fed7aa",
          }}
        >
          <h3 style={{ marginTop: 0 }}>
            🤝 No strong reuse match
          </h3>

          <p style={{ color: "#475569" }}>
            Mealora could not find a strong reuse recipe.
            Consider the donation options above if the
            food is safe to donate.
          </p>
        </div>
      )}

    </div>
  );
}
