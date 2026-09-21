import { useEffect, useMemo, useRef, useState } from "react";
import { createWorker } from "tesseract.js";
import { supabase } from "./supabaseClient";
import { getPantryFeatures } from "./pantryFeatures";
import { matchGroceryText } from "./ocrMatcher";
import {
  preprocessImage,
  getOCRParameters,
  extractUsefulOCRText,
  parseGroceryRows,
} from "./ocrUtils";

const RECOMMENDATION_API = "http://127.0.0.1:5000/api/recommendations";
const RECIPE_ASSISTANT_API = "http://127.0.0.1:5000/api/recipe-assistant";

const RECIPE_LANGUAGES = [
  "English",
  "தமிழ்",
  "తెలుగు",
  "ಕನ್ನಡ",
  "മലയാളം",
  "हिन्दी",
  "मराठी",
  "বাংলা",
];

const CATEGORY_MAP = {
  rice: "grains",
  "basmati rice": "grains",
  wheat: "grains",
  "wheat flour": "grains",
  flour: "grains",
  oats: "grains",

  tomato: "vegetables",
  tomatoes: "vegetables",
  spinach: "vegetables",
  potato: "vegetables",
  potatoes: "vegetables",
  onion: "vegetables",
  onions: "vegetables",
  carrot: "vegetables",
  carrots: "vegetables",
  cucumber: "vegetables",
  cucumbers: "vegetables",
  capsicum: "vegetables",
  capsicums: "vegetables",
  "bell pepper": "vegetables",
  brinjal: "vegetables",
  brinjals: "vegetables",
  eggplant: "vegetables",

  apple: "fruits",
  apples: "fruits",
  banana: "fruits",
  bananas: "fruits",
  orange: "fruits",
  oranges: "fruits",
  mango: "fruits",
  mangoes: "fruits",

  milk: "dairy",
  curd: "dairy",
  yogurt: "dairy",
  yoghurt: "dairy",
  butter: "dairy",
  cheese: "dairy",
  paneer: "dairy",

  egg: "protein",
  eggs: "protein",
  chicken: "protein",
  fish: "protein",

  "toor dal": "pulses",
  "moong dal": "pulses",
  "urad dal": "pulses",
  "chana dal": "pulses",

  salt: "spices",
  sugar: "other",
  oil: "other",
  "cooking oil": "other",
  "mustard oil": "other",
  "mustard seed oil": "other",
  "sarson oil": "other",
  "sarson ka tel": "other",
};

const NUMBER_WORDS = {
  a: 1,
  an: 1,
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
  eleven: 11,
  twelve: 12,
  thirteen: 13,
  fourteen: 14,
  fifteen: 15,
  sixteen: 16,
  seventeen: 17,
  eighteen: 18,
  nineteen: 19,
  twenty: 20,
};

const UNIT_ALIASES = {
  kg: "kg",
  kgs: "kg",
  kilo: "kg",
  kilos: "kg",
  kilogram: "kg",
  kilograms: "kg",

  g: "g",
  gram: "g",
  grams: "g",

  l: "litre",
  liter: "litre",
  liters: "litre",
  litre: "litre",
  litres: "litre",

  ml: "ml",
  milliliter: "ml",
  milliliters: "ml",
  millilitre: "ml",
  millilitres: "ml",

  piece: "piece",
  pieces: "piece",
  egg: "piece",
  eggs: "piece",

  dozen: "dozen",
};

function normalizeIngredientName(name) {
  const cleanedName = name
    .trim()
    .toLowerCase()
    .replace(/[.,!?]/g, "")
    .replace(/\s+/g, " ");

  const singularForms = {
    potatoes: "potato",
    tomatoes: "tomato",
    onions: "onion",
    carrots: "carrot",
    cucumbers: "cucumber",
    capsicums: "capsicum",
    brinjals: "brinjal",
    eggplants: "eggplant",
    apples: "apple",
    bananas: "banana",
    oranges: "orange",
    mangoes: "mango",
    eggs: "egg",
    lemons: "lemon",
    limes: "lime",
    chillies: "chilli",
    chilies: "chilli",
    peppers: "pepper",
  };

  return singularForms[cleanedName] || cleanedName;
}

function getCategoryForIngredient(normalizedName) {
  if (CATEGORY_MAP[normalizedName]) {
    return CATEGORY_MAP[normalizedName];
  }

  const matchedKey = Object.keys(CATEGORY_MAP).find(
    (key) =>
      normalizedName.includes(key) || key.includes(normalizedName)
  );

  return matchedKey ? CATEGORY_MAP[matchedKey] : "other";
}

function getTodayDate() {
  return new Date().toISOString().split("T")[0];
}

function addDaysToDate(dateString, days) {
  const date = new Date(`${dateString}T00:00:00`);
  date.setDate(date.getDate() + Number(days));

  return date.toISOString().split("T")[0];
}

function formatDateToISO(year, month, day) {
  const date = new Date(year, month - 1, day);

  if (
    date.getFullYear() !== year ||
    date.getMonth() !== month - 1 ||
    date.getDate() !== day
  ) {
    return null;
  }

  return date.toISOString().split("T")[0];
}

function parseNaturalDate(text, baseDate = new Date()) {
  const cleaned = text
    .trim()
    .toLowerCase()
    .replace(/,/g, "")
    .replace(/\bthe\b/g, "")
    .replace(/\bof\b/g, "")
    .replace(/\s+/g, " ");

  const today = new Date(baseDate);
  today.setHours(0, 0, 0, 0);

  if (cleaned === "today") {
    return today.toISOString().split("T")[0];
  }

  if (cleaned === "yesterday") {
    const date = new Date(today);
    date.setDate(date.getDate() - 1);

    return date.toISOString().split("T")[0];
  }

  if (cleaned === "tomorrow") {
    const date = new Date(today);
    date.setDate(date.getDate() + 1);

    return date.toISOString().split("T")[0];
  }

  const relativeMatch = cleaned.match(
    /(?:in\s+)?(\d+)\s+(day|days|week|weeks)\b/
  );

  if (relativeMatch) {
    const amount = Number(relativeMatch[1]);
    const unit = relativeMatch[2];

    const days = unit.startsWith("week")
      ? amount * 7
      : amount;

    return addDaysToDate(
      today.toISOString().split("T")[0],
      days
    );
  }

  const ordinalNumericMatch = cleaned.match(
    /^(\d{1,2})(?:st|nd|rd|th)?[/-](\d{1,2})[/-](\d{2,4})$/
  );

  if (ordinalNumericMatch) {
    let year = Number(ordinalNumericMatch[3]);

    if (year < 100) {
      year += 2000;
    }

    return formatDateToISO(
      year,
      Number(ordinalNumericMatch[2]),
      Number(ordinalNumericMatch[1])
    );
  }

  const monthNames = {
    january: 1,
    jan: 1,
    february: 2,
    feb: 2,
    march: 3,
    mar: 3,
    april: 4,
    apr: 4,
    may: 5,
    june: 6,
    jun: 6,
    july: 7,
    jul: 7,
    august: 8,
    aug: 8,
    september: 9,
    sep: 9,
    sept: 9,
    october: 10,
    oct: 10,
    november: 11,
    nov: 11,
    december: 12,
    dec: 12,
  };

  const monthDateMatch = cleaned.match(
    /^(?:on\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)(?:\s+(\d{4}))?$/
  );

  if (monthDateMatch) {
    const day = Number(monthDateMatch[1]);
    const month = monthNames[monthDateMatch[2]];

    if (!month) {
      return null;
    }

    const year = monthDateMatch[3]
      ? Number(monthDateMatch[3])
      : today.getFullYear();

    let result = formatDateToISO(year, month, day);

    if (!result) {
      return null;
    }

    if (!monthDateMatch[3]) {
      const resultDate = new Date(`${result}T00:00:00`);

      if (resultDate < today) {
        result = formatDateToISO(year + 1, month, day);
      }
    }

    return result;
  }

  const monthDayMatch = cleaned.match(
    /^(?:on\s+)?([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?$/
  );

  if (monthDayMatch) {
    const month = monthNames[monthDayMatch[1]];
    const day = Number(monthDayMatch[2]);

    if (!month) {
      return null;
    }

    const year = monthDayMatch[3]
      ? Number(monthDayMatch[3])
      : today.getFullYear();

    let result = formatDateToISO(year, month, day);

    if (!result) {
      return null;
    }

    if (!monthDayMatch[3]) {
      const resultDate = new Date(`${result}T00:00:00`);

      if (resultDate < today) {
        result = formatDateToISO(year + 1, month, day);
      }
    }

    return result;
  }

  return null;
}

function extractExplicitDates(text) {
  const normalized = text
    .toLowerCase()
    .replace(/,/g, "")
    .replace(/\s+/g, " ");

  const dates = {
    purchase_date: null,
    expiry_date: null,
  };

  const purchaseMatch = normalized.match(
    /(?:bought|purchased|purchase(?:d)?|got)\s+(?:on\s+)?(.+?)(?=(?:expires?|expiry|expiration)\b|$)/
  );

  if (purchaseMatch) {
    dates.purchase_date = parseNaturalDate(
      purchaseMatch[1].trim()
    );
  }

  const expiryMatch = normalized.match(
    /(?:expires?|expiry|expiration)\s+(?:on\s+)?(.+)$/
  );

  if (expiryMatch) {
    dates.expiry_date = parseNaturalDate(
      expiryMatch[1].trim()
    );
  }

  return dates;
}

function isLowStock(item) {
  if (
    item.low_stock_threshold === null ||
    item.low_stock_threshold === undefined
  ) {
    return false;
  }

  return Number(item.quantity) <= Number(item.low_stock_threshold);
}

function getDaysToExpiry(expiryDate) {
  if (!expiryDate) {
    return null;
  }

  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const expiry = new Date(`${expiryDate}T00:00:00`);
  expiry.setHours(0, 0, 0, 0);

  const difference = expiry.getTime() - today.getTime();

  return Math.ceil(
    difference / (1000 * 60 * 60 * 24)
  );
}

function getExpiryStatus(expiryDate) {
  const days = getDaysToExpiry(expiryDate);

  if (days === null) {
    return {
      label: "No expiry date",
      type: "normal",
    };
  }

  if (days <= 0) {
    return {
      label: days < 0 ? "Expired" : "Expires today",
      type: "expired",
    };
  }

  if (days <= 3) {
    return {
      label: `Expires in ${days} day${
        days === 1 ? "" : "s"
      }`,
      type: "critical",
    };
  }

  if (days <= 7) {
    return {
      label: `Expires in ${days} days`,
      type: "soon",
    };
  }

  return {
    label: `Expires in ${days} days`,
    type: "normal",
  };
}

function parseQuantity(text) {
  const cleaned = text.trim().toLowerCase();

  if (NUMBER_WORDS[cleaned] !== undefined) {
    return NUMBER_WORDS[cleaned];
  }

  const numericValue = Number(cleaned);

  return Number.isNaN(numericValue)
    ? null
    : numericValue;
}


function parseRecipeIngredientAmount(text) {
  const cleaned = String(text || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");

  const unitPattern =
    "(kg|kgs|kilo|kilos|kilogram|kilograms|g|gram|grams|l|liter|liters|litre|litres|ml|milliliter|milliliters|millilitre|millilitres|piece|pieces|pc|pcs|dozen|cup|cups|tbsp|tablespoon|tablespoons|tsp|teaspoon|teaspoons)";

  const numberPattern =
    "(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|\\d+(?:\\.\\d+)?)";

  const match = cleaned.match(
    new RegExp(
      "^" + numberPattern +
      "\\s*(?:" + unitPattern + ")?\\s*(?:of\\s+)?(.+?)$",
      "i"
    )
  );

  if (!match) {
    return { quantity: null, unit: null, ingredient: cleaned };
  }

  const unitAliases = {
    kg: "kg", kgs: "kg", kilo: "kg", kilos: "kg",
    kilogram: "kg", kilograms: "kg",
    g: "g", gram: "g", grams: "g",
    l: "litre", liter: "litre", liters: "litre",
    litre: "litre", litres: "litre",
    ml: "ml", milliliter: "ml", milliliters: "ml",
    millilitre: "ml", millilitres: "ml",
    piece: "piece", pieces: "piece", pc: "piece",
    pcs: "piece", dozen: "dozen",
    cup: "cup", cups: "cup",
    tbsp: "tbsp", tablespoon: "tbsp", tablespoons: "tbsp",
    tsp: "tsp", teaspoon: "tsp", teaspoons: "tsp",
  };

  return {
    quantity: parseQuantity(match[1]),
    unit: match[2] ? unitAliases[match[2].toLowerCase()] || match[2].toLowerCase() : null,
    ingredient: (match[3] || "")
      .replace(/\b(finely|roughly|thinly|thickly|chopped|sliced|diced|minced|grated|crushed|ground|fresh|small|medium|large|optional|to taste|as needed)\b/gi, " ")
      .replace(/\s+/g, " ")
      .trim(),
  };
}

function convertRecipeQuantity(quantity, fromUnit, toUnit) {
  if (quantity === null || quantity === undefined) return null;

  const from = fromUnit || toUnit;
  const to = toUnit || from;
  if (!from || !to) return null;
  if (from === to) return Number(quantity);

  const mass = { g: 1, kg: 1000 };
  const volume = { ml: 1, litre: 1000 };

  if (mass[from] && mass[to]) {
    return (Number(quantity) * mass[from]) / mass[to];
  }

  if (volume[from] && volume[to]) {
    return (Number(quantity) * volume[from]) / volume[to];
  }

  if (from === "dozen" && to === "piece") return Number(quantity) * 12;
  if (from === "piece" && to === "dozen") return Number(quantity) / 12;

  return null;
}

function recipeIngredientMatchesPantry(recipeIngredient, pantryItem) {
  const recipeNormalized = normalizeIngredientName(recipeIngredient);
  const pantryNormalized = normalizeIngredientName(
    pantryItem.normalized_name || pantryItem.ingredient_name || ""
  );

  if (!recipeNormalized || !pantryNormalized) return false;
  if (recipeNormalized === pantryNormalized) return true;

  const recipeWords = recipeNormalized.split(" ").filter((word) => word.length >= 3);
  const pantryWords = pantryNormalized.split(" ").filter((word) => word.length >= 3);

  return (
    recipeWords.length > 0 &&
    pantryWords.length > 0 &&
    (
      recipeWords.every((word) => pantryWords.includes(word)) ||
      pantryNormalized.includes(recipeNormalized) ||
      recipeNormalized.includes(pantryNormalized)
    )
  );
}

function cleanIngredientPhrase(phrase) {
  return phrase
    .replace(
      /^(please\s+)?(can you\s+)?(could you\s+)?/i,
      ""
    )
    .replace(
      /^(add|put|place|include|store|keep|save|buy|bought|i bought|i have)\s+/i,
      ""
    )
    .replace(/\s+/g, " ")
    .trim();
}

function parseSpokenIngredients(text) {
  let cleanedText = text
    .toLowerCase()
    .replace(/[!?]/g, "")
    .replace(/\bplease\b/g, "")
    .replace(/\bcan you\b/g, "")
    .replace(/\bcould you\b/g, "")
    .replace(/\bfor me\b/g, "")
    .replace(/\binto my pantry\b/g, "")
    .replace(/\bin my pantry\b/g, "")
    .replace(/\bto my pantry\b/g, "")
    .trim();

  cleanedText = cleanedText
    .replace(/\s+and\s+/g, ",")
    .replace(/\s*,\s*/g, ",");

  const parts = cleanedText
    .split(",")
    .map((part) => cleanIngredientPhrase(part))
    .filter(Boolean);

  const results = [];

  for (const part of parts) {
    const pattern =
      /^(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|\d+(?:\.\d+)?)\s*(kg|kgs|kilo|kilos|kilogram|kilograms|g|gram|grams|l|liter|liters|litre|litres|ml|milliliter|milliliters|millilitre|millilitres|piece|pieces|dozen|egg|eggs)?\s*(?:of\s+)?(.+)$/i;

    const match = part.match(pattern);

    if (!match) {
      continue;
    }

    const quantity = parseQuantity(match[1]);

    if (!quantity || quantity <= 0) {
      continue;
    }

    let unit = match[2]
      ? UNIT_ALIASES[match[2].toLowerCase()]
      : null;

    let ingredient = match[3].trim();

    ingredient = ingredient
      .replace(
        /\b(bought|purchased|expires?|expiry|expiration)\b.*$/i,
        ""
      )
      .trim();

    if (!unit) {
      if (/^(egg|eggs)$/i.test(ingredient)) {
        unit = "piece";
      } else {
        continue;
      }
    }

    ingredient = ingredient
      .replace(/^(of|some|the)\s+/i, "")
      .trim();

    const normalizedName =
      normalizeIngredientName(ingredient);

    if (!normalizedName) {
      continue;
    }

    results.push({
      ingredient_name:
        ingredient.charAt(0).toUpperCase() +
        ingredient.slice(1),
      normalized_name: normalizedName,
      quantity,
      unit,
      category:
        getCategoryForIngredient(normalizedName),
    });
  }

  return results;
}

async function getShelfLife(normalizedName) {
  const { data, error } = await supabase
    .from("ingredient_shelf_life")
    .select("default_shelf_life_days")
    .eq("normalized_name", normalizedName)
    .maybeSingle();

  if (error) {
    console.error(
      "Shelf-life lookup error:",
      error
    );

    return null;
  }

  return data?.default_shelf_life_days ?? null;
}

async function estimateExpiry(
  normalizedName,
  purchaseDate
) {
  const shelfLife = await getShelfLife(
    normalizedName
  );

  if (!shelfLife) {
    return null;
  }

  return addDaysToDate(
    purchaseDate,
    shelfLife
  );
}


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


function App() {
  const [currentPage, setCurrentPage] = useState("pantry");
  const [activeTab, setActiveTab] = useState("manual");
  const [recommendationResults, setRecommendationResults] = useState([]);
  const [festivalSpecial, setFestivalSpecial] = useState(null);
  const [recommendationLoading, setRecommendationLoading] = useState(false);
  const [recommendationError, setRecommendationError] = useState("");

  const [mealType, setMealType] = useState("Dinner");
  const [diet, setDiet] = useState("Vegetarian");
  const [cuisine, setCuisine] = useState("South Indian");
  const [maxTime, setMaxTime] = useState(30);
  const [topK, setTopK] = useState(10);
  const [autoMode, setAutoMode] = useState(true);
  const [location, setLocation] = useState(null);
  const [contextInfo, setContextInfo] = useState(null);
  const [assistantLanguage, setAssistantLanguage] = useState("English");
  const [assistantRecipe, setAssistantRecipe] = useState(null);
  // Keep the original dataset recipe separate from Gemini's translated output.
  // Language changes must ALWAYS translate the original recipe, never the translated text.
  const [assistantSourceRecipe, setAssistantSourceRecipe] = useState(null);
  const assistantRequestRef = useRef(0);
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [assistantError, setAssistantError] = useState("");
  const [cookingRecipe, setCookingRecipe] = useState(false);
  const [customDate, setCustomDate] = useState(getTodayDate());
  const [customTime, setCustomTime] = useState("19:00");
  const [customLocation, setCustomLocation] = useState("Chennai");

  const [pantryItems, setPantryItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  const [searchTerm, setSearchTerm] = useState("");

  const [ingredientName, setIngredientName] = useState("");
  const [quantity, setQuantity] = useState("");
  const [unit, setUnit] = useState("");
  const [category, setCategory] = useState("");
  const [purchaseDate, setPurchaseDate] = useState(getTodayDate());
  const [expiryDate, setExpiryDate] = useState("");
  const [expiryIsEstimated, setExpiryIsEstimated] = useState(false);
  const [lowStockThreshold, setLowStockThreshold] = useState("");

  const [editingItemId, setEditingItemId] = useState(null);

  const [isListening, setIsListening] = useState(false);
  const [voiceText, setVoiceText] = useState("");
  const [voiceItems, setVoiceItems] = useState([]);

  const [ocrText, setOcrText] = useState("");
  const [ocrItems, setOcrItems] = useState([]);
  const [isOCRProcessing, setIsOCRProcessing] = useState(false);
  const [ocrProgress, setOcrProgress] = useState(0);

  const recognitionRef = useRef(null);
  const ocrWorkerRef = useRef(null);

  async function openRecipeAssistant(recipe, languageOverride = null, rememberSource = true) {
    const requestedLanguage = languageOverride || assistantLanguage;

    if (!recipe?.recipe_id) {
      setAssistantError("The selected recipe does not have a valid recipe ID.");
      return;
    }

    // Store ONLY the original Mealora dataset identity. Never replace this with
    // Gemini's translated recipe name. This is critical for language switching.
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

    try {
      const response = await fetch(RECIPE_ASSISTANT_API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recipe_id: sourceRecipe.recipe_id,
          recipe_name: sourceRecipe.recipe_name,
          language: requestedLanguage,
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Unable to prepare the selected recipe.");
      }

      // Ignore an older Gemini response if the user changed language again.
      if (requestId !== assistantRequestRef.current) return;

      setAssistantRecipe(data.recipe);
    } catch (error) {
      if (requestId !== assistantRequestRef.current) return;
      console.error("Gemini recipe assistant error:", error);
      setAssistantError(error.message || "Unable to prepare the recipe.");
    } finally {
      if (requestId === assistantRequestRef.current) {
        setAssistantLoading(false);
      }
    }
  }

  async function handleAssistantLanguageChange(language) {
    setAssistantLanguage(language);

    // If a recipe is open, ALWAYS translate the original Mealora recipe.
    // Do not use assistantRecipe.recipe_name because Gemini may already have
    // translated it into Telugu/Tamil/etc.
    if (assistantSourceRecipe?.recipe_id) {
      await openRecipeAssistant(
        assistantSourceRecipe,
        language,
        false
      );
    }
  }


  async function cookRecipeAndDeductFromPantry() {
    if (!assistantRecipe) return;

    const recipeIngredients = assistantRecipe.ingredients || [];
    if (recipeIngredients.length === 0) {
      setAssistantError("This recipe has no ingredient list to deduct from the pantry.");
      return;
    }

    setCookingRecipe(true);
    setAssistantError("");
    setSuccessMessage("");

    try {
      const deductions = [];
      const unmatched = [];

      for (const ingredientText of recipeIngredients) {
        const parsed = parseRecipeIngredientAmount(ingredientText);
        const recipeIngredient = parsed.ingredient || String(ingredientText).trim().toLowerCase();

        const candidates = pantryItems.filter((item) =>
          recipeIngredientMatchesPantry(recipeIngredient, item)
        );

        if (candidates.length === 0) {
          unmatched.push(ingredientText);
          continue;
        }

        candidates.sort((a, b) => {
          const aExact =
            normalizeIngredientName(recipeIngredient) ===
            normalizeIngredientName(a.normalized_name || a.ingredient_name || "");
          const bExact =
            normalizeIngredientName(recipeIngredient) ===
            normalizeIngredientName(b.normalized_name || b.ingredient_name || "");
          return Number(bExact) - Number(aExact);
        });

        const pantryItem = candidates[0];

        if (parsed.quantity === null || parsed.quantity <= 0) {
          unmatched.push(`${ingredientText} (quantity could not be determined)`);
          continue;
        }

        const pantryUnit = pantryItem.unit || "";
        const deduction = convertRecipeQuantity(parsed.quantity, parsed.unit, pantryUnit);

        if (deduction === null || Number.isNaN(deduction) || deduction <= 0) {
          unmatched.push(
            `${ingredientText} (unit not compatible with pantry unit ${pantryUnit})`
          );
          continue;
        }

        const currentQuantity = Number(pantryItem.quantity);

        if (Number.isNaN(currentQuantity) || currentQuantity < deduction) {
          unmatched.push(
            `${ingredientText} (only ${pantryItem.quantity} ${pantryUnit} available)`
          );
          continue;
        }

        deductions.push({
          pantryItem,
          amount: deduction,
          unit: pantryUnit,
          remaining: currentQuantity - deduction,
        });
      }

      if (deductions.length === 0) {
        throw new Error(
          "No recipe ingredients could be safely matched to your pantry with compatible quantities."
        );
      }

      const deductionSummary = deductions
        .map((item) => `• ${item.pantryItem.ingredient_name}: ${item.amount} ${item.unit}`)
        .join("\n");

      const confirmed = window.confirm(
        `Cook ${assistantRecipe.recipe_name}?\n\nMealora will deduct these ingredients from your pantry:\n\n${deductionSummary}\n\nIngredients that cannot be safely deducted will be left unchanged. Continue?`
      );

      if (!confirmed) return;

      for (const deduction of deductions) {
        const { error: updateError } = await supabase
          .from("pantry_items")
          .update({
            quantity: deduction.remaining,
            updated_at: new Date().toISOString(),
          })
          .eq("id", deduction.pantryItem.id);

        if (updateError) {
          console.error("Error deducting cooked ingredient:", updateError);
          throw new Error(`Unable to update ${deduction.pantryItem.ingredient_name}.`);
        }
      }

      await fetchPantryItems();

      const skippedMessage =
        unmatched.length > 0
          ? ` ${unmatched.length} ingredient(s) could not be safely deducted and were left unchanged.`
          : "";

      setSuccessMessage(
        `🍳 ${assistantRecipe.recipe_name} cooked! Pantry quantities were updated.${skippedMessage}`
      );

      setAssistantRecipe(null);
      setAssistantSourceRecipe(null);
      window.speechSynthesis?.cancel();
    } catch (error) {
      console.error("Cook recipe / pantry deduction error:", error);
      setAssistantError(
        error.message || "Unable to deduct recipe ingredients from the pantry."
      );
    } finally {
      setCookingRecipe(false);
    }
  }

  function speakRecipe() {
    if (!assistantRecipe || !window.speechSynthesis) return;

    window.speechSynthesis.cancel();

    const text = [
      assistantRecipe.recipe_name,
      "Ingredients.",
      ...(assistantRecipe.ingredients || []),
      "Instructions.",
      ...(assistantRecipe.instructions || []),
    ].join(". ");

    const utterance = new SpeechSynthesisUtterance(text);

    const voiceLanguageMap = {
      English: "en-IN",
      "தமிழ்": "ta-IN",
      "తెలుగు": "te-IN",
      "ಕನ್ನಡ": "kn-IN",
      "മലയാളം": "ml-IN",
      "हिन्दी": "hi-IN",
      "मराठी": "mr-IN",
      "বাংলা": "bn-IN",
    };

    utterance.lang = voiceLanguageMap[assistantLanguage] || "en-IN";
    utterance.rate = 0.9;
    window.speechSynthesis.speak(utterance);
  }

  async function loadLiveContext() {
    try {
      let coords = location;

      if (!coords && navigator.geolocation) {
        coords = await new Promise((resolve) => {
          navigator.geolocation.getCurrentPosition(
            (position) => {
              const value = {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
              };
              setLocation(value);
              resolve(value);
            },
            () => resolve(null),
            { enableHighAccuracy: false, timeout: 5000, maximumAge: 300000 }
          );
        });
      }

      const payload = {
        mode: autoMode ? "automatic" : "custom",
        latitude: coords?.latitude ?? null,
        longitude: coords?.longitude ?? null,
      };

      if (!autoMode) {
        payload.date = customDate;
        payload.time = customTime;
        payload.location_city = customLocation;
        payload.location = customLocation;
        payload.meal_type = mealType;
        payload.diet = diet;
        payload.cuisine = cuisine;
        payload.max_time = Number(maxTime);
      } else {
        payload.meal_type = "Auto";
        payload.diet = "Auto";
        payload.cuisine = "Auto";
        payload.max_time = 30;
      }

      const response = await fetch("http://127.0.0.1:5000/api/context", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (response.ok && data.success) {
        setContextInfo(data.context);
        setRecommendationError("");
      } else {
        throw new Error(data.error || "Context fetch failed");
      }
    } catch (error) {
      console.error("Live context error:", error);
      setRecommendationError(
        `Backend error: ${error.message || "Unable to fetch live context."}`
      );
    }
  }


  function getFestivalPantryStatus(recipe) {
    const ingredients = Array.isArray(recipe?.ingredients)
      ? recipe.ingredients
      : [];

    const missing = [];
    let availableCount = 0;

    for (const ingredientText of ingredients) {
      const parsed = parseRecipeIngredientAmount(ingredientText);
      const ingredient = parsed.ingredient || String(ingredientText).trim().toLowerCase();
      const match = pantryItems.some((item) =>
        recipeIngredientMatchesPantry(ingredient, item)
      );

      if (match) {
        availableCount += 1;
      } else {
        missing.push(String(ingredientText));
      }
    }

    return {
      total: ingredients.length,
      available: availableCount,
      missing,
    };
  }

  function openShoppingSite(url) {
    window.open(url, "_blank", "noopener,noreferrer");
  }

  async function loadRecommendations() {
    setRecommendationLoading(true);
    setRecommendationError("");

    try {
      let coords = location;

      if (!coords && navigator.geolocation) {
        coords = await new Promise((resolve) => {
          navigator.geolocation.getCurrentPosition(
            (position) => {
              const value = {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
              };
              setLocation(value);
              resolve(value);
            },
            () => resolve(null),
            {
              enableHighAccuracy: false,
              timeout: 5000,
              maximumAge: 300000,
            }
          );
        });
      }

      const payload = {
        reference_recipe: "Balu shahi",
        top_k: Number(topK),
        latitude: coords?.latitude ?? null,
        longitude: coords?.longitude ?? null,
        mode: autoMode ? "automatic" : "custom",
      };

      if (autoMode) {
        payload.meal_type = "Auto";
        payload.diet = "Auto";
        payload.cuisine = "Auto";
        payload.max_time = 30;
      } else {
        payload.meal_type = mealType;
        payload.diet = diet;
        payload.cuisine = cuisine;
        payload.max_time = Number(maxTime);
        payload.date = customDate;
        payload.time = customTime;
        payload.location_city = customLocation;
        payload.location = customLocation;
      }

      const response = await fetch(RECOMMENDATION_API, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(
          data.error || "Unable to load recommendations."
        );
      }

      setRecommendationResults(data.recommendations || []);
      setFestivalSpecial(data.festival_special || null);
      setContextInfo(data.context || null);
    } catch (error) {
      console.error("Recommendation API error:", error);
      setRecommendationResults([]);
      setFestivalSpecial(null);
      setRecommendationError(
        "Unable to connect to the Mealora recommendation engine. Make sure the Flask backend is running on port 5000."
      );
    } finally {
      setRecommendationLoading(false);
    }
  }

  useEffect(() => {
    if (currentPage === "recommendations") {
      loadLiveContext();
    }
  }, [currentPage, autoMode]);

  async function fetchPantryItems() {
    setLoading(true);
    setError("");

    const { data, error } = await supabase
      .from("pantry_items")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) {
      console.error(
        "Error fetching pantry items:",
        error
      );

      setError("Unable to load pantry items.");
      setLoading(false);
      return;
    }

    setPantryItems(data || []);
    setLoading(false);
  }

  useEffect(() => {
    fetchPantryItems();

  
  return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }

      if (ocrWorkerRef.current) {
        ocrWorkerRef.current
          .terminate()
          .catch(() => {});
      }
    };
  }, []);

  const recommendationPantry = useMemo(() => {
    return pantryItems
      .map((item) => item.ingredient_name)
      .filter(Boolean);
  }, [pantryItems]);

  function resetForm() {
    setIngredientName("");
    setQuantity("");
    setUnit("");
    setCategory("");
    setPurchaseDate(getTodayDate());
    setExpiryDate("");
    setExpiryIsEstimated(false);
    setLowStockThreshold("");
    setEditingItemId(null);
  }

  async function recalculateEstimatedExpiry(
    normalizedName,
    selectedPurchaseDate
  ) {
    const estimated = await estimateExpiry(
      normalizedName,
      selectedPurchaseDate
    );

    if (estimated) {
      setExpiryDate(estimated);
      setExpiryIsEstimated(true);
    }
  }

  async function handleIngredientNameBlur() {
    const normalizedName =
      normalizeIngredientName(ingredientName);

    if (!normalizedName || editingItemId) {
      return;
    }

    if (!category) {
      setCategory(
        getCategoryForIngredient(normalizedName)
      );
    }

    if (!expiryDate) {
      await recalculateEstimatedExpiry(
        normalizedName,
        purchaseDate
      );
    }
  }

  async function handlePurchaseDateChange(newDate) {
    setPurchaseDate(newDate);

    if (
      !editingItemId &&
      (!expiryDate || expiryIsEstimated)
    ) {
      const normalizedName =
        normalizeIngredientName(ingredientName);

      if (normalizedName) {
        await recalculateEstimatedExpiry(
          normalizedName,
          newDate
        );
      }
    }
  }

  function handleExpiryDateChange(newDate) {
    setExpiryDate(newDate);
    setExpiryIsEstimated(false);
  }

  function startEditing(item) {
    setEditingItemId(item.id);
    setIngredientName(item.ingredient_name || "");
    setQuantity(item.quantity || "");
    setUnit(item.unit || "");
    setCategory(item.category || "");
    setPurchaseDate(
      item.purchase_date || getTodayDate()
    );
    setExpiryDate(item.expiry_date || "");
    setExpiryIsEstimated(false);
    setLowStockThreshold(item.low_stock_threshold ?? "");

    setActiveTab("manual");
    setError("");
    setSuccessMessage("");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  }

  async function handleAddOrUpdateIngredient(event) {
    event.preventDefault();

    setError("");
    setSuccessMessage("");

    const normalizedName =
      normalizeIngredientName(ingredientName);

    const numericQuantity = Number(quantity);

    const numericThreshold =
      lowStockThreshold === ""
        ? null
        : Number(lowStockThreshold);

    if (!normalizedName) {
      setError("Please enter an ingredient name.");
      return;
    }

    if (!numericQuantity || numericQuantity <= 0) {
      setError("Please enter a valid quantity.");
      return;
    }

    if (!unit) {
      setError("Please select a unit.");
      return;
    }

    if (!category) {
      setError("Please select a category.");
      return;
    }

    if (!purchaseDate) {
      setError("Please select a purchase date.");
      return;
    }

    if (
      numericThreshold !== null &&
      (Number.isNaN(numericThreshold) ||
        numericThreshold < 0)
    ) {
      setError(
        "Please enter a valid low-stock threshold."
      );
      return;
    }

    let finalExpiryDate = expiryDate || null;

    if (!finalExpiryDate && !editingItemId) {
      finalExpiryDate = await estimateExpiry(
        normalizedName,
        purchaseDate
      );
    }

    if (editingItemId) {
      const { error: updateError } = await supabase
        .from("pantry_items")
        .update({
          ingredient_name: ingredientName.trim(),
          normalized_name: normalizedName,
          quantity: numericQuantity,
          unit,
          category,
          purchase_date: purchaseDate,
          expiry_date: finalExpiryDate,
          low_stock_threshold: numericThreshold,
          updated_at: new Date().toISOString(),
        })
        .eq("id", editingItemId);

      if (updateError) {
        console.error(
          "Error updating pantry item:",
          updateError
        );

        setError("Unable to update the pantry item.");
        return;
      }

      setSuccessMessage(
        `${ingredientName.trim()} was updated successfully.`
      );

      resetForm();
      await fetchPantryItems();
      return;
    }

    const {
      data: existingItem,
      error: existingError,
    } = await supabase
      .from("pantry_items")
      .select("*")
      .eq("normalized_name", normalizedName)
      .eq("unit", unit)
      .maybeSingle();

    if (existingError) {
      console.error(
        "Error checking existing item:",
        existingError
      );

      setError(
        "Unable to check existing pantry items."
      );
      return;
    }

    if (existingItem) {
      const newQuantity =
        Number(existingItem.quantity) +
        numericQuantity;

      const { error: updateError } = await supabase
        .from("pantry_items")
        .update({
          quantity: newQuantity,
          purchase_date: purchaseDate,
          expiry_date:
            finalExpiryDate ||
            existingItem.expiry_date,
          low_stock_threshold:
            numericThreshold ??
            existingItem.low_stock_threshold,
          updated_at: new Date().toISOString(),
        })
        .eq("id", existingItem.id);

      if (updateError) {
        console.error(
          "Error updating pantry item:",
          updateError
        );

        setError(
          "Unable to update the pantry item."
        );
        return;
      }

      setSuccessMessage(
        `${ingredientName.trim()} already existed, so its quantity was updated.`
      );
    } else {
      const { error: insertError } = await supabase
        .from("pantry_items")
        .insert({
          ingredient_name: ingredientName.trim(),
          normalized_name: normalizedName,
          quantity: numericQuantity,
          unit,
          category,
          purchase_date: purchaseDate,
          expiry_date: finalExpiryDate,
          low_stock_threshold: numericThreshold,
        });

      if (insertError) {
        console.error(
          "Error adding pantry item:",
          insertError
        );

        setError("Unable to add the pantry item.");
        return;
      }

      setSuccessMessage(
        `${ingredientName.trim()} was added to your pantry.`
      );
    }

    resetForm();
    await fetchPantryItems();
  }

  async function handleDeleteIngredient(
    itemId,
    itemName
  ) {
    const confirmed = window.confirm(
      `Are you sure you want to delete ${itemName} from your pantry?`
    );

    if (!confirmed) {
      return;
    }

    setError("");
    setSuccessMessage("");

    const { error: deleteError } = await supabase
      .from("pantry_items")
      .delete()
      .eq("id", itemId);

    if (deleteError) {
      console.error(
        "Error deleting pantry item:",
        deleteError
      );

      setError(
        "Unable to delete the pantry item."
      );
      return;
    }

    if (editingItemId === itemId) {
      resetForm();
    }

    setSuccessMessage(
      `${itemName} was removed from your pantry.`
    );

    await fetchPantryItems();
  }

  async function enrichVoiceItems(parsedItems, dates) {
    const defaultPurchaseDate =
      dates.purchase_date || getTodayDate();

    return Promise.all(
      parsedItems.map(async (item) => {
        const estimatedExpiry = dates.expiry_date
          ? dates.expiry_date
          : await estimateExpiry(
              item.normalized_name,
              defaultPurchaseDate
            );

        return {
          ...item,
          purchase_date: defaultPurchaseDate,
          expiry_date: estimatedExpiry || "",
          expiry_is_estimated: !dates.expiry_date,
        };
      })
    );
  }

  function startVoiceRecognition() {
    setError("");
    setSuccessMessage("");
    setVoiceText("");
    setVoiceItems([]);

    const SpeechRecognition =
      window.SpeechRecognition ||
      window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      setError(
        "Voice recognition is not supported in this browser. Please use Chrome."
      );
      return;
    }

    const recognition = new SpeechRecognition();

    recognition.lang = "en-IN";
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = async (event) => {
      const transcript =
        event.results[0][0].transcript;

      setVoiceText(transcript);

      const parsedItems =
        parseSpokenIngredients(transcript);

      if (parsedItems.length === 0) {
        setError(
          "I heard your speech, but could not extract the ingredient details."
        );
        return;
      }

      const dates =
        extractExplicitDates(transcript);

      const enrichedItems =
        await enrichVoiceItems(
          parsedItems,
          dates
        );

      setVoiceItems(enrichedItems);
      setError("");
    };

    recognition.onerror = (event) => {
      console.error(
        "Speech recognition error:",
        event.error
      );

      setError(
        `Voice recognition error: ${event.error}`
      );

      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
  }

  function updateVoiceItem(index, field, value) {
    setVoiceItems((currentItems) =>
      currentItems.map(
        (item, itemIndex) =>
          itemIndex === index
            ? {
                ...item,
                [field]:
                  field === "quantity"
                    ? Number(value)
                    : value,
              }
            : item
      )
    );
  }

  function removeVoiceItem(index) {
    setVoiceItems((currentItems) =>
      currentItems.filter(
        (_, itemIndex) =>
          itemIndex !== index
      )
    );
  }

  function updateVoiceExpiry(index, newDate) {
    setVoiceItems((currentItems) =>
      currentItems.map(
        (item, itemIndex) =>
          itemIndex === index
            ? {
                ...item,
                expiry_date: newDate,
                expiry_is_estimated: false,
              }
            : item
      )
    );
  }

  async function confirmVoiceItems() {
    setError("");
    setSuccessMessage("");

    if (voiceItems.length === 0) {
      setError(
        "There are no voice items to add."
      );
      return;
    }

    for (const item of voiceItems) {
      if (
        !item.ingredient_name ||
        !item.quantity ||
        !item.unit ||
        !item.purchase_date
      ) {
        setError(
          "Please make sure every voice item has complete information."
        );
        return;
      }

      const normalizedName =
        normalizeIngredientName(
          item.ingredient_name
        );

      const {
        data: existingItem,
        error: existingError,
      } = await supabase
        .from("pantry_items")
        .select("*")
        .eq(
          "normalized_name",
          normalizedName
        )
        .eq("unit", item.unit)
        .maybeSingle();

      if (existingError) {
        console.error(
          "Error checking voice item:",
          existingError
        );

        setError(
          "Unable to check the pantry before adding voice items."
        );
        return;
      }

      if (existingItem) {
        const { error: updateError } =
          await supabase
            .from("pantry_items")
            .update({
              quantity:
                Number(existingItem.quantity) +
                Number(item.quantity),
              purchase_date:
                item.purchase_date,
              expiry_date:
                item.expiry_date ||
                existingItem.expiry_date,
              category:
                item.category ||
                existingItem.category,
              updated_at:
                new Date().toISOString(),
            })
            .eq(
              "id",
              existingItem.id
            );

        if (updateError) {
          console.error(
            "Error updating voice item:",
            updateError
          );

          setError(
            `Unable to update ${item.ingredient_name}.`
          );
          return;
        }
      } else {
        const { error: insertError } =
          await supabase
            .from("pantry_items")
            .insert({
              ingredient_name:
                item.ingredient_name.trim(),
              normalized_name:
                normalizedName,
              quantity: Number(
                item.quantity
              ),
              unit: item.unit,
              category:
                item.category || "other",
              purchase_date:
                item.purchase_date,
              expiry_date:
                item.expiry_date || null,
              low_stock_threshold: null,
            });

        if (insertError) {
          console.error(
            "Error inserting voice item:",
            insertError
          );

          setError(
            `Unable to add ${item.ingredient_name}.`
          );
          return;
        }
      }
    }

    setVoiceItems([]);
    setVoiceText("");

    setSuccessMessage(
      "Voice ingredients were added to your pantry."
    );

    await fetchPantryItems();
  }

  async function handleOCRUpload(event) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    setError("");
    setSuccessMessage("");
    setOcrText("");
    setOcrItems([]);
    setIsOCRProcessing(true);
    setOcrProgress(0);

    try {
      if (!ocrWorkerRef.current) {
        ocrWorkerRef.current =
          await createWorker("eng", 1, {
            logger: (message) => {
              if (
                typeof message.progress ===
                "number"
              ) {
                setOcrProgress(
                  Math.round(
                    message.progress * 100
                  )
                );
              }
            },
          });
      }

      const worker =
        ocrWorkerRef.current;

      await worker.setParameters(
        getOCRParameters()
      );

      setOcrProgress(5);

      const processedImage =
        await preprocessImage(file);

      setOcrProgress(10);

      const firstPass =
        await worker.recognize(
          processedImage
        );

      const firstText =
        firstPass?.data?.text || "";

      setOcrProgress(55);

      const secondPass =
        await worker.recognize(file);

      const secondText =
        secondPass?.data?.text || "";

      setOcrProgress(90);

      const combinedText =
        extractUsefulOCRText(
          `${firstText}\n${secondText}`
        );

      setOcrText(
        combinedText
      );

      const parsedRows =
        parseGroceryRows(
          combinedText
        );

      if (
        parsedRows.length === 0
      ) {
        setError(
          "OCR could read the bill, but no probable grocery rows were found. You can still inspect the extracted text above."
        );

        return;
      }

      /*
       * IMPORTANT:
       * Do NOT fall back to raw OCR text here.
       *
       * A row is allowed into the verification
       * screen only when the canonical grocery
       * vocabulary recognizes it with sufficient
       * confidence.
       *
       * This prevents:
       * - addresses
       * - phone numbers
       * - invoice numbers
       * - account numbers
       * - HSN codes
       * - unrelated OCR noise
       *
       * from becoming pantry items.
       */
      const candidateItems = [];

      for (const item of parsedRows) {
        const rawIngredient =
          item.ingredientName ||
          item.ingredient ||
          "";

        const cleanedRawIngredient =
          rawIngredient
            .trim()
            .replace(/\s+/g, " ");

        if (
          cleanedRawIngredient.length <
          4
        ) {
          continue;
        }

        let vocabularyMatch =
          await matchGroceryText(
            cleanedRawIngredient
          );

        /*
         * Targeted fallback for common Indian
         * grocery terminology that may not yet
         * exist in the canonical CSV.
         */
        if (!vocabularyMatch) {
          const candidate =
            cleanedRawIngredient
              .toLowerCase()
              .replace(/[^a-z\s]/g, " ")
              .replace(/\s+/g, " ")
              .trim();

          if (
            candidate.includes("mustard oil") ||
            candidate.includes("mustard seed oil") ||
            candidate.includes("sarson oil") ||
            candidate.includes("sarson ka tel")
          ) {
            vocabularyMatch = {
              canonical_name: "mustard oil",
              alias: cleanedRawIngredient,
              category: "other",
              confidence: 1,
            };
          }
        }

        if (!vocabularyMatch) {
          continue;
        }

        const confidence =
          Number(
            vocabularyMatch.confidence ||
            0
          );

        /*
         * The matcher itself allows moderately
         * fuzzy matches, but OCR pantry insertion
         * needs a stricter threshold so short
         * garbage such as "Rea" or "Th Lr"
         * cannot accidentally match food names.
         */
        if (
          confidence < 0.78
        ) {
          continue;
        }

        const matchedName =
          vocabularyMatch.canonical_name;

        const normalizedName =
          normalizeIngredientName(
            matchedName
          );

        if (!normalizedName) {
          continue;
        }

        const purchaseDate =
          getTodayDate();

        const estimatedExpiry =
          await estimateExpiry(
            normalizedName,
            purchaseDate
          );

        candidateItems.push({
          ingredient_name:
            matchedName,

          normalized_name:
            normalizedName,

          quantity:
            item.quantity,

          unit:
            item.unit,

          category:
            vocabularyMatch.category ||
            getCategoryForIngredient(
              normalizedName
            ),

          purchase_date:
            purchaseDate,

          expiry_date:
            estimatedExpiry || "",

          expiry_is_estimated:
            Boolean(
              estimatedExpiry
            ),

          source_line:
            item.sourceLine,

          dataset_match:
            true,

          match_confidence:
            confidence,
        });
      }

      /*
       * Remove duplicate products that came from
       * the two OCR passes.
       *
       * Keep separate quantities only when the
       * unit differs.
       */
      const uniqueItems =
        new Map();

      for (const item of candidateItems) {
        const key =
          `${item.normalized_name}|${item.unit}`;

        if (!uniqueItems.has(key)) {
          uniqueItems.set(
            key,
            item
          );
        }
      }

      const enrichedItems =
        Array.from(
          uniqueItems.values()
        );

      if (
        enrichedItems.length === 0
      ) {
        setError(
          "OCR could read the bill, but no grocery products could be confidently recognized from the Mealora vocabulary. You can still inspect the extracted text above."
        );

        return;
      }

      setOcrItems(
        enrichedItems
      );

      setOcrProgress(100);
    } catch (ocrError) {
      console.error(
        "OCR processing error:",
        ocrError
      );

      setError(
        "Unable to process the grocery bill."
      );
    } finally {
      setIsOCRProcessing(false);
    }
  }

  function updateOCRItem(
    index,
    field,
    value
  ) {
    setOcrItems((currentItems) =>
      currentItems.map(
        (item, itemIndex) =>
          itemIndex === index
            ? {
                ...item,
                [field]:
                  field === "quantity"
                    ? Number(value)
                    : value,
              }
            : item
      )
    );
  }

  function updateOCRExpiry(
    index,
    newDate
  ) {
    setOcrItems((currentItems) =>
      currentItems.map(
        (item, itemIndex) =>
          itemIndex === index
            ? {
                ...item,
                expiry_date:
                  newDate,
                expiry_is_estimated:
                  false,
              }
            : item
      )
    );
  }

  function removeOCRItem(index) {
    setOcrItems((currentItems) =>
      currentItems.filter(
        (_, itemIndex) =>
          itemIndex !== index
      )
    );
  }

  async function confirmOCRItems() {
    setError("");
    setSuccessMessage("");

    if (ocrItems.length === 0) {
      setError(
        "There are no OCR items to add."
      );
      return;
    }

    for (const item of ocrItems) {
      if (
        !item.ingredient_name ||
        !item.quantity ||
        !item.unit ||
        !item.purchase_date
      ) {
        setError(
          "Please complete every detected item before adding it."
        );
        return;
      }

      const normalizedName =
        normalizeIngredientName(
          item.ingredient_name
        );

      const {
        data: existingItem,
        error: existingError,
      } = await supabase
        .from("pantry_items")
        .select("*")
        .eq(
          "normalized_name",
          normalizedName
        )
        .eq("unit", item.unit)
        .maybeSingle();

      if (existingError) {
        console.error(
          "Error checking OCR item:",
          existingError
        );

        setError(
          "Unable to check the pantry before adding OCR items."
        );
        return;
      }

      if (existingItem) {
        const { error: updateError } =
          await supabase
            .from("pantry_items")
            .update({
              quantity:
                Number(existingItem.quantity) +
                Number(item.quantity),
              purchase_date:
                item.purchase_date,
              expiry_date:
                item.expiry_date ||
                existingItem.expiry_date,
              category:
                item.category ||
                existingItem.category,
              updated_at:
                new Date().toISOString(),
            })
            .eq(
              "id",
              existingItem.id
            );

        if (updateError) {
          console.error(
            "Error updating OCR item:",
            updateError
          );

          setError(
            `Unable to update ${item.ingredient_name}.`
          );
          return;
        }
      } else {
        const { error: insertError } =
          await supabase
            .from("pantry_items")
            .insert({
              ingredient_name:
                item.ingredient_name.trim(),
              normalized_name:
                normalizedName,
              quantity: Number(
                item.quantity
              ),
              unit: item.unit,
              category:
                item.category || "other",
              purchase_date:
                item.purchase_date,
              expiry_date:
                item.expiry_date || null,
              low_stock_threshold: null,
            });

        if (insertError) {
          console.error(
            "Error inserting OCR item:",
            insertError
          );

          setError(
            `Unable to add ${item.ingredient_name}.`
          );
          return;
        }
      }
    }

    setOcrItems([]);
    setOcrText("");

    setSuccessMessage(
      "Verified grocery-bill items were added to your pantry."
    );

    await fetchPantryItems();
  }

  const categoryCount = useMemo(() => {
    return new Set(
      pantryItems
        .map((item) => item.category)
        .filter(Boolean)
    ).size;
  }, [pantryItems]);

  const expiringItems = useMemo(() => {
    return pantryItems
      .filter((item) => {
        const days = getDaysToExpiry(
          item.expiry_date
        );

        return days !== null && days <= 7;
      })
      .sort(
        (a, b) =>
          getDaysToExpiry(a.expiry_date) -
          getDaysToExpiry(b.expiry_date)
      );
  }, [pantryItems]);

  const lowStockItems = useMemo(() => {
    return pantryItems.filter((item) =>
      isLowStock(item)
    );
  }, [pantryItems]);

  const filteredPantryItems =
    useMemo(() => {
      const search =
        searchTerm
          .trim()
          .toLowerCase();

      if (!search) {
        return pantryItems;
      }

      return pantryItems.filter(
        (item) =>
          item.ingredient_name
            .toLowerCase()
            .includes(search) ||
          item.normalized_name
            .toLowerCase()
            .includes(search) ||
          item.category
            .toLowerCase()
            .includes(search)
      );
    }, [
      pantryItems,
      searchTerm,
    ]);

  const analyticsSummary = useMemo(() => {
    const totalItems = pantryItems.length;

    const categoryMap = {};

    pantryItems.forEach((item) => {
      const rawCategory =
        item.category || "other";

      const key =
        rawCategory
          .trim()
          .toLowerCase();

      const displayName =
        key.charAt(0).toUpperCase() +
        key.slice(1);

      if (!categoryMap[key]) {
        categoryMap[key] = {
          name: displayName,
          count: 0,
        };
      }

      categoryMap[key].count += 1;
    });

    const categoryEntries =
      Object.values(categoryMap).sort(
        (a, b) => b.count - a.count
      );

    const itemsWithExpiry =
      pantryItems.filter(
        (item) => item.expiry_date
      );

    const itemsWithoutExpiry =
      totalItems -
      itemsWithExpiry.length;

    return {
      totalItems,
      categoryEntries,
      itemsWithExpiry:
        itemsWithExpiry.length,
      itemsWithoutExpiry,
      expiringSoonCount:
        expiringItems.length,
      lowStockCount:
        lowStockItems.length,
    };
  }, [
    pantryItems,
    expiringItems,
    lowStockItems,
  ]);


  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Mealora</h1>
          <p>
            Smart Pantry Management
          </p>
        </div>

        <button className="profile-button">
          Profile
        </button>
      </header>

      <div
        style={{
          display: "flex",
          gap: "10px",
          justifyContent: "center",
          padding: "16px 24px",
          borderBottom: "1px solid #e5e7eb",
          background: "#ffffff",
        }}
      >
        <button
          className={currentPage === "pantry" ? "tab active" : "tab"}
          onClick={() => setCurrentPage("pantry")}
        >
          🥕 My Pantry
        </button>

        <button
          className={currentPage === "recommendations" ? "tab active" : "tab"}
          onClick={() => setCurrentPage("recommendations")}
        >
          🍽️ Recommendations
        </button>
      </div>

      <main className="pantry-container">
        {currentPage === "pantry" && (
          <>
            <section className="overview-section">
          <h2>My Pantry</h2>

          <div className="overview-grid">
            <div className="overview-card">
              <span>
                Total Items
              </span>

              <strong>
                {pantryItems.length}
              </strong>
            </div>

            <div className="overview-card">
              <span>
                Categories
              </span>

              <strong>
                {categoryCount}
              </strong>
            </div>

            <div className="overview-card">
              <span>
                Expiring Soon
              </span>

              <strong>
                {expiringItems.length}
              </strong>
            </div>

            <div className="overview-card">
              <span>
                Low Stock
              </span>

              <strong>
                {lowStockItems.length}
              </strong>
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-heading">
            <div>
              <h2>
                {editingItemId
                  ? "Edit Ingredient"
                  : "Add to Pantry"}
              </h2>

              <p>
                {editingItemId
                  ? "Update the pantry information below."
                  : "Add ingredients manually, by voice, or by scanning a bill."}
              </p>
            </div>
          </div>

          {!editingItemId && (
            <div className="input-tabs">
              <button
                className={
                  activeTab === "manual"
                    ? "tab active"
                    : "tab"
                }
                onClick={() =>
                  setActiveTab("manual")
                }
              >
                Manual
              </button>

              <button
                className={
                  activeTab === "voice"
                    ? "tab active"
                    : "tab"
                }
                onClick={() =>
                  setActiveTab("voice")
                }
              >
                Voice
              </button>

              <button
                className={
                  activeTab === "ocr"
                    ? "tab active"
                    : "tab"
                }
                onClick={() =>
                  setActiveTab("ocr")
                }
              >
                Scan Bill
              </button>
            </div>
          )}

          {(activeTab === "manual" ||
            editingItemId) && (
            <form
              className="form-grid"
              onSubmit={
                handleAddOrUpdateIngredient
              }
            >
              <input
                type="text"
                placeholder="Ingredient name"
                value={ingredientName}
                onChange={(event) =>
                  setIngredientName(
                    event.target.value
                  )
                }
                onBlur={
                  handleIngredientNameBlur
                }
              />

              <input
                type="number"
                placeholder="Quantity"
                min="0"
                step="any"
                value={quantity}
                onChange={(event) =>
                  setQuantity(
                    event.target.value
                  )
                }
              />

              <select
                value={unit}
                onChange={(event) =>
                  setUnit(event.target.value)
                }
              >
                <option value="">
                  Select unit
                </option>
                <option value="kg">
                  kg
                </option>
                <option value="g">
                  g
                </option>
                <option value="litre">
                  Litre
                </option>
                <option value="ml">
                  ml
                </option>
                <option value="piece">
                  Piece
                </option>
                <option value="dozen">
                  Dozen
                </option>
              </select>

              <select
                value={category}
                onChange={(event) =>
                  setCategory(
                    event.target.value
                  )
                }
              >
                <option value="">
                  Select category
                </option>
                <option value="vegetables">
                  Vegetables
                </option>
                <option value="fruits">
                  Fruits
                </option>
                <option value="grains">
                  Grains
                </option>
                <option value="dairy">
                  Dairy
                </option>
                <option value="protein">
                  Protein
                </option>
                <option value="pulses">
                  Pulses
                </option>
                <option value="spices">
                  Spices
                </option>
                <option value="other">
                  Other
                </option>
              </select>

              <label className="date-field">
                <span>
                  Purchase date
                </span>

                <input
                  type="date"
                  value={purchaseDate}
                  onChange={(event) =>
                    handlePurchaseDateChange(
                      event.target.value
                    )
                  }
                />
              </label>

              <label className="date-field">
                <span>
                  Expiry date
                </span>

                <input
                  type="date"
                  value={expiryDate}
                  onChange={(event) =>
                    handleExpiryDateChange(
                      event.target.value
                    )
                  }
                />

                {expiryIsEstimated &&
                  expiryDate && (
                    <small className="field-note">
                      Estimated from
                      shelf-life reference.
                    </small>
                  )}
              </label>

              <label className="date-field">
                <span>
                  Low-stock threshold
                </span>

                <input
                  type="number"
                  min="0"
                  step="any"
                  placeholder="Optional"
                  value={lowStockThreshold}
                  onChange={(event) =>
                    setLowStockThreshold(
                      event.target.value
                    )
                  }
                />
              </label>

              <div className="form-actions">
                <button
                  className="primary-button"
                  type="submit"
                >
                  {editingItemId
                    ? "Save Changes"
                    : "Add Ingredient"}
                </button>

                {editingItemId && (
                  <button
                    className="secondary-button"
                    type="button"
                    onClick={resetForm}
                  >
                    Cancel
                  </button>
                )}
              </div>
            </form>
          )}

          {!editingItemId &&
            activeTab === "voice" && (
              <div className="voice-panel">
                <h3>
                  Voice Entry
                </h3>

                <p>
                  Try:
                  <strong>
                    {" "}
                    “Please add 1 kg
                    of potatoes.”
                  </strong>
                </p>

                <p>
                  Or:
                  <strong>
                    {" "}
                    “Add 1 kg
                    potatoes bought
                    yesterday, expires
                    in 3 days.”
                  </strong>
                </p>

                <p>
                  Or:
                  <strong>
                    {" "}
                    “I have 1 kg
                    potatoes, bought
                    on 24 July 2026,
                    expires on 30
                    July 2026.”
                  </strong>
                </p>

                <button
                  className="primary-button"
                  type="button"
                  onClick={
                    startVoiceRecognition
                  }
                  disabled={isListening}
                >
                  {isListening
                    ? "Listening..."
                    : "Start Listening"}
                </button>

                {voiceText && (
                  <div className="voice-transcript">
                    <strong>
                      Heard:
                    </strong>

                    <p>
                      {voiceText}
                    </p>
                  </div>
                )}

                {voiceItems.length > 0 && (
                  <div className="voice-confirmation">
                    <h4>
                      Confirm detected
                      ingredients
                    </h4>

                    <p className="confirmation-note">
                      Explicit dates from
                      your speech are used
                      directly. Missing dates
                      are filled automatically.
                    </p>

                    <div className="voice-items">
                      {voiceItems.map(
                        (item, index) => (
                          <div
                            className="voice-item"
                            key={`${item.normalized_name}-${index}`}
                          >
                            <input
                              type="text"
                              value={
                                item.ingredient_name
                              }
                              onChange={(event) =>
                                updateVoiceItem(
                                  index,
                                  "ingredient_name",
                                  event.target.value
                                )
                              }
                            />

                            <input
                              type="number"
                              min="0"
                              step="any"
                              value={item.quantity}
                              onChange={(event) =>
                                updateVoiceItem(
                                  index,
                                  "quantity",
                                  event.target.value
                                )
                              }
                            />

                            <select
                              value={item.unit}
                              onChange={(event) =>
                                updateVoiceItem(
                                  index,
                                  "unit",
                                  event.target.value
                                )
                              }
                            >
                              <option value="kg">
                                kg
                              </option>
                              <option value="g">
                                g
                              </option>
                              <option value="litre">
                                Litre
                              </option>
                              <option value="ml">
                                ml
                              </option>
                              <option value="piece">
                                Piece
                              </option>
                              <option value="dozen">
                                Dozen
                              </option>
                            </select>

                            <select
                              value={
                                item.category
                              }
                              onChange={(event) =>
                                updateVoiceItem(
                                  index,
                                  "category",
                                  event.target.value
                                )
                              }
                            >
                              <option value="vegetables">
                                Vegetables
                              </option>
                              <option value="fruits">
                                Fruits
                              </option>
                              <option value="grains">
                                Grains
                              </option>
                              <option value="dairy">
                                Dairy
                              </option>
                              <option value="protein">
                                Protein
                              </option>
                              <option value="pulses">
                                Pulses
                              </option>
                              <option value="spices">
                                Spices
                              </option>
                              <option value="other">
                                Other
                              </option>
                            </select>

                            <div className="voice-date-field">
                              <label>
                                Purchase
                              </label>

                              <input
                                type="date"
                                value={
                                  item.purchase_date
                                }
                                onChange={(event) =>
                                  updateVoiceItem(
                                    index,
                                    "purchase_date",
                                    event.target.value
                                  )
                                }
                              />
                            </div>

                            <div className="voice-date-field">
                              <label>
                                Expiry
                              </label>

                              <input
                                type="date"
                                value={
                                  item.expiry_date
                                }
                                onChange={(event) =>
                                  updateVoiceExpiry(
                                    index,
                                    event.target.value
                                  )
                                }
                              />

                              {item.expiry_is_estimated &&
                                item.expiry_date && (
                                  <small className="field-note">
                                    Estimated
                                  </small>
                                )}
                            </div>

                            <button
                              className="small-button delete-button"
                              type="button"
                              onClick={() =>
                                removeVoiceItem(
                                  index
                                )
                              }
                            >
                              Remove
                            </button>
                          </div>
                        )
                      )}
                    </div>

                    <button
                      className="primary-button"
                      type="button"
                      onClick={
                        confirmVoiceItems
                      }
                    >
                      Confirm & Add
                      to Pantry
                    </button>
                  </div>
                )}
              </div>
            )}

          {!editingItemId &&
            activeTab === "ocr" && (
              <div className="ocr-panel">
                <h3>
                  Scan Grocery Bill
                </h3>

                <p>
                  Upload a grocery bill. Mealora
                  will read the bill, identify
                  probable grocery rows, and let
                  you verify them before saving.
                </p>

                <input
                  type="file"
                  accept="image/*"
                  onChange={
                    handleOCRUpload
                  }
                  disabled={
                    isOCRProcessing
                  }
                />

                {isOCRProcessing && (
                  <div className="ocr-progress">
                    <p>
                      Reading bill...
                      {" "}
                      {ocrProgress}%
                    </p>

                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{
                          width:
                            `${ocrProgress}%`,
                        }}
                      />
                    </div>
                  </div>
                )}

                {ocrText && (
                  <details className="ocr-text-details">
                    <summary>
                      View extracted text
                    </summary>

                    <pre>
                      {ocrText}
                    </pre>
                  </details>
                )}

                {ocrItems.length > 0 && (
                  <div className="ocr-confirmation">
                    <h4>
                      Verify detected items
                    </h4>

                    <p className="confirmation-note">
                      OCR is not assumed to be
                      perfect. Correct anything
                      before adding it to your pantry.
                    </p>

                    <div className="ocr-items">
                      {ocrItems.map(
                        (item, index) => (
                          <div
                            className="ocr-item"
                            key={`${item.normalized_name}-${index}`}
                          >
                            <div className="ocr-source-line">
                              <small>
                                OCR:
                              </small>

                              <span>
                                {item.source_line}
                              </span>
                            </div>

                            <input
                              type="text"
                              value={
                                item.ingredient_name
                              }
                              onChange={(event) =>
                                updateOCRItem(
                                  index,
                                  "ingredient_name",
                                  event.target.value
                                )
                              }
                            />

                            <input
                              type="number"
                              min="0"
                              step="any"
                              value={
                                item.quantity
                              }
                              onChange={(event) =>
                                updateOCRItem(
                                  index,
                                  "quantity",
                                  event.target.value
                                )
                              }
                            />

                            <select
                              value={
                                item.unit
                              }
                              onChange={(event) =>
                                updateOCRItem(
                                  index,
                                  "unit",
                                  event.target.value
                                )
                              }
                            >
                              <option value="kg">
                                kg
                              </option>
                              <option value="g">
                                g
                              </option>
                              <option value="litre">
                                Litre
                              </option>
                              <option value="ml">
                                ml
                              </option>
                              <option value="piece">
                                Piece
                              </option>
                              <option value="dozen">
                                Dozen
                              </option>
                              <option value="lb">
                                lb
                              </option>
                              <option value="gallon">
                                Gallon
                              </option>
                              <option value="box">
                                Box
                              </option>
                              <option value="bag">
                                Bag
                              </option>
                              <option value="pack">
                                Pack
                              </option>
                              <option value="loaf">
                                Loaf
                              </option>
                              <option value="cup">
                                Cup
                              </option>
                            </select>

                            <select
                              value={
                                item.category
                              }
                              onChange={(event) =>
                                updateOCRItem(
                                  index,
                                  "category",
                                  event.target.value
                                )
                              }
                            >
                              <option value="vegetables">
                                Vegetables
                              </option>
                              <option value="fruits">
                                Fruits
                              </option>
                              <option value="grains">
                                Grains
                              </option>
                              <option value="dairy">
                                Dairy
                              </option>
                              <option value="protein">
                                Protein
                              </option>
                              <option value="pulses">
                                Pulses
                              </option>
                              <option value="spices">
                                Spices
                              </option>
                              <option value="other">
                                Other
                              </option>
                            </select>

                            <div className="voice-date-field">
                              <label>
                                Purchase
                              </label>

                              <input
                                type="date"
                                value={
                                  item.purchase_date
                                }
                                onChange={(event) =>
                                  updateOCRItem(
                                    index,
                                    "purchase_date",
                                    event.target.value
                                  )
                                }
                              />
                            </div>

                            <div className="voice-date-field">
                              <label>
                                Expiry
                              </label>

                              <input
                                type="date"
                                value={
                                  item.expiry_date
                                }
                                onChange={(event) =>
                                  updateOCRExpiry(
                                    index,
                                    event.target.value
                                  )
                                }
                              />

                              {item.expiry_is_estimated &&
                                item.expiry_date && (
                                  <small className="field-note">
                                    Estimated
                                  </small>
                                )}
                            </div>

                            <button
                              className="small-button delete-button"
                              type="button"
                              onClick={() =>
                                removeOCRItem(
                                  index
                                )
                              }
                            >
                              Remove
                            </button>
                          </div>
                        )
                      )}
                    </div>

                    <button
                      className="primary-button"
                      type="button"
                      onClick={
                        confirmOCRItems
                      }
                    >
                      Confirm & Add
                      to Pantry
                    </button>
                  </div>
                )}
              </div>
            )}

          {error && (
            <p className="message error-message">
              {error}
            </p>
          )}

          {successMessage && (
            <p className="message success-message">
              {successMessage}
            </p>
          )}
        </section>

        <section className="section-card">
          <div className="section-heading">
            <div>
              <h2>
                Current Pantry
              </h2>

              <p>
                Your ingredients are loaded
                from Supabase.
              </p>
            </div>

            <input
              className="search-input"
              type="search"
              placeholder="Search ingredients..."
              value={
                searchTerm
              }
              onChange={(event) =>
                setSearchTerm(
                  event.target.value
                )
              }
            />
          </div>

          {loading && (
            <div className="empty-state">
              <h3>
                Loading pantry...
              </h3>

              <p>
                Please wait.
              </p>
            </div>
          )}

          {!loading &&
            filteredPantryItems.length === 0 && (
              <div className="empty-state">
                <h3>
                  {searchTerm
                    ? "No matching ingredients"
                    : "Your pantry is empty"}
                </h3>

                <p>
                  {searchTerm
                    ? "Try a different search term."
                    : "Add your first ingredient to get started."}
                </p>
              </div>
            )}

          {!loading &&
            filteredPantryItems.length > 0 && (
              <div className="pantry-list">
                {filteredPantryItems.map(
                  (item) => {
                    const expiryStatus =
                      getExpiryStatus(
                        item.expiry_date
                      );

                    const lowStock =
                      isLowStock(item);

                    return (
                      <div
                        className="pantry-item"
                        key={item.id}
                      >
                        <div className="pantry-item-main">
                          <h3>
                            {
                              item.ingredient_name
                            }
                          </h3>

                          <p>
                            {
                              item.quantity
                            }{" "}
                            {item.unit} ·{" "}
                            {
                              item.category
                            }
                          </p>

                          <div className="date-summary">
                            <span>
                              Purchased:{" "}
                              {item.purchase_date ||
                                "Not recorded"}
                            </span>

                            <span>
                              Expires:{" "}
                              {item.expiry_date ||
                                "Not recorded"}
                            </span>
                          </div>

                          {lowStock && (
                            <span className="status-badge low-stock-badge">
                              Low stock
                            </span>
                          )}
                        </div>

                        <div className="pantry-item-right">
                          <span
                            className={`expiry-badge ${expiryStatus.type}`}
                          >
                            {
                              expiryStatus.label
                            }
                          </span>

                          <div className="item-actions">
                            <button
                              className="small-button"
                              type="button"
                              onClick={() =>
                                startEditing(
                                  item
                                )
                              }
                            >
                              Edit
                            </button>

                            <button
                              className="small-button delete-button"
                              type="button"
                              onClick={() =>
                                handleDeleteIngredient(
                                  item.id,
                                  item.ingredient_name
                                )
                              }
                            >
                              Delete
                            </button>
                          </div>
                        </div>
                      </div>
                    );
                  }
                )}
              </div>
            )}
        </section>

        <section className="section-card">
          <div className="section-heading">
            <div>
              <h2>
                Needs Attention
              </h2>

              <p>
                Ingredients that may need
                to be used soon or restocked.
              </p>
            </div>
          </div>

          <div className="attention-grid">
            <div className="attention-card">
              <h3>
                Expiring Soon
              </h3>

              {expiringItems.length ===
              0 ? (
                <p>
                  No items expiring
                  within 7 days.
                </p>
              ) : (
                <div className="attention-list">
                  {expiringItems
                    .slice(0, 5)
                    .map((item) => {
                      const status =
                        getExpiryStatus(
                          item.expiry_date
                        );

                      return (
                        <div
                          className="attention-item"
                          key={item.id}
                        >
                          <span>
                            {
                              item.ingredient_name
                            }
                          </span>

                          <span
                            className={`text-status ${status.type}`}
                          >
                            {
                              status.label
                            }
                          </span>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>

            <div className="attention-card">
              <h3>
                Low Stock
              </h3>

              {lowStockItems.length ===
              0 ? (
                <p>
                  No low-stock items.
                </p>
              ) : (
                <div className="attention-list">
                  {lowStockItems
                    .slice(0, 5)
                    .map((item) => (
                      <div
                        className="attention-item"
                        key={item.id}
                      >
                        <span>
                          {
                            item.ingredient_name
                          }
                        </span>

                        <span className="text-status low-stock-text">
                          {
                            item.quantity
                          }{" "}
                          {
                            item.unit
                          }
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="section-card">
          <div className="section-heading">
            <div>
              <h2>Pantry Analytics</h2>

              <p>
                A quick visual summary of your
                pantry health and composition.
              </p>
            </div>
          </div>

          {/* KPI CARDS */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(auto-fit, minmax(190px, 1fr))",
              gap: "16px",
              marginBottom: "24px",
            }}
          >
            <div
              style={{
                padding: "20px",
                borderRadius: "16px",
                background:
                  "linear-gradient(135deg, #eef6ff, #ffffff)",
                border:
                  "1px solid #cfe2ff",
                boxShadow:
                  "0 5px 16px rgba(30, 80, 140, 0.08)",
              }}
            >
              <div
                style={{
                  fontSize: "13px",
                  color: "#64748b",
                  marginBottom: "8px",
                }}
              >
                Total Ingredients
              </div>

              <div
                style={{
                  fontSize: "32px",
                  fontWeight: "800",
                  color: "#1e3a5f",
                }}
              >
                {analyticsSummary.totalItems}
              </div>

              <div
                style={{
                  marginTop: "6px",
                  fontSize: "12px",
                  color: "#64748b",
                }}
              >
                Items currently in pantry
              </div>
            </div>

            <div
              style={{
                padding: "20px",
                borderRadius: "16px",
                background:
                  "linear-gradient(135deg, #f4f0ff, #ffffff)",
                border:
                  "1px solid #ddd2ff",
                boxShadow:
                  "0 5px 16px rgba(80, 50, 150, 0.08)",
              }}
            >
              <div
                style={{
                  fontSize: "13px",
                  color: "#64748b",
                  marginBottom: "8px",
                }}
              >
                Categories
              </div>

              <div
                style={{
                  fontSize: "32px",
                  fontWeight: "800",
                  color: "#55378f",
                }}
              >
                {
                  analyticsSummary.categoryEntries
                    .length
                }
              </div>

              <div
                style={{
                  marginTop: "6px",
                  fontSize: "12px",
                  color: "#64748b",
                }}
              >
                Different food groups
              </div>
            </div>

            <div
              style={{
                padding: "20px",
                borderRadius: "16px",
                background:
                  "linear-gradient(135deg, #fff8e8, #ffffff)",
                border:
                  "1px solid #f3dfae",
                boxShadow:
                  "0 5px 16px rgba(160, 110, 30, 0.08)",
              }}
            >
              <div
                style={{
                  fontSize: "13px",
                  color: "#64748b",
                  marginBottom: "8px",
                }}
              >
                Expiring Soon
              </div>

              <div
                style={{
                  fontSize: "32px",
                  fontWeight: "800",
                  color: "#a16207",
                }}
              >
                {
                  analyticsSummary.expiringSoonCount
                }
              </div>

              <div
                style={{
                  marginTop: "6px",
                  fontSize: "12px",
                  color: "#64748b",
                }}
              >
                Need attention within 7 days
              </div>
            </div>

            <div
              style={{
                padding: "20px",
                borderRadius: "16px",
                background:
                  "linear-gradient(135deg, #fff0f0, #ffffff)",
                border:
                  "1px solid #f5cccc",
                boxShadow:
                  "0 5px 16px rgba(160, 40, 40, 0.08)",
              }}
            >
              <div
                style={{
                  fontSize: "13px",
                  color: "#64748b",
                  marginBottom: "8px",
                }}
              >
                Low Stock
              </div>

              <div
                style={{
                  fontSize: "32px",
                  fontWeight: "800",
                  color: "#b42318",
                }}
              >
                {
                  analyticsSummary.lowStockCount
                }
              </div>

              <div
                style={{
                  marginTop: "6px",
                  fontSize: "12px",
                  color: "#64748b",
                }}
              >
                Items that may need restocking
              </div>
            </div>
          </div>

          {/* CATEGORY + COMPLETENESS */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "minmax(0, 1.4fr) minmax(280px, 1fr)",
              gap: "20px",
            }}
          >
            <div
              style={{
                padding: "20px",
                borderRadius: "16px",
                border:
                  "1px solid #e2e8f0",
                background: "#ffffff",
                boxShadow:
                  "0 4px 14px rgba(15, 23, 42, 0.05)",
              }}
            >
              <h3
                style={{
                  marginTop: 0,
                  marginBottom: "18px",
                }}
              >
                Category Distribution
              </h3>

              {analyticsSummary.categoryEntries
                .length === 0 ? (
                <p>No pantry data yet.</p>
              ) : (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "14px",
                  }}
                >
                  {analyticsSummary.categoryEntries.map(
                    (entry) => {
                      const percentage =
                        analyticsSummary.totalItems >
                        0
                          ? Math.round(
                              (entry.count /
                                analyticsSummary.totalItems) *
                                100
                            )
                          : 0;

                      return (
                        <div
                          key={entry.name}
                        >
                          <div
                            style={{
                              display: "flex",
                              justifyContent:
                                "space-between",
                              alignItems:
                                "center",
                              marginBottom: "6px",
                              fontSize: "14px",
                            }}
                          >
                            <strong>
                              {entry.name}
                            </strong>

                            <span
                              style={{
                                color:
                                  "#64748b",
                              }}
                            >
                              {entry.count}{" "}
                              · {percentage}%
                            </span>
                          </div>

                          <div
                            style={{
                              width: "100%",
                              height: "10px",
                              borderRadius: "999px",
                              background:
                                "#edf2f7",
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                width:
                                  `${percentage}%`,
                                height: "100%",
                                borderRadius:
                                  "999px",
                                background:
                                  "linear-gradient(90deg, #4f46e5, #7c3aed)",
                                transition:
                                  "width 0.3s ease",
                              }}
                            />
                          </div>
                        </div>
                      );
                    }
                  )}
                </div>
              )}
            </div>

            <div
              style={{
                padding: "20px",
                borderRadius: "16px",
                border:
                  "1px solid #e2e8f0",
                background: "#ffffff",
                boxShadow:
                  "0 4px 14px rgba(15, 23, 42, 0.05)",
              }}
            >
              <h3
                style={{
                  marginTop: 0,
                  marginBottom: "18px",
                }}
              >
                Pantry Health
              </h3>

              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "16px",
                }}
              >
                <div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent:
                        "space-between",
                      marginBottom: "6px",
                      fontSize: "14px",
                    }}
                  >
                    <span>
                      Expiry information
                    </span>

                    <strong>
                      {
                        analyticsSummary.itemsWithExpiry
                      }
                      /
                      {
                        analyticsSummary.totalItems
                      }
                    </strong>
                  </div>

                  <div
                    style={{
                      height: "9px",
                      borderRadius:
                        "999px",
                      background:
                        "#edf2f7",
                    }}
                  >
                    <div
                      style={{
                        width:
                          analyticsSummary.totalItems >
                          0
                            ? `${Math.round(
                                (analyticsSummary.itemsWithExpiry /
                                  analyticsSummary.totalItems) *
                                  100
                              )}%`
                            : "0%",
                        height: "100%",
                        borderRadius:
                          "999px",
                        background:
                          "linear-gradient(90deg, #16a34a, #22c55e)",
                      }}
                    />
                  </div>
                </div>

                <div
                  style={{
                    padding: "14px",
                    borderRadius: "12px",
                    background:
                      "#fff8e1",
                    border:
                      "1px solid #f4df9f",
                  }}
                >
                  <div
                    style={{
                      fontSize: "13px",
                      color: "#7c5a00",
                    }}
                  >
                    Expiring soon
                  </div>

                  <div
                    style={{
                      fontSize: "24px",
                      fontWeight: "800",
                      color: "#a16207",
                      marginTop: "3px",
                    }}
                  >
                    {
                      analyticsSummary.expiringSoonCount
                    }
                  </div>
                </div>

                <div
                  style={{
                    padding: "14px",
                    borderRadius: "12px",
                    background:
                      "#fff1f2",
                    border:
                      "1px solid #fecdd3",
                  }}
                >
                  <div
                    style={{
                      fontSize: "13px",
                      color: "#9f1239",
                    }}
                  >
                    Low stock
                  </div>

                  <div
                    style={{
                      fontSize: "24px",
                      fontWeight: "800",
                      color: "#be123c",
                      marginTop: "3px",
                    }}
                  >
                    {
                      analyticsSummary.lowStockCount
                    }
                  </div>
                </div>

                <div
                  style={{
                    padding: "14px",
                    borderRadius: "12px",
                    background:
                      "#f0fdf4",
                    border:
                      "1px solid #bbf7d0",
                  }}
                >
                  <div
                    style={{
                      fontSize: "13px",
                      color: "#166534",
                    }}
                  >
                    Healthy pantry items
                  </div>

                  <div
                    style={{
                      fontSize: "24px",
                      fontWeight: "800",
                      color: "#15803d",
                      marginTop: "3px",
                    }}
                  >
                    {Math.max(
                      0,
                      analyticsSummary.totalItems -
                        analyticsSummary.expiringSoonCount -
                        analyticsSummary.lowStockCount
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

          </>
        )}

        {currentPage === "recommendations" && (
          <section className="section-card" style={{ marginTop: "24px" }}>
            <div style={{ marginBottom: "22px" }}>
              <h2 style={{ margin: 0 }}>🍽️ Smart Recommendations</h2>
              <p style={{ marginTop: "8px", marginBottom: 0 }}>
                Personalized recipes ranked by Mealora's AI recommendation engine.
              </p>
            </div>

            <div
              style={{
                padding: "20px",
                borderRadius: "16px",
                background: "#f8fafc",
                border: "1px solid #e2e8f0",
              }}
            >
              <h3 style={{ marginTop: 0, marginBottom: "16px" }}>
                🎯 Recommendation Preferences
              </h3>

              <div
                style={{
                  display: "flex",
                  justifyContent: "center",
                  gap: "12px",
                  flexWrap: "wrap",
                  marginBottom: "18px",
                }}
              >
                <button
                  type="button"
                  className={autoMode ? "primary-button" : "secondary-button"}
                  onClick={() => setAutoMode(true)}
                >
                  🤖 Smart Automatic
                </button>

                <button
                  type="button"
                  className={!autoMode ? "primary-button" : "secondary-button"}
                  onClick={() => setAutoMode(false)}
                >
                  🎯 Custom
                </button>
              </div>

              {autoMode ? (
                <div
                  style={{
                    padding: "15px",
                    borderRadius: "12px",
                    background: "#ecfdf5",
                    border: "1px solid #bbf7d0",
                    textAlign: "center",
                    fontSize: "14px",
                  }}
                >
                  <strong>Smart Automatic mode</strong>
                  <div style={{ marginTop: "5px", color: "#475569" }}>
                    Mealora automatically fetches the current date, time, location and weather,
                    detects the meal type and checks the festival calendar.
                    The live values are shown below before you request recipes.
                  </div>
                </div>
              ) : (
                <>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                      gap: "14px",
                      marginBottom: "18px",
                    }}
                  >
                    <label>
                      <span style={{ display: "block", fontSize: "13px", fontWeight: "700", marginBottom: "6px" }}>
                        Date
                      </span>
                      <input
                        type="date"
                        value={customDate}
                        onChange={(event) => setCustomDate(event.target.value)}
                        style={{ width: "100%", padding: "10px", borderRadius: "10px", border: "1px solid #cbd5e1" }}
                      />
                    </label>

                    <label>
                      <span style={{ display: "block", fontSize: "13px", fontWeight: "700", marginBottom: "6px" }}>
                        Time
                      </span>
                      <input
                        type="time"
                        value={customTime}
                        onChange={(event) => setCustomTime(event.target.value)}
                        style={{ width: "100%", padding: "10px", borderRadius: "10px", border: "1px solid #cbd5e1" }}
                      />
                    </label>

                    <label>
                      <span style={{ display: "block", fontSize: "13px", fontWeight: "700", marginBottom: "6px" }}>
                        Location
                      </span>
                      <input
                        type="text"
                        value={customLocation}
                        onChange={(event) => setCustomLocation(event.target.value)}
                        placeholder="e.g. Chennai"
                        style={{ width: "100%", padding: "10px", borderRadius: "10px", border: "1px solid #cbd5e1" }}
                      />
                    </label>
                  </div>

                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                      gap: "14px",
                    }}
                  >
                    <label>
                      <span style={{ display: "block", fontSize: "13px", fontWeight: "700", marginBottom: "6px" }}>
                        Meal Type
                      </span>
                      <select
                        value={mealType}
                        onChange={(event) => setMealType(event.target.value)}
                        style={{ width: "100%", padding: "10px", borderRadius: "10px", border: "1px solid #cbd5e1" }}
                      >
                        <option value="Breakfast">Breakfast</option>
                        <option value="Lunch">Lunch</option>
                        <option value="Dinner">Dinner</option>
                        <option value="Snack">Snack</option>
                      </select>
                    </label>

                    <label>
                      <span style={{ display: "block", fontSize: "13px", fontWeight: "700", marginBottom: "6px" }}>
                        Diet
                      </span>
                      <select
                        value={diet}
                        onChange={(event) => setDiet(event.target.value)}
                        style={{ width: "100%", padding: "10px", borderRadius: "10px", border: "1px solid #cbd5e1" }}
                      >
                        <option value="Vegetarian">Vegetarian</option>
                        <option value="Non-Vegetarian">Non-Vegetarian</option>
                        <option value="Vegan">Vegan</option>
                        <option value="Jain">Jain</option>
                      </select>
                    </label>

                    <label>
                      <span style={{ display: "block", fontSize: "13px", fontWeight: "700", marginBottom: "6px" }}>
                        Cuisine
                      </span>
                      <select
                        value={cuisine}
                        onChange={(event) => setCuisine(event.target.value)}
                        style={{ width: "100%", padding: "10px", borderRadius: "10px", border: "1px solid #cbd5e1" }}
                      >
                        <option value="South Indian">South Indian</option>
                        <option value="North Indian">North Indian</option>
                        <option value="East Indian">East Indian</option>
                        <option value="West Indian">West Indian</option>
                        <option value="Indian">Indian</option>
                      </select>
                    </label>

                    <label>
                      <span style={{ display: "block", fontSize: "13px", fontWeight: "700", marginBottom: "6px" }}>
                        Maximum Time
                      </span>
                      <select
                        value={maxTime}
                        onChange={(event) => setMaxTime(Number(event.target.value))}
                        style={{ width: "100%", padding: "10px", borderRadius: "10px", border: "1px solid #cbd5e1" }}
                      >
                        <option value={15}>15 minutes</option>
                        <option value={30}>30 minutes</option>
                        <option value={45}>45 minutes</option>
                        <option value={60}>60 minutes</option>
                        <option value={90}>90 minutes</option>
                      </select>
                    </label>

                    <label>
                      <span style={{ display: "block", fontSize: "13px", fontWeight: "700", marginBottom: "6px" }}>
                        Number of Recipes
                      </span>
                      <select
                        value={topK}
                        onChange={(event) => setTopK(Number(event.target.value))}
                        style={{ width: "100%", padding: "10px", borderRadius: "10px", border: "1px solid #cbd5e1" }}
                      >
                        <option value={5}>5 Recipes</option>
                        <option value={10}>10 Recipes</option>
                        <option value={15}>15 Recipes</option>
                      </select>
                    </label>
                  </div>
                </>
              )}

            {/* FETCHED CONTEXT TILES */}
            {contextInfo && (
              <div style={{ marginTop: "22px" }}>
                <h3 style={{ marginBottom: "14px" }}>
                  📡 Live Context Used by Mealora
                </h3>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
                    gap: "12px",
                  }}
                >
                  {[
                    [
                      "📅 Date & Time",
                      `${contextInfo.day || "—"} · ${contextInfo.date || "—"} · ${contextInfo.time || "—"}`,
                    ],
                    [
                      "🍽️ Meal Type",
                      contextInfo.meal_type || "—",
                    ],
                    [
                      "📍 Location",
                      contextInfo.location_name || contextInfo.location_city || "Location unavailable",
                    ],
                    [
                      "🌤️ Weather",
                      contextInfo.weather?.available
                        ? `${contextInfo.weather.condition} · ${contextInfo.weather.temperature}°C`
                        : "Weather unavailable",
                    ],
                    [
                      "🎉 Occasion",
                      contextInfo.festival?.name || "No major festival today",
                    ],
                    [
                      "🥗 Diet & Cuisine",
                      `${contextInfo.diet || "—"} · ${contextInfo.cuisine || "—"}`,
                    ],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      style={{
                        padding: "16px",
                        borderRadius: "14px",
                        background: "#fff",
                        border: "1px solid #e2e8f0",
                      }}
                    >
                      <div style={{ fontSize: "12px", color: "#64748b", fontWeight: "700" }}>
                        {label}
                      </div>
                      <div style={{ marginTop: "6px", fontSize: "16px", fontWeight: "700" }}>
                        {value}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}


              {/* GET RECOMMENDATIONS BUTTON - CENTERED BELOW MODES */}
              <div style={{ display: "flex", justifyContent: "center", marginTop: "20px" }}>
                <button
                  className="primary-button"
                  type="button"
                  onClick={loadRecommendations}
                  disabled={recommendationLoading}
                  style={{ minWidth: "240px" }}
                >
                  {recommendationLoading
                    ? "Finding recipes..."
                    : "✨ Get Recommendations"}
                </button>
              </div>
            </div>

            {recommendationError && (
              <div
                style={{
                  padding: "14px 16px",
                  borderRadius: "12px",
                  background: "#fff1f2",
                  border: "1px solid #fecdd3",
                  color: "#9f1239",
                  marginTop: "20px",
                }}
              >
                {recommendationError}
              </div>
            )}

            {recommendationLoading && (
              <div className="empty-state" style={{ marginTop: "20px" }}>
                <h3>Finding the best recipes...</h3>
                <p>Mealora is running the recommendation pipeline.</p>
              </div>
            )}

            {!recommendationLoading &&
              !recommendationError &&
              recommendationResults.length === 0 && (
                <div className="empty-state" style={{ marginTop: "20px" }}>
                  <h3>No recommendations loaded yet</h3>
                  <p>
                    Smart Automatic mode will load them automatically.
                    Custom mode lets you change the context.
                  </p>
                </div>
              )}

            {!recommendationLoading && recommendationResults.length > 0 && (
              <>
                <div
                  style={{
                    marginTop: "22px",
                    padding: "14px 16px",
                    borderRadius: "12px",
                    background: "#eff6ff",
                    border: "1px solid #dbeafe",
                    fontSize: "14px",
                  }}
                >
                  <strong>
                    {autoMode ? "Smart context:" : "Custom context:"}
                  </strong>{" "}
                  {contextInfo
                    ? `${contextInfo.meal_type} · ${contextInfo.diet} · ${contextInfo.cuisine} · ${contextInfo.max_time} min`
                    : `${mealType} · ${diet} · ${cuisine} · ${maxTime} min`}
                </div>

                <div
                  style={{
                    marginTop: "16px",
                    padding: "14px 16px",
                    borderRadius: "12px",
                    background: "#faf5ff",
                    border: "1px solid #e9d5ff",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    gap: "12px",
                    flexWrap: "wrap",
                  }}
                >
                  <div>
                    <strong>🌐 Recipe Language</strong>
                    <div style={{ fontSize: "12px", color: "#64748b", marginTop: "3px" }}>
                      Mealora selects the recipe; Gemini only presents the selected recipe in your language.
                    </div>
                  </div>
                  <select
                    value={assistantLanguage}
                    onChange={(event) => handleAssistantLanguageChange(event.target.value)}
                    style={{
                      padding: "9px 12px",
                      borderRadius: "10px",
                      border: "1px solid #cbd5e1",
                      background: "#fff",
                      minWidth: "150px",
                    }}
                  >
                    {RECIPE_LANGUAGES.map((language) => (
                      <option key={language} value={language}>
                        {language}
                      </option>
                    ))}
                  </select>
                </div>

                {festivalSpecial && (
                  <div
                    style={{
                      marginTop: "22px",
                      borderRadius: "20px",
                      padding: "2px",
                      background: "linear-gradient(135deg, #f59e0b, #ec4899, #7c3aed)",
                      boxShadow: "0 14px 38px rgba(124, 58, 237, 0.20)",
                    }}
                  >
                    {(() => {
                      const pantryStatus = getFestivalPantryStatus(festivalSpecial);
                      const missing = pantryStatus.missing || [];
                      return (
                        <div
                          style={{
                            borderRadius: "18px",
                            padding: "22px",
                            background: "linear-gradient(135deg, #fff7ed 0%, #fff 52%, #faf5ff 100%)",
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "flex-start",
                              gap: "16px",
                              flexWrap: "wrap",
                            }}
                          >
                            <div>
                              <div
                                style={{
                                  display: "inline-flex",
                                  alignItems: "center",
                                  gap: "7px",
                                  padding: "6px 10px",
                                  borderRadius: "999px",
                                  background: "#fef3c7",
                                  color: "#92400e",
                                  fontSize: "12px",
                                  fontWeight: "800",
                                  letterSpacing: "0.02em",
                                }}
                              >
                                🎉 FESTIVAL SPECIAL
                              </div>
                              <h2 style={{ margin: "12px 0 5px", lineHeight: 1.25 }}>
                                {festivalSpecial.recipe_name}
                              </h2>
                              <div style={{ color: "#7c3aed", fontWeight: "700", fontSize: "14px" }}>
                                🪔 {festivalSpecial.festival}
                              </div>
                              <p style={{ margin: "8px 0 0", color: "#475569", fontSize: "13px" }}>
                                A traditional festival recipe highlighted by Mealora for today's occasion.
                              </p>
                            </div>

                            <div
                              style={{
                                padding: "10px 13px",
                                borderRadius: "12px",
                                background: "#fff",
                                border: "1px solid #fed7aa",
                                textAlign: "center",
                                minWidth: "120px",
                              }}
                            >
                              <div style={{ fontSize: "11px", color: "#64748b" }}>Pantry availability</div>
                              <strong style={{ display: "block", marginTop: "3px", fontSize: "18px" }}>
                                {pantryStatus.available}/{pantryStatus.total}
                              </strong>
                              <div style={{ fontSize: "11px", color: "#64748b" }}>ingredient types</div>
                            </div>
                          </div>

                          {missing.length > 0 && (
                            <div
                              style={{
                                marginTop: "16px",
                                padding: "13px 14px",
                                borderRadius: "12px",
                                background: "#fff",
                                border: "1px solid #fed7aa",
                              }}
                            >
                              <strong>🛒 Missing ingredients</strong>
                              <div style={{ marginTop: "6px", fontSize: "13px", color: "#475569", lineHeight: 1.55 }}>
                                {missing.slice(0, 8).join(" • ")}
                                {missing.length > 8 ? ` • +${missing.length - 8} more` : ""}
                              </div>
                              <div style={{ marginTop: "11px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
                                {[
                                  ["Blinkit", "https://blinkit.com/"],
                                  ["Zepto", "https://www.zeptonow.com/"],
                                  ["Instamart", "https://www.swiggy.com/instamart"],
                                  ["BigBasket", "https://www.bigbasket.com/"],
                                ].map(([label, url]) => (
                                  <button
                                    key={label}
                                    type="button"
                                    onClick={() => openShoppingSite(url)}
                                    style={{
                                      border: "1px solid #e2e8f0",
                                      background: "#fff",
                                      borderRadius: "10px",
                                      padding: "8px 12px",
                                      fontWeight: "700",
                                      cursor: "pointer",
                                    }}
                                  >
                                    🛍️ {label}
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}

                          <div style={{ marginTop: "16px", display: "flex", gap: "10px", flexWrap: "wrap" }}>
                            <button
                              type="button"
                              className="primary-button"
                              onClick={() => openRecipeAssistant(festivalSpecial)}
                            >
                              🌐 Open Festival Recipe in {assistantLanguage}
                            </button>
                            <div
                              style={{
                                alignSelf: "center",
                                fontSize: "12px",
                                color: "#64748b",
                              }}
                            >
                              Pantry availability does not hide this festival special.
                            </div>
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                )}

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                    gap: "20px",
                    marginTop: "20px",
                  }}
                >
                  {recommendationResults.map((recipe, index) => (
                    <div
                      key={recipe.recipe_id}
                      style={{
                        border: "1px solid #e5e7eb",
                        borderRadius: "16px",
                        padding: "18px",
                        background: "#fff",
                        boxShadow:
                          index === 0
                            ? "0 8px 24px rgba(15, 23, 42, 0.10)"
                            : "0 4px 14px rgba(15, 23, 42, 0.05)",
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", alignItems: "flex-start" }}>
                        <div>
                          <div style={{ fontSize: "12px", fontWeight: "700", color: "#64748b" }}>
                            #{index + 1} RECOMMENDATION
                          </div>
                          <h3 style={{ margin: "7px 0 0", lineHeight: 1.35 }}>
                            {recipe.recipe_name}
                          </h3>
                        </div>

                        <div style={{ minWidth: "72px", textAlign: "center", padding: "8px", borderRadius: "10px", background: "#f0fdf4" }}>
                          <div style={{ fontSize: "11px", color: "#64748b" }}>Score</div>
                          <strong>{Number(recipe.final_score || 0).toFixed(3)}</strong>
                        </div>
                      </div>

                      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "8px", marginTop: "16px" }}>
                        {[
                          ["Pantry match", `${(Number(recipe.pantry_score || 0) * 100).toFixed(0)}%`],
                          ["Expiry score", `${(Number(recipe.expiry_score || 0) * 100).toFixed(0)}%`],
                          ["Content match", `${(Number(recipe.content_score || 0) * 100).toFixed(0)}%`],
                          ["Collaborative", `${(Number(recipe.collaborative_score || 0) * 100).toFixed(0)}%`],
                        ].map(([label, value]) => (
                          <div key={label} style={{ padding: "10px", borderRadius: "10px", background: "#f8fafc" }}>
                            <small style={{ color: "#64748b" }}>{label}</small>
                            <strong style={{ display: "block", marginTop: "3px" }}>{value}</strong>
                          </div>
                        ))}
                      </div>

                      <div
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
                      </div>

                      <button
                        type="button"
                        className="primary-button"
                        onClick={() => openRecipeAssistant(recipe)}
                        style={{ width: "100%", marginTop: "14px" }}
                      >
                        🌐 Open Recipe in {assistantLanguage}
                      </button>
                    </div>
                  ))}
                </div>
              </>
            )}
          </section>
        )}


        {assistantLoading && (
          <div
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(15, 23, 42, 0.35)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 1000,
              padding: "20px",
            }}
          >
            <div
              style={{
                background: "#fff",
                borderRadius: "18px",
                padding: "28px",
                maxWidth: "520px",
                width: "100%",
                textAlign: "center",
                boxShadow: "0 20px 50px rgba(15,23,42,.2)",
              }}
            >
              <div style={{ fontSize: "34px" }}>🌐</div>
              <h3 style={{ margin: "12px 0 6px" }}>
                Preparing your recipe
              </h3>
              <p style={{ color: "#64748b", margin: 0 }}>
                Gemini is translating the recipe selected by Mealora.
              </p>
            </div>
          </div>
        )}

        {assistantError && (
          <div
            style={{
              position: "fixed",
              right: "20px",
              bottom: "20px",
              maxWidth: "420px",
              padding: "14px 16px",
              borderRadius: "12px",
              background: "#fff1f2",
              border: "1px solid #fecdd3",
              color: "#9f1239",
              zIndex: 1100,
              boxShadow: "0 8px 24px rgba(15,23,42,.12)",
            }}
          >
            <strong>Recipe assistant:</strong> {assistantError}
            <button
              type="button"
              onClick={() => setAssistantError("")}
              style={{
                marginLeft: "10px",
                border: 0,
                background: "transparent",
                cursor: "pointer",
              }}
            >
              ✕
            </button>
          </div>
        )}

        {assistantRecipe && (
          <div
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(15, 23, 42, 0.45)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 1000,
              padding: "20px",
            }}
            onClick={() => {
              window.speechSynthesis?.cancel();
              setAssistantRecipe(null);
            }}
          >
            <div
              style={{
                background: "#fff",
                borderRadius: "20px",
                maxWidth: "820px",
                width: "100%",
                maxHeight: "88vh",
                overflowY: "auto",
                padding: "26px",
                boxShadow: "0 25px 70px rgba(15,23,42,.25)",
              }}
              onClick={(event) => event.stopPropagation()}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  gap: "15px",
                }}
              >
                <div>
                  <div style={{ fontSize: "12px", fontWeight: "700", color: "#7c3aed" }}>
                    GEMINI RECIPE PRESENTATION
                  </div>
                  <h2 style={{ margin: "7px 0 4px" }}>
                    {assistantRecipe.recipe_name}
                  </h2>
                  <div style={{ fontSize: "13px", color: "#64748b" }}>
                    Presented in {assistantRecipe.language}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => {
                    window.speechSynthesis?.cancel();
                    setAssistantRecipe(null);
                    setAssistantSourceRecipe(null);
                  }}
                  style={{
                    border: "1px solid #cbd5e1",
                    background: "#fff",
                    borderRadius: "10px",
                    padding: "8px 11px",
                    cursor: "pointer",
                  }}
                >
                  ✕
                </button>
              </div>

              <div
                style={{
                  marginTop: "16px",
                  padding: "12px 14px",
                  borderRadius: "12px",
                  background: "#faf5ff",
                  border: "1px solid #e9d5ff",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: "12px",
                  flexWrap: "wrap",
                }}
              >
                <div>
                  <strong>🌐 Recipe Language</strong>
                  <div style={{ fontSize: "12px", color: "#64748b", marginTop: "3px" }}>
                    Changing this will translate the recipe text and instructions.
                  </div>
                </div>
                <select
                  value={assistantLanguage}
                  onChange={(event) => handleAssistantLanguageChange(event.target.value)}
                  disabled={assistantLoading}
                  style={{
                    padding: "9px 12px",
                    borderRadius: "10px",
                    border: "1px solid #cbd5e1",
                    background: "#fff",
                    minWidth: "150px",
                  }}
                >
                  {RECIPE_LANGUAGES.map((language) => (
                    <option key={language} value={language}>
                      {language}
                    </option>
                  ))}
                </select>
              </div>

              <div style={{ marginTop: "18px", display: "flex", gap: "10px", flexWrap: "wrap" }}>
                <button
                  type="button"
                  className="primary-button"
                  onClick={speakRecipe}
                >
                  🔊 Read Aloud
                </button>
                <button
                  type="button"
                  className="primary-button"
                  onClick={cookRecipeAndDeductFromPantry}
                  disabled={cookingRecipe}
                >
                  {cookingRecipe
                    ? "🍳 Updating Pantry..."
                    : "🍳 Cook & Update Pantry"}
                </button>
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => openRecipeAssistant(
                    assistantSourceRecipe || {
                      recipe_id: assistantRecipe.recipe_id,
                      recipe_name: assistantRecipe.recipe_name,
                    }
                  )}
                >
                  🔄 Refresh Translation
                </button>
              </div>

              <section style={{ marginTop: "22px" }}>
                <h3>🧺 Ingredients</h3>
                <ul style={{ lineHeight: 1.7, paddingLeft: "22px" }}>
                  {(assistantRecipe.ingredients || []).map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              </section>

              <section style={{ marginTop: "22px" }}>
                <h3>👩‍🍳 Instructions</h3>
                <ol style={{ lineHeight: 1.8, paddingLeft: "22px" }}>
                  {(assistantRecipe.instructions || []).map((item, index) => (
                    <li key={index} style={{ marginBottom: "8px" }}>{item}</li>
                  ))}
                </ol>
              </section>

              {assistantRecipe.note && (
                <div
                  style={{
                    marginTop: "20px",
                    padding: "12px 14px",
                    borderRadius: "10px",
                    background: "#f8fafc",
                    color: "#475569",
                    fontSize: "13px",
                  }}
                >
                  {assistantRecipe.note}
                </div>
              )}

              <div
                style={{
                  marginTop: "18px",
                  fontSize: "11px",
                  color: "#94a3b8",
                  borderTop: "1px solid #e2e8f0",
                  paddingTop: "12px",
                }}
              >
                Recipe selection and ranking are performed by Mealora's ML recommendation engine.
                Gemini is used only for multilingual recipe presentation.
              </div>
            </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default App;