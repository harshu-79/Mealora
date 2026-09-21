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

function App() {
  const [activeTab, setActiveTab] = useState("manual");

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

      <main className="pantry-container">
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
      </main>
    </div>
  );
}

export default App;