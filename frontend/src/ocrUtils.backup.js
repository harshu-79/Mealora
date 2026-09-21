import { PSM } from "tesseract.js";

export async function preprocessImage(file) {
  const image = await loadImage(file);

  const scale = Math.max(
    2,
    Math.min(4, 1600 / Math.max(image.width, image.height))
  );

  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", {
    willReadFrequently: true,
  });

  canvas.width = Math.round(image.width * scale);
  canvas.height = Math.round(image.height * scale);

  context.drawImage(
    image,
    0,
    0,
    canvas.width,
    canvas.height
  );

  const imageData = context.getImageData(
    0,
    0,
    canvas.width,
    canvas.height
  );

  const pixels = imageData.data;

  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];

    const gray =
      0.299 * r +
      0.587 * g +
      0.114 * b;

    pixels[i] = gray;
    pixels[i + 1] = gray;
    pixels[i + 2] = gray;
  }

  context.putImageData(imageData, 0, 0);

  const processedBlob = await new Promise(
    (resolve, reject) => {
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(
              new Error(
                "Unable to preprocess image."
              )
            );
            return;
          }

          resolve(blob);
        },
        "image/png",
        1
      );
    }
  );

  return processedBlob;
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();

    image.onload = () => resolve(image);
    image.onerror = () =>
      reject(
        new Error(
          "Unable to load the bill image."
        )
      );

    image.src = URL.createObjectURL(file);
  });
}

export function getOCRParameters() {
  return {
    tessedit_pageseg_mode:
      PSM.SINGLE_BLOCK,

    preserve_interword_spaces: "1",

    user_defined_dpi: "300",
  };
}

export function extractUsefulOCRText(text) {
  return text
    .split(/\r?\n/)
    .map((line) =>
      line
        .replace(/[|]/g, " ")
        .replace(/\s+/g, " ")
        .trim()
    )
    .filter(Boolean)
    .join("\n");
}

export function parseGroceryRows(text) {
  const lines = extractUsefulOCRText(text)
    .split("\n")
    .filter(Boolean);

  const results = [];
  const seen = new Set();

  for (const originalLine of lines) {
    let line = originalLine;

    if (isHeaderOrNonItemLine(line)) {
      continue;
    }

    line = removeCurrencyAndPriceData(line);

    const parsed = parseQuantityFromLine(line);

    if (!parsed) {
      continue;
    }

    let {
      ingredient,
      quantity,
      unit,
    } = parsed;

    ingredient = cleanIngredientName(
      ingredient
    );

    if (!ingredient) {
      continue;
    }

    const normalizedName =
      normalizeSingular(
        ingredient
      );

    if (
      !normalizedName ||
      normalizedName.length < 2
    ) {
      continue;
    }

    const key =
      `${normalizedName}|${unit}`;

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);

    results.push({
      ingredientName:
        capitalizeWords(ingredient),

      normalizedName,

      quantity,

      unit,

      sourceLine: originalLine,
    });
  }

  return results;
}

function parseQuantityFromLine(line) {
  const cleaned = line
    .replace(/\$/g, " ")
    .replace(/₹/g, " ")
    .replace(/,/g, "")
    .replace(/\s+/g, " ")
    .trim();

  /*
    Examples supported:

    Apples 3
    Milk 1 gallon
    Bread 2 loaves
    Eggs 1 dozen
    Tomatoes 5 lbs
    Rice 2 kg
    Spinach 500 g
    2 kg Rice
  */

  const tokens = cleaned.split(" ");

  const unitMap = {
    kg: "kg",
    kgs: "kg",
    kilo: "kg",
    kilos: "kg",
    kilogram: "kg",
    kilograms: "kg",

    g: "g",
    gram: "g",
    grams: "g",

    lb: "lb",
    lbs: "lb",
    pound: "lb",
    pounds: "lb",

    l: "litre",
    liter: "litre",
    liters: "litre",
    litre: "litre",
    litres: "litre",
    gallon: "gallon",
    gallons: "gallon",

    ml: "ml",
    milliliter: "ml",
    milliliters: "ml",
    millilitre: "ml",
    millilitres: "ml",

    piece: "piece",
    pieces: "piece",
    pc: "piece",
    pcs: "piece",

    dozen: "dozen",

    loaf: "loaf",
    loaves: "loaf",

    box: "box",
    boxes: "box",

    bag: "bag",
    bags: "bag",

    pack: "pack",
    packs: "pack",

    bottle: "bottle",
    bottles: "bottle",

    can: "can",
    cans: "can",

    jar: "jar",
    jars: "jar",

    cup: "cup",
    cups: "cup",
  };

  /*
    Case 1:
    2 kg rice
    500 g spinach
  */
  if (
    tokens.length >= 2 &&
    isNumber(tokens[0])
  ) {
    let quantity = Number(tokens[0]);
    let unit = "piece";
    let ingredientStart = 1;

    if (
      tokens[1] &&
      unitMap[tokens[1].toLowerCase()]
    ) {
      unit =
        unitMap[
          tokens[1].toLowerCase()
        ];

      ingredientStart = 2;
    }

    const ingredient =
      tokens
        .slice(ingredientStart)
        .join(" ")
        .trim();

    if (ingredient) {
      return {
        ingredient,
        quantity,
        unit,
      };
    }
  }

  /*
    Case 2:
    Apples 3
    Tomatoes 5 lbs
    Milk 1 gallon
  */

  for (let i = 1; i < tokens.length; i += 1) {
    const token = tokens[i];

    if (!isNumber(token)) {
      continue;
    }

    const quantity = Number(token);

    let unit = "piece";
    let ingredientEnd = i;

    const nextToken =
      tokens[i + 1]?.toLowerCase();

    if (
      nextToken &&
      unitMap[nextToken]
    ) {
      unit = unitMap[nextToken];
      ingredientEnd = i;

      return {
        ingredient: tokens
          .slice(0, ingredientEnd)
          .join(" ")
          .trim(),

        quantity,

        unit,
      };
    }

    return {
      ingredient: tokens
        .slice(0, ingredientEnd)
        .join(" ")
        .trim(),

      quantity,

      unit,
    };
  }

  return null;
}

function isNumber(value) {
  if (!value) {
    return false;
  }

  return /^\d+(?:\.\d+)?$/.test(
    value
  );
}

function removeCurrencyAndPriceData(line) {
  return line
    .replace(
      /(?:₹|\$)\s*\d+(?:\.\d+)?/g,
      " "
    )
    .replace(
      /\b\d+(?:\.\d+)?\s*\/\s*(?:lb|kg|g)\b/gi,
      " "
    )
    .replace(
      /\b(?:price|total|amount|mrp|rate)\b\s*:?\s*\d+(?:\.\d+)?/gi,
      " "
    )
    .replace(/\s+/g, " ")
    .trim();
}

function isHeaderOrNonItemLine(line) {
  const lower = line.toLowerCase();

  const ignoredWords = [
    "item name",
    "item",
    "quantity",
    "qty",
    "price per item",
    "total price",
    "totalprice",
    "description",
    "amount",
    "subtotal",
    "grand total",
    "tax",
    "gst",
    "discount",
    "thank you",
  ];

  return ignoredWords.some(
    (word) =>
      lower.includes(word)
  );
}

function cleanIngredientName(name) {
  return name
    .replace(
      /\b(?:item|product|name|description)\b/gi,
      " "
    )
    .replace(
      /\b(?:price|total|amount|rate|discount|gst|tax)\b.*$/i,
      ""
    )
    .replace(/[|:;]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeSingular(name) {
  const value = name
    .trim()
    .toLowerCase();

  const singularForms = {
    apples: "apple",
    bananas: "banana",
    oranges: "orange",
    mangoes: "mango",
    tomatoes: "tomato",
    potatoes: "potato",
    onions: "onion",
    carrots: "carrot",
    cucumbers: "cucumber",
    peppers: "pepper",
    eggs: "egg",
    loaves: "loaf",
    boxes: "box",
    bags: "bag",
    bottles: "bottle",
    jars: "jar",
    cans: "can",
    packs: "pack",
  };

  return (
    singularForms[value] ||
    value
  );
}

function capitalizeWords(text) {
  return text.replace(
    /\b\w/g,
    (character) =>
      character.toUpperCase()
  );
}