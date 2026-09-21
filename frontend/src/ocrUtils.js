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

  const unitMap = {
    kg: "kg",
    kgs: "kg",
    kilo: "kg",
    kilos: "kg",
    kilogram: "kg",
    kilograms: "kg",

    g: "g",
    gm: "g",
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
    ltr: "litre",

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

    packet: "packet",
    packets: "packet",

    bottle: "bottle",
    bottles: "bottle",

    can: "can",
    cans: "can",

    jar: "jar",
    jars: "jar",

    cup: "cup",
    cups: "cup",

    gallon: "gallon",
    gallons: "gallon",
  };

  /*
   * FIRST PRIORITY:
   * Find an explicit quantity + unit ANYWHERE
   * in the OCR line.
   *
   * Examples:
   * Rice 5 kg Rs 30/kg 150.00
   * Onion 500 gm Rs 16/kg 8.00
   * Mustard oil 2 litre Rs 45/l 90.00
   * Surf 2 packets Rs 30 60.00
   * Sunflower Oil 1 LTR 107.00
   */

  const explicitMatch = cleaned.match(
    /\b(\d+(?:\.\d+)?)\s*(kg|kgs|kilo|kilos|kilogram|kilograms|g|gm|gram|grams|lb|lbs|pound|pounds|l|ltr|liter|liters|litre|litres|ml|milliliter|milliliters|millilitre|millilitres|piece|pieces|pc|pcs|dozen|loaf|loaves|box|boxes|bag|bags|pack|packs|packet|packets|bottle|bottles|can|cans|jar|jars|cup|cups|gallon|gallons)\b/i
  );

  if (explicitMatch) {
    let ingredient = cleaned
      .slice(0, explicitMatch.index)
      .trim();

    /*
     * Remove bill serial number:
     * 1. Rice
     * 2 Tomatoes
     * 4. Mustard oil
     */
    ingredient = ingredient.replace(
      /^\s*\d+\s*[.)-]?\s*/,
      ""
    );

    /*
     * Remove HSN-like codes that appear before
     * the real quantity.
     */
    ingredient = ingredient.replace(
      /\b\d{4,8}\b/g,
      " "
    );

    ingredient = ingredient
      .replace(/\s+/g, " ")
      .trim();

    if (ingredient) {
      return {
        ingredient,
        quantity: Number(explicitMatch[1]),
        unit:
          unitMap[
            explicitMatch[2].toLowerCase()
          ] || "piece",
      };
    }
  }

  /*
   * SECOND PRIORITY:
   * Table rows that have an HSN/code followed
   * by a small quantity without an explicit unit.
   *
   * Example:
   * 2 Skybags Ronan School Backpack - 1006 1 1500...
   *
   * 1006 = HSN
   * 1    = quantity
   */

  const hsnMatch = cleaned.match(
    /\b\d{4,8}\s+(\d+(?:\.\d+)?)\b/
  );

  if (hsnMatch) {
    const quantity = Number(hsnMatch[1]);

    if (
      quantity > 0 &&
      quantity <= 100
    ) {
      let ingredient = cleaned
        .slice(0, hsnMatch.index)
        .trim();

      ingredient = ingredient.replace(
        /^\s*\d+\s*[.)-]?\s*/,
        ""
      );

      ingredient = ingredient
        .replace(/\b\d{4,8}\b/g, " ")
        .replace(/\s+/g, " ")
        .trim();

      if (ingredient) {
        return {
          ingredient,
          quantity,
          unit: "piece",
        };
      }
    }
  }

  /*
   * THIRD PRIORITY:
   * "Apples 3"
   * "Bread 2"
   *
   * Only use a standalone number at the END
   * of the line, never the first number.
   */

  const trailingMatch = cleaned.match(
    /^(.*\D)\s+(\d+(?:\.\d+)?)$/
  );

  if (trailingMatch) {
    const quantity =
      Number(trailingMatch[2]);

    if (
      quantity > 0 &&
      quantity <= 100
    ) {
      const ingredient =
        trailingMatch[1]
          .replace(
            /^\s*\d+\s*[.)-]?\s*/,
            ""
          )
          .trim();

      if (ingredient) {
        return {
          ingredient,
          quantity,
          unit: "piece",
        };
      }
    }
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