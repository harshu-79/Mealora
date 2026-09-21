const DATASET_URL =
  "/data/canonical_ingredients.csv";

let vocabularyPromise = null;

/* =========================================================
   BASIC NORMALIZATION
========================================================= */

function normalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^\w\s&'-]/g, " ")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/* =========================================================
   CSV PARSER
========================================================= */

function parseCSVLine(line) {
  const values = [];
  let current = "";
  let insideQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];

    if (char === '"') {
      if (
        insideQuotes &&
        line[i + 1] === '"'
      ) {
        current += '"';
        i += 1;
      } else {
        insideQuotes = !insideQuotes;
      }
    } else if (
      char === "," &&
      !insideQuotes
    ) {
      values.push(current);
      current = "";
    } else {
      current += char;
    }
  }

  values.push(current);

  return values;
}

/* =========================================================
   LOAD CANONICAL VOCABULARY
========================================================= */

async function loadVocabulary() {
  if (!vocabularyPromise) {
    vocabularyPromise = fetch(
      DATASET_URL
    )
      .then((response) => {
        if (!response.ok) {
          throw new Error(
            "Unable to load canonical grocery vocabulary."
          );
        }

        return response.text();
      })
      .then((text) => {
        const lines = text
          .split(/\r?\n/)
          .filter(Boolean);

        if (lines.length <= 1) {
          return [];
        }

        const headers =
          parseCSVLine(lines[0]);

        const canonicalIndex =
          headers.indexOf(
            "canonical_name"
          );

        const aliasIndex =
          headers.indexOf(
            "alias"
          );

        const normalizedAliasIndex =
          headers.indexOf(
            "normalized_alias"
          );

        const categoryIndex =
          headers.indexOf(
            "category"
          );

        const vocabulary = [];

        for (
          let i = 1;
          i < lines.length;
          i += 1
        ) {
          const values =
            parseCSVLine(
              lines[i]
            );

          const canonicalName =
            values[
              canonicalIndex
            ] || "";

          const alias =
            values[
              aliasIndex
            ] || "";

          const normalizedAlias =
            values[
              normalizedAliasIndex
            ] ||
            normalizeText(
              alias
            );

          const category =
            values[
              categoryIndex
            ] || "";

          if (
            canonicalName &&
            normalizedAlias
          ) {
            vocabulary.push({
              canonicalName,
              alias,
              normalizedAlias,
              category,
            });
          }
        }

        return vocabulary;
      });
  }

  return vocabularyPromise;
}

/* =========================================================
   INVOICE / RECEIPT NOISE FILTER
========================================================= */

const BLOCKED_TERMS = [
  "gstin",
  "invoice",
  "invoice no",
  "invoice number",
  "invoice date",
  "delivery date",
  "customer detail",
  "customer details",
  "store location",
  "place of supply",
  "phone",
  "mobile",
  "address",
  "bank",
  "account number",
  "ifsc",
  "upi",
  "gst",
  "igst",
  "cgst",
  "sgst",
  "taxable",
  "tax",
  "total",
  "subtotal",
  "grand total",
  "terms and conditions",
  "authorised signatory",
  "authorized signatory",
  "for order call",
  "original for recipient",
  "store maharashtra",
];

/* =========================================================
   DECIDE WHETHER A LINE CAN BE A PRODUCT LINE
========================================================= */

function isAdministrativeLine(line) {
  const normalized =
    normalizeText(line);

  if (!normalized) {
    return true;
  }

  if (
    BLOCKED_TERMS.some(
      (term) =>
        normalized.includes(term)
    )
  ) {
    return true;
  }

  /*
   * Phone/account-like number.
   */
  if (
    /\b\d{8,12}\b/.test(
      line
    )
  ) {
    return true;
  }

  /*
   * GSTIN-like sequence.
   */
  if (
    /\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[A-Z]\d\b/i.test(
      line
    )
  ) {
    return true;
  }

  return false;
}

/* =========================================================
   EXTRACT THE PRODUCT PORTION
========================================================= */

function extractProductCandidate(
  originalLine
) {
  if (
    isAdministrativeLine(
      originalLine
    )
  ) {
    return null;
  }

  let line =
    originalLine
      .replace(/\t/g, " ")
      .replace(/\s+/g, " ")
      .trim();

  /*
   * Remove invoice serial number:
   *
   * 1 Kellogg's ...
   * 2 Skybags ...
   */
  line =
    line.replace(
      /^\s*\d+\s*[.)-]?\s*/,
      ""
    );

  /*
   * Remove HSN/SAC codes.
   */
  line =
    line.replace(
      /\b\d{4,8}\b/g,
      " "
    );

  /*
   * Remove currency values.
   */
  line =
    line.replace(
      /(?:₹|\$)\s*\d+(?:\.\d+)?/gi,
      " "
    );

  /*
   * Find quantity + unit.
   *
   * Example:
   * Sunpride Sunflower Oil 1 LTR 107.00
   *
   * We keep:
   * Sunpride Sunflower Oil
   */
  const quantityUnitMatch =
    line.match(
      /\b\d+(?:\.\d+)?\s*(?:kg|kgs|g|gm|grams?|lb|lbs|l|ltr|litre|litres|ml|pcs?|pieces?|dozen|loaf|loaves|box|boxes|bag|bags|pack|packs|bottle|bottles|can|cans|jar|jars|cup|cups|gallon|gallons)\b/i
    );

  if (
    quantityUnitMatch
  ) {
    line =
      line.slice(
        0,
        quantityUnitMatch.index
      );
  } else {
    /*
     * If there's no unit, try:
     *
     * Product 1 1500 180 1680
     */
    const numberMatch =
      line.match(
        /\b\d+(?:\.\d+)?\b/
      );

    if (
      numberMatch
    ) {
      line =
        line.slice(
          0,
          numberMatch.index
        );
    }
  }

  /*
   * Remove leading/trailing punctuation.
   */
  line =
    line
      .replace(
        /^[\s:|,.-]+|[\s:|,.-]+$/g,
        ""
      )
      .replace(
        /\s+/g,
        " "
      )
      .trim();

  if (
    !line ||
    line.length < 2
  ) {
    return null;
  }

  return line;
}

/* =========================================================
   LEVENSHTEIN
========================================================= */

function levenshtein(
  a,
  b
) {
  const rows =
    a.length + 1;

  const columns =
    b.length + 1;

  const matrix =
    Array.from(
      {
        length: rows,
      },
      () =>
        new Array(
          columns
        ).fill(0)
    );

  for (
    let i = 0;
    i < rows;
    i += 1
  ) {
    matrix[i][0] = i;
  }

  for (
    let j = 0;
    j < columns;
    j += 1
  ) {
    matrix[0][j] = j;
  }

  for (
    let i = 1;
    i < rows;
    i += 1
  ) {
    for (
      let j = 1;
      j < columns;
      j += 1
    ) {
      const cost =
        a[i - 1] ===
        b[j - 1]
          ? 0
          : 1;

      matrix[i][j] =
        Math.min(
          matrix[i - 1][j] +
            1,
          matrix[i][j - 1] +
            1,
          matrix[i - 1][
            j - 1
          ] + cost
        );
    }
  }

  return matrix[
    rows - 1
  ][columns - 1];
}

/* =========================================================
   TEXT SIMILARITY
========================================================= */

function similarity(
  a,
  b
) {
  if (!a || !b) {
    return 0;
  }

  if (a === b) {
    return 1;
  }

  if (
    a.includes(b) ||
    b.includes(a)
  ) {
    const smaller =
      Math.min(
        a.length,
        b.length
      );

    const larger =
      Math.max(
        a.length,
        b.length
      );

    return (
      0.85 +
      (smaller /
        larger) *
        0.15
    );
  }

  const distance =
    levenshtein(
      a,
      b
    );

  return Math.max(
    0,
    1 -
      distance /
        Math.max(
          a.length,
          b.length
        )
  );
}

function tokenOverlap(
  a,
  b
) {
  const aTokens =
    new Set(
      normalizeText(a)
        .split(" ")
        .filter(Boolean)
    );

  const bTokens =
    new Set(
      normalizeText(b)
        .split(" ")
        .filter(Boolean)
    );

  if (
    aTokens.size === 0 ||
    bTokens.size === 0
  ) {
    return 0;
  }

  let intersection = 0;

  for (
    const token of aTokens
  ) {
    if (
      bTokens.has(token)
    ) {
      intersection += 1;
    }
  }

  return (
    intersection /
    Math.max(
      aTokens.size,
      bTokens.size
    )
  );
}

/* =========================================================
   MATCH PRODUCT CANDIDATE
========================================================= */

async function matchCandidate(
  candidate
) {
  const vocabulary =
    await loadVocabulary();

  const normalizedInput =
    normalizeText(
      candidate
    );

  if (
    !normalizedInput
  ) {
    return null;
  }

  let bestMatch = null;
  let bestScore = 0;

  for (
    const item of vocabulary
  ) {
    const alias =
      item.normalizedAlias;

    /*
     * Exact alias match.
     */
    if (
      normalizedInput ===
      alias
    ) {
      return {
        canonical_name:
          item.canonicalName,
        alias:
          item.alias,
        category:
          item.category,
        confidence: 1,
      };
    }

    /*
     * Strong token/similarity matching.
     */
    const similarityScore =
      similarity(
        normalizedInput,
        alias
      );

    const overlapScore =
      tokenOverlap(
        normalizedInput,
        alias
      );

    const score =
      similarityScore * 0.65 +
      overlapScore * 0.35;

    if (
      score >
      bestScore
    ) {
      bestScore =
        score;

      bestMatch = {
        canonical_name:
          item.canonicalName,

        alias:
          item.alias,

        category:
          item.category,

        confidence:
          Number(
            score.toFixed(3)
          ),
      };
    }
  }

  /*
   * Higher threshold prevents false positives
   * such as addresses accidentally matching
   * "cooking oil".
   */
  if (
    !bestMatch ||
    bestScore <
      0.68
  ) {
    return null;
  }

  return bestMatch;
}

/* =========================================================
   PUBLIC FUNCTION
========================================================= */

export async function matchGroceryText(
  text
) {
  const candidate =
    extractProductCandidate(
      text
    );

  if (!candidate) {
    return null;
  }

  const match =
    await matchCandidate(
      candidate
    );

  if (!match) {
    return null;
  }

  return {
    source_line:
      text,

    detected_text:
      candidate,

    ...match,
  };
}

/* =========================================================
   MATCH MULTIPLE OCR LINES
========================================================= */

export async function matchGroceryLines(
  lines
) {
  const results = [];

  const seen =
    new Set();

  for (
    const line of lines
  ) {
    const match =
      await matchGroceryText(
        line
      );

    if (!match) {
      continue;
    }

    const key =
      `${match.canonical_name}|${match.category}`;

    if (
      seen.has(key)
    ) {
      continue;
    }

    seen.add(key);

    results.push(
      match
    );
  }

  return results;
}