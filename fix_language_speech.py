from pathlib import Path
from datetime import datetime
import shutil

APP = Path("/Users/harshitha/mealora/frontend/src/App.jsx")
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

s = APP.read_text(encoding="utf-8")
shutil.copy2(APP, APP.with_name(f"App.jsx.before_language_speech_fix_{stamp}"))

# ---------------------------------------------------------
# 1. Add translation cache
# ---------------------------------------------------------
if "assistantTranslationCacheRef" not in s:
    marker = "const recognitionRef = useRef(null);"
    replacement = """const recognitionRef = useRef(null);
  const assistantTranslationCacheRef = useRef({});"""
    s = s.replace(marker, replacement, 1)

# ---------------------------------------------------------
# 2. Cache recipe translations inside openRecipeAssistant
# ---------------------------------------------------------
old = """    const requestId = ++assistantRequestRef.current;
    setAssistantLoading(true);
    setAssistantError("");

    try {
      const response = await fetch(RECIPE_ASSISTANT_API, {"""

new = """    const cacheKey = `${sourceRecipe.recipe_id || sourceRecipe.recipe_name}__${requestedLanguage}`;

    // Reuse an already translated recipe immediately.
    const cachedRecipe = assistantTranslationCacheRef.current[cacheKey];
    if (cachedRecipe) {
      setAssistantRecipe(cachedRecipe);
      setAssistantLoading(false);
      setAssistantError("");
      return;
    }

    const requestId = ++assistantRequestRef.current;
    setAssistantLoading(true);
    setAssistantError("");

    try {
      const response = await fetch(RECIPE_ASSISTANT_API, {"""

if old in s:
    s = s.replace(old, new, 1)

# ---------------------------------------------------------
# 3. Store successful translation in cache
# ---------------------------------------------------------
old = """      setAssistantRecipe({
        ...data.recipe,
        ingredients: normalizeRecipeLines(data.recipe?.ingredients),
        instructions: normalizeRecipeLines(data.recipe?.instructions),
      });"""

new = """      const normalizedRecipe = {
        ...data.recipe,
        ingredients: normalizeRecipeLines(data.recipe?.ingredients),
        instructions: normalizeRecipeLines(data.recipe?.instructions),
      };

      assistantTranslationCacheRef.current[cacheKey] = normalizedRecipe;
      setAssistantRecipe(normalizedRecipe);"""

if old in s:
    s = s.replace(old, new, 1)

# ---------------------------------------------------------
# 4. Replace speakRecipe with natural speech normalization
# ---------------------------------------------------------
start = s.find("  function speakRecipe() {")
end = s.find("\n  async function loadLiveContext()", start)

if start != -1 and end != -1:
    new_speech = r'''  function speakRecipe() {
    if (!assistantRecipe || !window.speechSynthesis) return;

    window.speechSynthesis.cancel();

    function speechText(value) {
      return String(value || "")
        // Common cooking units — prevent letter-by-letter reading.
        .replace(/\bC[\s.-]*U[\s.-]*P(?:S)?\b/gi, "cups")
        .replace(/\bT[\s.-]*B[\s.-]*S[\s.-]*P\b/gi, "tablespoon")
        .replace(/\bT[\s.-]*S[\s.-]*P\b/gi, "teaspoon")
        .replace(/\bTBSP\b/gi, "tablespoon")
        .replace(/\bTSP\b/gi, "teaspoon")
        .replace(/\bKG\b/gi, "kilograms")
        .replace(/\bG\b/gi, "grams")
        .replace(/\bML\b/gi, "milliliters")
        .replace(/\bL\b/gi, "liters")
        .replace(/\bOZ\b/gi, "ounces")
        .replace(/\bLB\b/gi, "pounds")
        // Common fraction symbols.
        .replace(/½/g, " one half ")
        .replace(/¼/g, " one quarter ")
        .replace(/¾/g, " three quarters ")
        .replace(/⅓/g, " one third ")
        .replace(/⅔/g, " two thirds ")
        .replace(/⅛/g, " one eighth ")
        .replace(/⅜/g, " three eighths ")
        .replace(/⅝/g, " five eighths ")
        .replace(/⅞/g, " seven eighths ")
        // Prevent short ALL-CAPS words from being spelled out.
        .replace(/\b([A-Z]{2,5})\b/g, (match) => match.toLowerCase())
        .replace(/\s+/g, " ")
        .trim();
    }

    const ingredients = normalizeRecipeLines(assistantRecipe.ingredients);
    const instructions = normalizeRecipeLines(assistantRecipe.instructions);

    const text = [
      speechText(assistantRecipe.recipe_name),
      "Ingredients.",
      ...ingredients.map(speechText),
      "Instructions.",
      ...instructions.map(speechText),
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
    utterance.pitch = 1;

    window.speechSynthesis.speak(utterance);
  }
'''
    s = s[:start] + new_speech + s[end:]

APP.write_text(s, encoding="utf-8")

print("✅ Language caching added.")
print("✅ Read-aloud unit normalization added.")
print("✅ Fraction pronunciation improved.")
print("✅ Backup created.")
