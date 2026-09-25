#!/usr/bin/env python3
from pathlib import Path
import re, shutil, datetime

ROOT=Path("/Users/harshitha/mealora")
SRC=ROOT/"frontend/src"
BACKEND=ROOT/"backend/app.py"
STAMP=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def backup(p):
    b=p.with_name(p.name+f".before_final_fix_{STAMP}")
    shutil.copy2(p,b)
    print("BACKUP",b)

def find_weekly():
    p=SRC/"WeeklyMealPlanner.jsx"
    if p.exists(): return p
    for x in SRC.glob("*.jsx"):
        try:
            t=x.read_text(encoding="utf-8")
            if "Weekly Meal Planner" in t or "Pantry utilization" in t:
                return x
        except Exception:
            pass
    return None

def patch_backend():
    if not BACKEND.exists(): return
    s=BACKEND.read_text(encoding="utf-8")
    old="""        language = data.get("language", "English")
        result = _gemini_translate_recipe(recipe, language)

        return jsonify({
"""
    new="""        language = data.get("language", "English")

        # English recipe display must work even when Gemini is unavailable.
        if str(language).strip().lower() == "english":
            result = {
                "recipe_id": recipe.get("recipe_id"),
                "recipe_name": recipe.get("recipe_name"),
                "language": "English",
                "ingredients": recipe.get("ingredients", []),
                "instructions": recipe.get("instructions", []),
                "note": "Recipe presented directly from Mealora's recipe dataset."
            }
        else:
            result = _gemini_translate_recipe(recipe, language)

        return jsonify({
"""
    if old in s:
        backup(BACKEND)
        s=s.replace(old,new,1)
        BACKEND.write_text(s,encoding="utf-8")
        print("backend fixed")
    else:
        print("backend English fallback pattern not found (may already be fixed or different file)")

def patch_weekly():
    p=find_weekly()
    if not p:
        print("Weekly planner JSX not found")
        return
    s=p.read_text(encoding="utf-8"); original=s
    backup(p)

    if "onOpenRecipe" not in s:
        for pat in [
            r'(function\s+WeeklyMealPlanner\s*\(\s*\{)([^}]*)(\}\s*\)\s*\{)',
            r'(const\s+WeeklyMealPlanner\s*=\s*\(\s*\{)([^}]*)(\}\s*\)\s*=>)'
        ]:
            m=re.search(pat,s)
            if m:
                body=m.group(2).strip()
                body=(body+", " if body else "")+"onOpenRecipe"
                s=s[:m.start()]+m.group(1)+body+m.group(3)+s[m.end():]
                break

    s=re.sub(r'(?im)^[ \t]*.*Mealora score:.*\n','',s)
    s=re.sub(r'(?im)^[ \t]*.*\bRF:.*\bHAN:.*\n','',s)

    s=s.replace("Recipe variety","Different recipes").replace("Recipe Variety","Different recipes")
    s=s.replace("Pantry utilization%","Pantry utilization")
    s=s.replace("Expiry reduction%","Expiry reduction")

    if "_weeklyPantryUtilization" not in s:
        helper="""
  // UI metrics are derived from the actual selected weekly plan.
  const _weeklyRows = Array.isArray(weeklyPlan)
    ? weeklyPlan
    : (weeklyPlan?.meals || weeklyPlan?.plan || weeklyPlan?.recipes || []);
  const _weeklyPantryUtilization = _weeklyRows.length
    ? Math.round(_weeklyRows.reduce((sum, r) => sum + Number(r.pantry_score || r.pantryScore || 0), 0) / _weeklyRows.length * 100)
    : 0;
  const _weeklyExpiryReduction = _weeklyRows.length
    ? Math.round(_weeklyRows.reduce((sum, r) => sum + Number(r.expiry_score || r.expiryScore || 0), 0) / _weeklyRows.length * 100)
    : 0;
  const _weeklyDifferentRecipes = new Set(
    _weeklyRows.map(r => String(r.recipe_name || r.recipeName || "").trim()).filter(Boolean)
  ).size;
"""
        idx=s.find("\n  return ")
        if idx>=0:
            s=s[:idx]+helper+s[idx:]

    # Conservative replacement of values directly after the labels.
    s=re.sub(r'(Pantry utilization\s*</[^>]+>\s*<[^>]+>)([^<{}]*|\{[^}]*\})(</[^>]+>)',
             r'\1{_weeklyPantryUtilization}%\3',s,count=1,flags=re.S)
    s=re.sub(r'(Expiry reduction\s*</[^>]+>\s*<[^>]+>)([^<{}]*|\{[^}]*\})(</[^>]+>)',
             r'\1{_weeklyExpiryReduction}%\3',s,count=1,flags=re.S)
    s=re.sub(r'(Different recipes\s*</[^>]+>\s*<[^>]+>)([^<{}]*|\{[^}]*\})(</[^>]+>)',
             r'\1{_weeklyDifferentRecipes}\3',s,count=1,flags=re.S)

    if "Open Recipe in English" not in s and "onOpenRecipe" in s:
        for expr in [r'\{meal\.recipe_name\}',r'\{item\.recipe_name\}',r'\{recipe\.recipe_name\}']:
            if re.search(expr,s):
                btn="""\
              <button type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  const selected = (meal || item || recipe);
                  onOpenRecipe?.({
                    recipe_id: selected?.recipe_id,
                    recipe_name: selected?.recipe_name
                  });
                }}
                style={{marginTop:"8px",width:"100%",padding:"8px 10px",borderRadius:"9px",border:"1px solid #cbd5e1",background:"#fff",cursor:"pointer",fontWeight:700}}>
                🌐 Open Recipe in English
              </button>"""
                s=re.sub(expr, lambda m:m.group(0)+"\n"+btn, s,count=1)
                break

    p.write_text(s,encoding="utf-8")
    print("weekly patched:",p, "changed=",s!=original)

def patch_app():
    APP=SRC/"App.jsx"
    if not APP.exists():
        print("App.jsx not found"); return
    s=APP.read_text(encoding="utf-8"); original=s
    backup(APP)

    # Remove the top-right recommendation score/final_score box.
    s=re.sub(
        r'\s*<div\s+style=\{\{\s*minWidth:\s*"72px"[\s\S]*?Number\(recipe\.final_score \|\| 0\)\.toFixed\(3\)[\s\S]*?</div>\s*</div>',
        '\n                      </div>', s, count=5)

    # Replace the fourth metric with nutrition instead of collaborative score.
    s=s.replace(
        '["Collaborative", `${(Number(recipe.collaborative_score || 0) * 100).toFixed(0)}%`]',
        '["Nutrition", recipe.nutrition_available ? `${(Number(recipe.nutrition_score || 0) * 100).toFixed(0)}%` : "Not available"]'
    )

    # Remove collaborative wording from the recommendation explanation.
    s=re.sub(r'\s*if \(collaborative > 0\.05\) \{[\s\S]*?\n\s*\}', '', s, count=1)

    # Pass recipe-opening callback to weekly planner if the component is used here.
    if "<WeeklyMealPlanner" in s and "onOpenRecipe=" not in s:
        s=s.replace("<WeeklyMealPlanner", "<WeeklyMealPlanner\n              onOpenRecipe={openRecipeAssistant}", 1)

    APP.write_text(s,encoding="utf-8")
    print("App.jsx patched:",s!=original)

def main():
    print("MEALORA FINAL FIX")
    patch_backend()
    patch_app()
    patch_weekly()
    print("DONE")

if __name__=="__main__":
    main()
