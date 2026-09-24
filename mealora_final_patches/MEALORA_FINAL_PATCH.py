#!/usr/bin/env python3
from pathlib import Path
import re, shutil
ROOT=Path.cwd(); FRONTEND=ROOT/'frontend'/'src'; BACKEND=ROOT/'backend'; NOTEBOOKS=ROOT/'notebooks'
APP=FRONTEND/'App.jsx'; WEEKLY=FRONTEND/'WeeklyMealPlanner.jsx'; BACKEND_APP=BACKEND/'app.py'; RANKER=NOTEBOOKS/'final_ranker.py'
def backup(p):
    if p.exists():
        b=p.with_suffix(p.suffix+'.before_final_patch')
        if not b.exists(): shutil.copy2(p,b); print('Backup:',b)
def rep(t,o,n,label):
    if o not in t: print('SKIP:',label); return t
    print('PATCH:',label); return t.replace(o,n,1)
if APP.exists():
    backup(APP); t=APP.read_text(encoding='utf-8')
    t=t.replace('  const collaborative = Number(recipe.collaborative_score || 0);\n','')
    t=t.replace('''  if (collaborative > 0.05) {\n    reasons.push("👥 Supported by user-preference signals");\n  }\n\n''','')
    t=t.replace('<div style={{ fontSize: "11px", color: "#64748b" }}>Score</div>','<div style={{ fontSize: "11px", color: "#64748b" }}>ML suitability</div>')
    old='''[\n                          ["Pantry match", `${(Number(recipe.pantry_score || 0) * 100).toFixed(0)}%`],\n                          ["Expiry score", `${(Number(recipe.expiry_score || 0) * 100).toFixed(0)}%`],\n                          ["Content match", `${(Number(recipe.content_score || 0) * 100).toFixed(0)}%`],\n                          ["Collaborative", `${(Number(recipe.collaborative_score || 0) * 100).toFixed(0)}%`],\n                        ]'''
    new='''[\n                          ["Pantry match", `${(Number(recipe.pantry_score || 0) * 100).toFixed(0)}%`],\n                          ["Expiry score", `${(Number(recipe.expiry_score || 0) * 100).toFixed(0)}%`],\n                          ["Content match", `${(Number(recipe.content_score || 0) * 100).toFixed(0)}%`],\n                          ["Nutrition", recipe.nutrition_available ? `${Number(recipe.calories || 0).toFixed(0)} kcal` : "Not available"],\n                        ]'''
    t=rep(t,old,new,'recommendation cards')
    marker='''                      <div\n                        style={{\n                          marginTop: "14px",\n                          padding: "14px",\n                          borderRadius: "12px",\n                          background:\n                            "linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%)",'''
    panel='''                      {recipe.nutrition_available ? (\n                        <div style={{ marginTop: "12px", padding: "14px", borderRadius: "12px", background: "#f0fdf4", border: "1px solid #dcfce7" }}>\n                          <strong>🥗 Nutrition per serving</strong>\n                          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: "8px", marginTop: "10px", fontSize: "12px", color: "#475569" }}>\n                            <span>Calories: <b>{Number(recipe.calories || 0).toFixed(0)} kcal</b></span>\n                            <span>Protein: <b>{Number(recipe.protein || 0).toFixed(1)} g</b></span>\n                            <span>Carbohydrates: <b>{Number(recipe.carbohydrates || 0).toFixed(1)} g</b></span>\n                            <span>Fat: <b>{Number(recipe.fat || 0).toFixed(1)} g</b></span>\n                            <span>Fibre: <b>{Number(recipe.fibre || 0).toFixed(1)} g</b></span>\n                            <span>Sodium: <b>{Number(recipe.sodium || 0).toFixed(0)} mg</b></span>\n                          </div>\n                        </div>\n                      ) : null}\n\n'''
    t=rep(t,marker,panel+marker,'nutrition panel')
    def tag(m):
        w=m.group(0)
        return w if 'onOpenRecipe=' in w else f'<WeeklyMealPlanner{m.group(1)} onOpenRecipe={{openRecipeAssistant}}>'
    t=re.sub(r'<WeeklyMealPlanner\\b([^>]*)>',tag,t)
    APP.write_text(t,encoding='utf-8')
if WEEKLY.exists():
    backup(WEEKLY); t=WEEKLY.read_text(encoding='utf-8')
    t=t.replace('export default function WeeklyMealPlanner({ pantryItems = [] }) {','export default function WeeklyMealPlanner({ pantryItems = [], onOpenRecipe }) {',1)
    old='''{summary && (\n        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginTop: 18 }}>\n          <div className="section-card"><small>Pantry utilization</small><strong>{summary.pantry_utilization}%</strong></div>\n          <div className="section-card"><small>Expiry reduction</small><strong>{summary.expiry_focus}%</strong></div>\n          <div className="section-card"><small>Recipe variety</small><strong>{summary.unique_recipes}</strong></div>\n          <div className="section-card"><small>Meals planned</small><strong>{plan.length}</strong></div>\n        </div>\n      )}'''
    new='''{summary && (\n        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginTop: 18 }}>\n          <div className="section-card"><small>Pantry usage</small><strong>{summary.pantry_utilization != null ? `${Number(summary.pantry_utilization).toFixed(0)}%` : summary.pantry_usage != null ? `${Number(summary.pantry_usage).toFixed(0)}%` : summary.pantry_score != null ? `${(Number(summary.pantry_score)*100).toFixed(0)}%` : "—"}</strong></div>\n          <div className="section-card"><small>Expiry reduction</small><strong>{summary.expiry_reduction != null ? `${Number(summary.expiry_reduction).toFixed(0)}%` : summary.expiry_focus != null ? `${Number(summary.expiry_focus).toFixed(0)}%` : summary.waste_reduction != null ? `${(Number(summary.waste_reduction)*100).toFixed(0)}%` : "—"}</strong></div>\n          <div className="section-card"><small>Different recipes</small><strong>{summary.unique_recipes ?? summary.recipe_variety ?? new Set(plan.flatMap(d => d.meals || []).map(m => m.recipe_name)).size}</strong></div>\n          <div className="section-card"><small>Meals planned</small><strong>{plan.reduce((n,d)=>n+(d.meals?.length||0),0)}</strong></div>\n        </div>\n      )}'''
    t=rep(t,old,new,'weekly metrics')
    oldcard='''                  <div key={meal.meal_type} style={{ padding: 14, borderRadius: 12, background: "#f8fafc" }}>\n                    <div style={{ fontSize: 12, color: "#64748b", fontWeight: 700 }}>{meal.meal_type.toUpperCase()}</div>\n                    <strong style={{ display: "block", marginTop: 5 }}>{meal.recipe_name}</strong>\n                    <div style={{ marginTop: 8, fontSize: 13, color: "#64748b" }}>\n                      Score: {Number(meal.optimization_score || 0).toFixed(2)} · {meal.cooking_time || "Time N/A"} min\n                    </div>\n                    <div style={{ marginTop: 7, fontSize: 13 }}>{meal.reason}</div>\n                  </div>'''
    newcard='''                  <div key={meal.meal_type} onClick={() => onOpenRecipe && onOpenRecipe({ recipe_id: meal.recipe_id, recipe_name: meal.recipe_name })} role={onOpenRecipe ? "button" : undefined} tabIndex={onOpenRecipe ? 0 : undefined} style={{ padding: 14, borderRadius: 12, background: "#f8fafc", cursor: onOpenRecipe ? "pointer" : "default" }}>\n                    <div style={{ fontSize: 12, color: "#64748b", fontWeight: 700 }}>{String(meal.meal_type || "").toUpperCase()}</div>\n                    <strong style={{ display: "block", marginTop: 5 }}>{meal.recipe_name}</strong>\n                    <div style={{ marginTop: 9, fontSize: 13, color: "#1e293b", fontWeight: 700 }}>🌐 View Recipe</div>\n                    {meal.reason ? <div style={{ marginTop: 7, fontSize: 13 }}>{meal.reason}</div> : null}\n                  </div>'''
    t=rep(t,oldcard,newcard,'weekly recipe click')
    WEEKLY.write_text(t,encoding='utf-8')
if RANKER.exists():
    backup(RANKER); t=RANKER.read_text(encoding='utf-8')
    marker='''    return result[\n        [\n            "recipe_id",\n            "recipe_name",'''
    if marker in t and '"nutrition_available"' not in t:
        block='''    nutrition_cols = ["calories","carbohydrates","protein","fat","fibre","sodium","calcium","iron","vitamin_c","folate"]\n    for col in nutrition_cols:\n        if col not in result.columns:\n            result[col] = float("nan")\n    result["nutrition_available"] = result[nutrition_cols].notna().any(axis=1).astype(int)\n\n'''
        t=t.replace(marker,block+marker,1)
        needle='''            "hgb_suitability_score",\n            "final_score",\n        ]'''
        repl='''            "hgb_suitability_score",\n            "final_score",\n            "nutrition_available", "calories", "carbohydrates", "protein", "fat", "fibre",\n            "sodium", "calcium", "iron", "vitamin_c", "folate",\n        ]'''
        t=rep(t,needle,repl,'ranker nutrition fields')
    RANKER.write_text(t,encoding='utf-8')
if BACKEND_APP.exists():
    backup(BACKEND_APP); t=BACKEND_APP.read_text(encoding='utf-8')
    func=r'''@app.route("/api/leftover-recommendations", methods=["POST"])
def leftover_recommendations():
    try:
        import pandas as pd, re
        data=request.get_json() or {}; leftover=str(data.get("leftover","")).strip().lower()
        if not leftover: return jsonify({"success":False,"error":"Enter a leftover food or ingredient."}),400
        df=pd.read_csv(RECIPE_DATASET,low_memory=False)
        def pick(cols):
            return next((c for c in cols if c in df.columns),None)
        name_col=pick(["recipe_name","name","title","final_food_name"])
        ing_col=pick(["ingredients","ingredient_names","ingredient_list","Cleaned-Ingredients","cleaned_ingredients","TranslatedIngredients","ingredients_clean"])
        cuisine_col=pick(["cuisine","Cuisine","cuisine_type"])
        if not name_col or not ing_col: raise RuntimeError("Recipe dataset does not contain recipe-name and ingredient columns.")
        def tok(v): return {x for x in re.findall(r"[a-z0-9]+",str(v).lower()) if len(x)>2}
        q=tok(leftover); rows=[]
        for _,row in df.iterrows():
            name=str(row.get(name_col,"") or ""); ing=str(row.get(ing_col,"") or ""); hay=(name+" "+ing).lower(); overlap=q & tok(hay)
            if not overlap: continue
            score=len(overlap)/max(len(q),1)
            if leftover in hay: score+=.35
            rows.append({"recipe_id":str(row.get("recipe_id",row.get("id",""))),"recipe_name":name,"ingredients":ing,"cuisine":str(row.get(cuisine_col,"") or "") if cuisine_col else "","match_score":round(min(score,1),4),"reason":f"Uses ingredients matching '{leftover}'."})
        rows=sorted(rows,key=lambda x:x["match_score"],reverse=True)[:5]
        return jsonify({"success":True,"leftover":leftover,"reuse_possible":bool(rows),"recommendations":rows,"donation_suggestion":not bool(rows)})
    except Exception as e:
        import traceback; traceback.print_exc(); return jsonify({"success":False,"error":str(e)}),500
'''
    pat=re.compile(r'@app\.route\("/api/leftover-recommendations".*?\n(?=@app\.route|\nif __name__)',re.S)
    if pat.search(t): t=pat.sub(func,t,count=1); print('PATCH leftover endpoint')
    else:
        idx=t.find('\nif __name__'); t=(t+"\n"+func) if idx==-1 else (t[:idx]+"\n"+func+t[idx:]); print('ADD leftover endpoint')
    BACKEND_APP.write_text(t,encoding='utf-8')
print('DONE')
