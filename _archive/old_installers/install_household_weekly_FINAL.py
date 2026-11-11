
from pathlib import Path
from datetime import datetime
import shutil, re

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend" / "app.py"
if not BACKEND.exists():
    raise SystemExit(f"ERROR: backend/app.py not found at {BACKEND}")

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP = BACKEND.with_name(f"app.py.before_household_weekly_{STAMP}.py")

START = "# ===== MEALORA HOUSEHOLD WEEKLY PLANNER START ====="
END = "# ===== MEALORA HOUSEHOLD WEEKLY PLANNER END ====="

ROUTE = r'''
# ===== MEALORA HOUSEHOLD WEEKLY PLANNER START =====
# Household-first planner:
# pantry -> role-compatible candidates -> regional/context scoring
# -> practical meal composition -> compact NSGA-II optimization.
@app.route("/api/weekly-plan", methods=["POST"])
def weekly_plan_household():
    try:
        import ast
        import json
        import random
        import numpy as np
        import pandas as pd

        data = request.get_json() or {}
        pantry = data.get("pantry") or []
        household = data.get("household") or {}
        adults = max(int(household.get("adults", 0) or 0), 0)
        children = max(int(household.get("children", 0) or 0), 0)
        seniors = max(int(household.get("seniors", 0) or 0), 0)
        if adults + children + seniors <= 0:
            return jsonify({"success": False, "error": "Add at least one household member."}), 400

        diet = str(data.get("diet", "Mixed") or "Mixed")
        cuisine_requested = str(data.get("cuisine", "Auto") or "Auto")
        max_time = float(data.get("max_time", 45) or 45)
        include_snacks = bool(data.get("include_snacks", True))
        special_day = str(data.get("special_day", "Saturday") or "None")
        city_input = str(data.get("location_city") or "").strip()

        ctx = build_context({
            "mode": data.get("mode", "automatic"),
            "date": data.get("date"),
            "time": data.get("time"),
            "location_city": city_input,
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude"),
            "meal_type": data.get("meal_type", "Automatic"),
            "diet": diet,
            "cuisine": cuisine_requested if cuisine_requested != "Auto" else "Automatic",
            "max_time": max_time,
        })

        city = str(ctx.get("location_city") or city_input or "").strip()
        weather = ctx.get("weather") or {}
        weather_condition = str(weather.get("condition") or "").lower()
        temperature = weather.get("temperature")

        aliases = {
            "chilli": "chili", "green chilli": "green chili", "red chilli": "red chili",
            "capsicum": "bell pepper", "coriander leaves": "coriander",
            "coriander leaf": "coriander", "curd": "yogurt", "dahi": "yogurt",
            "brinjal": "eggplant", "lady s finger": "okra", "ladys finger": "okra",
            "bhindi": "okra", "pappu": "dal", "toor dal": "pigeon pea",
            "tuvar dal": "pigeon pea", "atta": "wheat flour", "maida": "refined flour",
            "semolina": "rava", "sooji": "rava", "jeera": "cumin",
            "haldi": "turmeric", "elaichi": "cardamom"
        }
        stop = {
            "the","and","with","for","from","into","cup","cups","tsp","tbsp",
            "tablespoon","tablespoons","teaspoon","teaspoons","piece","pieces",
            "small","large","medium","fresh","finely","roughly","chopped",
            "sliced","diced","minced","crushed","grated","to","taste","as","needed",
            "of","oil","water"
        }

        def norm(v):
            s = str(v or "").lower().replace("&", " and ")
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            for a, b in sorted(aliases.items(), key=lambda x: -len(x[0])):
                s = re.sub(r"\b" + re.escape(a) + r"\b", b, s)
            return s

        def toks(v):
            return {x for x in norm(v).split() if len(x) >= 3 and x not in stop}

        def clean_ing(v):
            s = str(v or "").strip()
            if not s or s.lower() in ("nan", "none", "null"):
                return ""
            s = re.sub(r"^\s*\d+(?:\.\d+)?\s*(?:kg|g|mg|ml|l|litre|liter|cup|cups|tbsp|tsp|piece|pieces|clove|cloves)?\s*(?:of\s+)?", "", s, flags=re.I)
            s = re.sub(r"\([^)]*\)", " ", s)
            s = re.sub(r"\b(chopped|sliced|diced|minced|grated|crushed|finely|roughly|freshly|to taste|as needed)\b", " ", s, flags=re.I)
            return re.sub(r"\s+", " ", s).strip(" ,.-")

        def parse_ingredients(v):
            if isinstance(v, list):
                return [clean_ing(x) for x in v if clean_ing(x)]
            s = str(v or "").strip()
            if not s or s.lower() == "nan":
                return []
            if s.startswith("[") and s.endswith("]"):
                for parser in (json.loads, ast.literal_eval):
                    try:
                        q = parser(s)
                        if isinstance(q, list):
                            return [clean_ing(x) for x in q if clean_ing(x)]
                    except Exception:
                        pass
            return [clean_ing(x) for x in s.replace("|", ",").replace(";", ",").split(",") if clean_ing(x)]

        pantry_names, expiry_names = [], []
        today = datetime.now().date()
        for item in pantry:
            n = item.get("normalized_name") or item.get("ingredient_name") or item.get("name")
            if not n:
                continue
            n = clean_ing(n)
            if not n:
                continue
            pantry_names.append(n)
            ex = item.get("expiry_date")
            if ex:
                try:
                    d = datetime.strptime(str(ex)[:10], "%Y-%m-%d").date()
                    if (d - today).days <= 7:
                        expiry_names.append(n)
                except Exception:
                    pass

        pantry_names = list(dict.fromkeys(pantry_names))
        expiry_names = list(dict.fromkeys(expiry_names))
        if not pantry_names:
            return jsonify({"success": False, "error": "Your pantry is empty. Add ingredients before generating a weekly plan."}), 400

        pantry_token_sets = [(p, toks(p)) for p in pantry_names if toks(p)]

        def ingredient_match(ingredient):
            a, at = norm(ingredient), toks(ingredient)
            for p, pt in pantry_token_sets:
                pn = norm(p)
                if a == pn or a in pn or pn in a or (at & pt):
                    return True
            return False

        def expiry_match(ingredient):
            return any(norm(ingredient) == norm(e) or norm(ingredient) in norm(e) or norm(e) in norm(ingredient) for e in expiry_names)

        if not hasattr(weekly_plan_household, "_df"):
            weekly_plan_household._df = pd.read_csv(RECIPE_DATASET, low_memory=False)
        df = weekly_plan_household._df.copy()

        def find_col(names):
            lower = {str(c).lower(): c for c in df.columns}
            for n in names:
                if n.lower() in lower:
                    return lower[n.lower()]
            return None

        name_col = find_col(["recipe_name","name","title","dish_name"])
        id_col = find_col(["recipe_id","id","recipeid"])
        ing_col = find_col(["ingredients","ingredient_names","ingredient_list","cleaned_ingredients","Cleaned-Ingredients","Ingredients"])
        meal_col = find_col(["meal_type","meal","mealtype"])
        cuisine_col = find_col(["cuisine","cuisine_type"])
        state_col = find_col(["state","state_name"])
        region_col = find_col(["region","region_name","regional"])
        diet_col = find_col(["diet","diet_type","diet_category"])
        time_col = find_col(["cooking_time_minutes","cooking_time","cook_time","total_time_minutes","total_time"])
        weather_col = find_col(["weather","weather_suitability","weather_type"])
        season_col = find_col(["season","seasonality"])

        if not name_col or not ing_col:
            raise RuntimeError("Recipe dataset must contain recipe_name and ingredients columns.")

        city_key = norm(city)
        city_regions = {
            "hyderabad":["hyderabad","hyderabadi","telangana","telugu","andhra","deccan"],
            "secunderabad":["hyderabad","hyderabadi","telangana","telugu","andhra","deccan"],
            "warangal":["telangana","telugu","andhra"],
            "vijayawada":["andhra","andhra pradesh","telugu","coastal andhra","vijayawada"],
            "visakhapatnam":["andhra","andhra pradesh","telugu","vizag","coastal andhra"],
            "vizag":["andhra","andhra pradesh","telugu","vizag"],
            "tirupati":["andhra","andhra pradesh","telugu"],
            "chennai":["tamil","tamil nadu","tamilnadu","chennai","chettinad"],
            "coimbatore":["tamil","tamil nadu","kongu"],
            "madurai":["tamil","tamil nadu","madurai","chettinad"],
            "bengaluru":["karnataka","kannada","udupi","mangalorean","bengaluru"],
            "bangalore":["karnataka","kannada","udupi","mangalorean","bengaluru"],
            "kochi":["kerala","malayalam","malabar","kochi"],
            "mumbai":["maharashtra","marathi","konkani","mumbai"],
            "kolkata":["bengal","bengali","west bengal","kolkata"],
            "delhi":["delhi","punjabi","mughlai","north indian"],
            "ahmedabad":["gujarat","gujarati","ahmedabad"],
            "jaipur":["rajasthan","rajasthani","jaipur"],
            "lucknow":["uttar pradesh","awadhi","lucknow"]
        }
        region_terms = city_regions.get(city_key, [])
        explicit_cuisine = norm(cuisine_requested)
        if explicit_cuisine not in ("auto","automatic","mixed","none","indian"):
            region_terms = list(dict.fromkeys(region_terms + [explicit_cuisine]))

        def row_text(row):
            return " ".join(str(row.get(c, "")) for c in [cuisine_col,state_col,region_col,name_col] if c).lower()

        def regional_score(row):
            if not region_terms:
                return 0.0
            text = row_text(row)
            hits = sum(1 for t in region_terms if t and t in text)
            return 1.0 if hits >= 2 else (0.75 if hits == 1 else 0.0)

        def classify(name, ingredients, meal_text=""):
            txt, mt = norm(name + " " + " ".join(ingredients)), norm(meal_text)
            mains = ["idli","dosa","uttapam","upma","poha","pongal","pessarattu","pesarattu","vada","appe",
                     "paniyaram","ragi dosa","adai","appam","puttu","paratha","puri","poori","chapati","roti",
                     "omelette","omelet","egg dosa","egg appam","sandwich","bread"]
            chutney = ["chutney","pachadi","thogayal","thuvayal","chammanthi","pickle","achar"]
            accomp = ["sambar","sambhar","dal","pappu","stew","kurma","korma","kadala curry","chana curry"]
            rice = ["white rice","rice","pulihora","lemon rice","curd rice","coconut rice","tomato rice",
                    "biryani","biriyani","pulao","fried rice","khichdi","khichuri"]
            flat = ["chapati","roti","naan","paratha","phulka","kulcha"]
            curry = ["curry","kurma","korma","masala","poriyal","porial","kootu","sabzi","dal","pappu",
                     "sambar","rasam","chicken","mutton","fish","prawn","prawns","shrimp","egg curry",
                     "paneer","kofta","stew"]
            veg = ["poriyal","kootu","sabzi","vegetable","thoran","bhaji","beans","cabbage","carrot",
                   "potato","brinjal","eggplant","okra","avial"]
            snack = ["pakora","pakoda","bajji","bonda","samosa","cutlet","murukku","mixture","sandwich",
                     "cake","laddu","ladoo","gulab jamun","halwa","tea","coffee","sundal"]

            if any(x in txt for x in chutney): return "breakfast_side"
            if any(x in txt for x in accomp): return "breakfast_accompaniment"
            if any(x in txt for x in mains): return "breakfast_main"
            if any(x in txt for x in snack): return "snack"
            if any(x in txt for x in flat): return "flatbread_main"
            if any(x in txt for x in rice): return "rice_main"
            if any(x in txt for x in veg): return "vegetable_side"
            if any(x in txt for x in curry): return "curry_side"
            if "breakfast" in mt or "morning" in mt: return "breakfast_main"
            if "snack" in mt: return "snack"
            return "other"

        staples = {"salt","sugar","water","oil","sunflower oil","groundnut oil","coconut oil",
                   "mustard oil","sesame oil","ghee","butter","black pepper","turmeric",
                   "turmeric powder","red chili","red chili powder","coriander powder",
                   "cumin","cumin powder","garam masala","hing","asafoetida"}

        def learned_score(row):
            for c in ["final_score","mealora_score","rf_suitability_score","ml_suitability_score","hgb_suitability_score","han_score"]:
                if c in row.index:
                    try:
                        v = float(row[c])
                        if np.isfinite(v):
                            return float(np.clip(v,0,1))
                    except Exception:
                        pass
            return 0.5

        def pantry_score(ingredients):
            matched = [x for x in ingredients if ingredient_match(x)]
            core = [x for x in ingredients if norm(x) not in staples and toks(x)]
            core_match = [x for x in core if ingredient_match(x)]
            core_cov = len(core_match) / max(1,len(core))
            all_cov = len(matched) / max(1,len(ingredients))
            score = min(1.0, 0.82*core_cov + 0.18*all_cov)
            matched_names = []
            for x in matched:
                for p,pt in pantry_token_sets:
                    if toks(x) & pt:
                        matched_names.append(p)
                        break
            return score,core_cov,list(dict.fromkeys(matched_names))

        def expiry_score(ingredients):
            if not expiry_names or not ingredients:
                return 0.0
            return float(min(1.0,sum(1 for x in ingredients if expiry_match(x))/max(1,len(expiry_names))))

        def diet_score(row):
            if diet.lower() == "mixed":
                return 1.0
            txt = " ".join(str(row.get(c,"")) for c in [name_col,diet_col,ing_col] if c).lower()
            nonveg = any(x in txt for x in ["chicken","mutton","fish","prawn","prawns","shrimp","egg","meat"])
            if diet.lower() == "vegetarian":
                return 0.0 if nonveg else 1.0
            return 1.0 if nonveg else 0.45

        def time_score(row):
            if not time_col:
                return 0.75
            try:
                t=float(row[time_col])
                return 1.0 if t<=max_time else max(0.0,1.0-(t-max_time)/max(max_time,1))
            except Exception:
                return 0.75

        records=[]
        for idx,row in df.iterrows():
            name=str(row.get(name_col,"")).strip()
            if not name or name.lower()=="nan":
                continue
            ingredients=parse_ingredients(row.get(ing_col,""))
            if not ingredients:
                continue

            ps,core_ps,matched=pantry_score(ingredients)
            ex=expiry_score(ingredients)
            reg=regional_score(row)
            dietfit=diet_score(row)
            timefit=time_score(row)
            mlfit=learned_score(row)

            weatherfit=0.5
            if weather_condition and weather_col and weather_condition in str(row.get(weather_col,"")).lower():
                weatherfit=1.0
            if temperature is not None:
                try:
                    temp=float(temperature)
                    txt=" ".join(str(row.get(c,"")) for c in [name_col,weather_col,season_col] if c).lower()
                    if temp>=30 and any(x in txt for x in ["curd","raita","lemon","buttermilk","summer"]):
                        weatherfit=max(weatherfit,0.75)
                    if temp<=20 and any(x in txt for x in ["rasam","soup","stew","hot","warm"]):
                        weatherfit=max(weatherfit,0.75)
                except Exception:
                    pass

            total=(0.34*core_ps+0.16*ps+0.14*reg+0.08*ex+0.08*dietfit+
                   0.07*timefit+0.07*mlfit+0.04*weatherfit+0.02*reg)

            records.append({
                "recipe_id":str(row.get(id_col)) if id_col else str(idx),
                "recipe_name":name,"ingredients":ingredients,
                "role":classify(name,ingredients,row.get(meal_col,"") if meal_col else ""),
                "pantry":ps,"core_pantry":core_ps,"expiry":ex,"regional":reg,
                "dietfit":dietfit,"timefit":timefit,"mlfit":mlfit,
                "weatherfit":weatherfit,"total":total,"matched":matched
            })

        recipes=pd.DataFrame(records)
        if recipes.empty:
            return jsonify({"success":False,"error":"No usable recipes were found in the Mealora dataset."}),400

        recipes["_name_key"]=recipes.recipe_name.map(lambda x:re.sub(r"[^a-z0-9]+"," ",str(x).lower()).strip())
        recipes=recipes.sort_values(["total","core_pantry","regional","expiry"],ascending=False).drop_duplicates("_name_key").reset_index(drop=True)

        regional=recipes[recipes.regional>=0.75].copy()
        candidate_base=regional if len(regional)>=12 else recipes.copy()
        if len(regional)<12 and region_terms:
            candidate_base["_rank"]=candidate_base.total+0.20*candidate_base.regional
            candidate_base=candidate_base.sort_values("_rank",ascending=False)

        def top_role(role,n=20):
            return candidate_base[candidate_base.role==role].sort_values(
                ["total","core_pantry","expiry","regional"],ascending=False
            ).head(n).reset_index(drop=True)

        bmain=top_role("breakfast_main",20)
        bside=top_role("breakfast_side",18)
        bacc=top_role("breakfast_accompaniment",18)
        rice=top_role("rice_main",20)
        flat=top_role("flatbread_main",16)
        curry=top_role("curry_side",24)
        veg=top_role("vegetable_side",18)
        snacks=top_role("snack",12)

        if len(bside)<4:
            bside=candidate_base[candidate_base.recipe_name.str.contains("chutney|pachadi|pickle|thogayal|thuvayal",case=False,regex=True,na=False)].head(18).reset_index(drop=True)
        if len(bacc)<4:
            bacc=candidate_base[candidate_base.recipe_name.str.contains("sambar|dal|pappu|stew|kurma|korma",case=False,regex=True,na=False)].head(18).reset_index(drop=True)

        lunch_mains=pd.concat([rice,flat],ignore_index=True).drop_duplicates("_name_key")
        dinner_mains=lunch_mains.copy()
        if len(lunch_mains)<4:
            lunch_mains=bmain.copy()
            dinner_mains=bmain.copy()

        lside_pool=pd.concat([bacc,curry,veg],ignore_index=True).drop_duplicates("_name_key").head(24)
        dside_pool=pd.concat([curry,veg,bacc],ignore_index=True).drop_duplicates("_name_key").head(24)

        if any(x.empty for x in [bmain,bside,bacc,lunch_mains,lside_pool,dinner_mains,dside_pool]):
            return jsonify({"success":False,"error":"The dataset does not contain enough pantry-compatible role-specific recipes for a complete household week."}),400

        def text_of(rec):
            return norm(str(rec.recipe_name)+" "+" ".join(rec.ingredients))

        def breakfast_pair_score(main,side,acc):
            mt,st,at=text_of(main),text_of(side),text_of(acc)
            score=0
            dosa_like=any(x in mt for x in ["dosa","idli","uttapam","vada","pessarattu","pesarattu","adai","ragi"])
            egg_like="egg" in mt
            flat_like=any(x in mt for x in ["paratha","poori","puri","chapati","roti"])
            appam_like=any(x in mt for x in ["appam","puttu"])
            if (dosa_like or egg_like) and any(x in st for x in ["chutney","pachadi","thogayal","thuvayal"]): score+=3
            if (dosa_like or egg_like) and any(x in at for x in ["sambar","dal","pappu"]): score+=3
            if flat_like and any(x in st for x in ["curd","yogurt","raita","pickle","achar"]): score+=3
            if appam_like and any(x in at for x in ["stew","kadala","chana","egg curry"]): score+=3
            return max(1,score)

        def choose(pool,used,bonus=None):
            if pool.empty:return None
            best,best_score=None,-1e9
            for _,rec in pool.iterrows():
                s=float(rec.total)
                if rec._name_key in used:s-=0.25
                if bonus:s+=bonus(rec)
                if s>best_score:best,best_score=rec,s
            return best

        # Compact NSGA-II over meal roles, not raw recipe space.
        role_pools={
            "bmain":bmain,"bside":bside,"bacc":bacc,
            "lmain":lunch_mains,"lside":lside_pool,
            "dmain":dinner_mains,"dside":dside_pool
        }
        roles=list(role_pools.keys())
        random.seed(42);np.random.seed(42)
        P=18;G=6
        slots=[(d,r) for d in range(7) for r in roles]
        population=[[random.randrange(len(role_pools[r])) for _,r in slots] for _ in range(P)]

        def decode(ch):
            days=[];pos=0
            for _ in range(7):
                d={}
                for r in roles:
                    d[r]=role_pools[r].iloc[ch[pos]]
                    pos+=1
                days.append(d)
            return days

        def obj(ch):
            days=decode(ch)
            vals=[d[r] for d in days for r in roles]
            pantry_v=float(np.mean([x.core_pantry for x in vals]))
            region_v=float(np.mean([x.regional for x in vals]))
            expiry_v=float(np.mean([x.expiry for x in vals]))
            learned_v=float(np.mean([x.mlfit for x in vals]))
            total_v=float(np.mean([x.total for x in vals]))
            unique_v=len(set(x._name_key for x in vals))/len(vals)
            mains=[d["bmain"]._name_key for d in days]+[d["lmain"]._name_key for d in days]+[d["dmain"]._name_key for d in days]
            main_unique=len(set(mains))/len(mains)
            return np.array([pantry_v,region_v,expiry_v,learned_v,total_v,unique_v,main_unique])

        def dominates(a,b):return np.all(a>=b) and np.any(a>b)

        def fronts(vals):
            n=len(vals);S=[[] for _ in range(n)];N=[0]*n;fs=[[]]
            for i in range(n):
                for j in range(n):
                    if i==j:continue
                    if dominates(vals[i],vals[j]):S[i].append(j)
                    elif dominates(vals[j],vals[i]):N[i]+=1
                if N[i]==0:fs[0].append(i)
            k=0
            while k<len(fs) and fs[k]:
                nxt=[]
                for i in fs[k]:
                    for j in S[i]:
                        N[j]-=1
                        if N[j]==0:nxt.append(j)
                if nxt:fs.append(nxt)
                k+=1
            return fs

        def crowd(front,vals):
            d={i:0.0 for i in front}
            if len(front)<=2:
                for i in front:d[i]=999
                return d
            for o in range(vals.shape[1]):
                s=sorted(front,key=lambda i:vals[i,o])
                d[s[0]]=d[s[-1]]=999
                lo,hi=vals[s[0],o],vals[s[-1],o]
                if hi==lo:continue
                for k in range(1,len(s)-1):
                    d[s[k]]+=(vals[s[k+1],o]-vals[s[k-1],o])/(hi-lo)
            return d

        for _ in range(G):
            vals=np.array([obj(c) for c in population])
            fs=fronts(vals);rank={};cd={}
            for rr,f in enumerate(fs):
                for i in f:rank[i]=rr
                cd.update(crowd(f,vals))

            def tournament():
                a,b=random.randrange(P),random.randrange(P)
                return population[a][:] if rank[a]<rank[b] or (rank[a]==rank[b] and cd[a]>=cd[b]) else population[b][:]

            offspring=[]
            while len(offspring)<P:
                a,b=tournament(),tournament()
                cut=random.randint(1,len(a)-1)
                for child in (a[:cut]+b[cut:],b[:cut]+a[cut:]):
                    if random.random()<0.15:
                        j=random.randrange(len(child))
                        rr=roles[j%len(roles)]
                        child[j]=random.randrange(len(role_pools[rr]))
                    offspring.append(child)
                    if len(offspring)>=P:break

            combined=population+offspring
            cvals=np.array([obj(c) for c in combined])
            new=[]
            for f in fronts(cvals):
                if len(new)+len(f)<=P:new.extend(f)
                else:
                    cd2=crowd(f,cvals)
                    new.extend(sorted(f,key=lambda i:cd2[i],reverse=True)[:P-len(new)])
                    break
            population=[combined[i] for i in new]

        vals=np.array([obj(c) for c in population])
        balanced=0.34*vals[:,0]+0.20*vals[:,1]+0.13*vals[:,2]+0.10*vals[:,3]+0.13*vals[:,4]+0.05*vals[:,5]+0.05*vals[:,6]
        days=decode(population[int(np.argmax(balanced))])

        used=set()

        def recipe(rec,is_treat=False):
            if rec is None:return None
            return {
                "recipe_id":str(rec.recipe_id),
                "recipe_name":str(rec.recipe_name),
                "suitability_score":round(float(rec.mlfit),3),
                "pantry_score":round(float(rec.core_pantry),3),
                "expiry_score":round(float(rec.expiry),3),
                "region_score":round(float(rec.regional),3),
                "context_score":round(float(max(rec.regional,rec.weatherfit)),3),
                "matched_pantry":list(rec.matched)[:8],
                "ingredients":list(rec.ingredients)[:20],
                "is_treat":bool(is_treat)
            }

        plan=[]
        day_names=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

        for di,day_name in enumerate(day_names):
            d=days[di]
            bmain=d["bmain"]
            bside_best=choose(bside,used,lambda r:0.12*breakfast_pair_score(bmain,r,d["bacc"])) or bside.iloc[0]
            bacc_best=choose(bacc,used,lambda r:0.12*breakfast_pair_score(bmain,bside_best,r)) or bacc.iloc[0]

            lmain=choose(lunch_mains,used) or lunch_mains.iloc[0]
            lside=choose(lside_pool,used) or lside_pool.iloc[0]
            dmain=choose(dinner_mains,used) or dinner_mains.iloc[0]
            dside=choose(dside_pool,used) or dside_pool.iloc[0]

            for rec in [bmain,bside_best,bacc_best,lmain,lside,dmain,dside]:
                used.add(rec._name_key)

            snack=None
            if include_snacks and not snacks.empty:
                snack=choose(snacks,used)
                if snack is not None:used.add(snack._name_key)

            special=day_name==special_day and special_day!="None"
            plan.append({
                "day":day_name,
                "theme":"Regional weekend special + household meal" if special else "Pantry-first Indian household day",
                "is_special_day":special,
                "breakfast":recipe(bmain),
                "breakfast_side_1":recipe(bside_best),
                "breakfast_side_2":recipe(bacc_best),
                "lunch_main":recipe(lmain, special and any(x in norm(lmain.recipe_name) for x in ["biryani","biriyani","pulao","fried rice"])),
                "lunch_side":recipe(lside),
                "dinner_main":recipe(dmain),
                "dinner_side":recipe(dside),
                "snack":recipe(snack) if snack is not None else None
            })

        all_recipes=[d[k] for d in plan for k in ["breakfast","breakfast_side_1","breakfast_side_2","lunch_main","lunch_side","dinner_main","dinner_side","snack"] if d.get(k)]
        pantry_util=float(np.mean([r["pantry_score"] for r in all_recipes])) if all_recipes else 0
        regional_util=float(np.mean([r["region_score"] for r in all_recipes])) if all_recipes else 0
        expiry_util=float(np.mean([r["expiry_score"] for r in all_recipes])) if all_recipes else 0
        context_util=float(np.mean([r["context_score"] for r in all_recipes])) if all_recipes else 0
        unique=len(set(r["recipe_name"].strip().lower() for r in all_recipes))

        return jsonify({
            "success":True,
            "context":ctx,
            "plan":plan,
            "summary":{
                "meals_planned":len(all_recipes),
                "unique_recipes":unique,
                "pantry_utilization":round(pantry_util*100,1),
                "expiry_utilization":round(expiry_util*100,1),
                "nutrition_score":50.0,
                "context_score":round(context_util*100,1),
                "regional_score":round(regional_util*100,1),
                "purchase_burden":round((1-pantry_util)*100,1),
                "algorithm":"Pantry-first role composition + Mealora learned signals + compact NSGA-II",
                "candidate_recipes_evaluated":int(len(recipes)),
                "pantry_grounded_candidates":int(len(candidate_base)),
                "pantry_items_received":int(len(pantry_names)),
                "regional_location":city or "Not resolved"
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success":False,"error":str(e)}),500
# ===== MEALORA HOUSEHOLD WEEKLY PLANNER END =====
'''

source = BACKEND.read_text(encoding="utf-8")
shutil.copy2(BACKEND, BACKUP)
source = re.sub(re.escape(START) + r".*?" + re.escape(END), "", source, flags=re.S)
source = re.sub(r'# ===== MEALORA WEEKLY PLANNER .*?START =====.*?# ===== MEALORA WEEKLY PLANNER .*?END =====', "", source, flags=re.S)
source = re.sub(r'@app\.route\("/api/weekly-plan",\s*methods=\["POST"\]\)\s*\ndef\s+\w+\(\):.*?(?=\n@app\.route|\nif __name__ == "__main__":)', "", source, flags=re.S)

anchor='@app.route("/api/recipe-assistant", methods=["POST"])'
if anchor not in source:
    raise SystemExit("Could not find /api/recipe-assistant insertion point in backend/app.py")
source=source.replace(anchor, ROUTE+"\n\n"+anchor, 1)

if "import re" not in source.split("\n")[:100]:
    source=source.replace("import json\n","import json\nimport re\n",1)

BACKEND.write_text(source,encoding="utf-8")
print("SUCCESS")
print("Patched:",BACKEND)
print("Backup:",BACKUP)
