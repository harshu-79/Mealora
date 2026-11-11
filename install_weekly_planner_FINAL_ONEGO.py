from pathlib import Path
from datetime import datetime
import re, shutil

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend" / "app.py"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP = BACKEND.with_name(f"app.py.before_weekly_final_{STAMP}")

START = "# ===== MEALORA WEEKLY PLANNER FINAL START ====="
END = "# ===== MEALORA WEEKLY PLANNER FINAL END ====="

ROUTE = r"""
# ===== MEALORA WEEKLY PLANNER FINAL START =====
@app.route("/api/weekly-plan", methods=["POST"])
def weekly_plan_final():
    try:
        import ast
        import random
        import numpy as np
        import pandas as pd

        data=request.get_json() or {}
        pantry=data.get("pantry") or []
        household=data.get("household") or {}
        adults=max(int(household.get("adults",0) or 0),0)
        children=max(int(household.get("children",0) or 0),0)
        seniors=max(int(household.get("seniors",0) or 0),0)
        if adults+children+seniors<=0:
            return jsonify({"success":False,"error":"Add at least one household member."}),400

        diet=str(data.get("diet","Mixed") or "Mixed")
        cuisine=str(data.get("cuisine","Auto") or "Auto")
        max_time=float(data.get("max_time",60) or 60)
        include_snacks=bool(data.get("include_snacks",True))
        special_day=str(data.get("special_day","Saturday") or "None")
        city=str(data.get("location_city") or data.get("location") or "").strip()
        lat=data.get("latitude"); lon=data.get("longitude")

        ctx=build_context({
            "mode":"custom",
            "date":data.get("date"),
            "time":data.get("time"),
            "location_city":city,
            "location":city,
            "latitude":lat,
            "longitude":lon,
            "meal_type":"Automatic",
            "diet":diet,
            "cuisine":cuisine,
            "max_time":max_time
        })

        aliases={"chilli":"chili","capsicum":"bell pepper","coriander leaves":"coriander",
                 "coriander leaf":"coriander","curd":"yogurt","dahi":"yogurt",
                 "brinjal":"eggplant","lady s finger":"okra","ladys finger":"okra",
                 "bhindi":"okra","pappu":"dal","atta":"wheat flour",
                 "semolina":"rava","sooji":"rava"}
        stop={"the","and","with","for","from","into","cup","cups","tsp","tbsp",
              "tablespoon","tablespoons","teaspoon","teaspoons","piece","pieces",
              "small","large","medium","fresh","finely","roughly","chopped",
              "sliced","diced","minced","crushed","to","taste","as","needed"}

        def norm(v):
            s=str(v or "").lower().replace("&"," and ")
            s=re.sub(r"[^a-z0-9\s]"," ",s); s=re.sub(r"\s+"," ",s).strip()
            for a,b in sorted(aliases.items(),key=lambda x:-len(x[0])):
                s=re.sub(rf"\b{re.escape(a)}\b",b,s)
            return s
        def toks(v): return {x for x in norm(v).split() if len(x)>=3 and x not in stop}
        def clean(v):
            s=str(v or "").strip()
            if not s or s.lower() in ("nan","none","null"): return ""
            s=re.sub(r"^\s*\d+(?:\.\d+)?\s*(?:kg|g|ml|l|litre|liter|cup|cups|tbsp|tsp|piece|pieces|clove|cloves)?\s*(?:of\s+)?","",s,flags=re.I)
            s=re.sub(r"\([^)]*\)"," ",s)
            s=re.sub(r"\b(chopped|sliced|diced|minced|grated|crushed|finely|roughly|freshly|to taste|as needed)\b"," ",s,flags=re.I)
            return re.sub(r"\s+"," ",s).strip(" ,.-")
        def parse(v):
            if isinstance(v,list): return [clean(x) for x in v if clean(x)]
            s=str(v or "").strip()
            if not s or s.lower()=="nan": return []
            if s.startswith("[") and s.endswith("]"):
                for p in (json.loads,ast.literal_eval):
                    try:
                        q=p(s)
                        if isinstance(q,list): return [clean(x) for x in q if clean(x)]
                    except: pass
            parts=re.split(r"\n|;|\|",s)
            if len(parts)==1: parts=[x.strip() for x in s.split(",")]
            return [clean(x) for x in parts if clean(x)]

        pantry_names=[]; expiry_names=[]; today=now_india().date()
        for item in pantry:
            n=item.get("normalized_name") or item.get("ingredient_name") or item.get("name")
            if not n: continue
            n=clean(n)
            if n: pantry_names.append(n)
            ex=item.get("expiry_date")
            if ex:
                try:
                    if (datetime.strptime(str(ex)[:10],"%Y-%m-%d").date()-today).days<=7:
                        expiry_names.append(n)
                except: pass
        pantry_names=list(dict.fromkeys(pantry_names)); expiry_names=list(dict.fromkeys(expiry_names))
        if not pantry_names:
            return jsonify({"success":False,"error":"Your pantry is empty. Add ingredients first."}),400

        def match(ing):
            a=norm(ing); at=toks(a)
            if not at: return None
            for p in pantry_names:
                b=norm(p); bt=toks(b)
                if a==b or a in b or b in a or (at&bt and len(at&bt)>=1): return p
            best=None;score=0
            for p in pantry_names:
                bt=toks(p); z=len(at&bt)/max(1,len(at|bt))
                if z>score:score=z;best=p
            return best if score>=.5 else None

        staples={"salt","sugar","water","oil","sunflower oil","groundnut oil","coconut oil",
                 "mustard oil","sesame oil","olive oil","ghee","butter","black pepper",
                 "turmeric powder","red chili powder","coriander powder","cumin powder","garam masala","hing"}

        def pscore(ings):
            core=[x for x in ings if norm(x) not in staples]
            matched=[match(x) for x in ings if match(x)]
            core_m=sum(1 for x in core if match(x))
            core_cov=core_m/max(1,len(core))
            all_cov=len(matched)/max(1,len(ings))
            exp=sum(1 for x in expiry_names if match(x))
            exp_use=0 if not expiry_names else min(1,exp/max(1,len(expiry_names)))
            return .75*core_cov+.25*all_cov,core_cov,exp_use,list(dict.fromkeys(matched))

        if not hasattr(weekly_plan_final,"_df"):
            weekly_plan_final._df=pd.read_csv(RECIPE_DATASET,low_memory=False)
        df=weekly_plan_final._df

        def col(names):
            low={str(c).lower():c for c in df.columns}
            for n in names:
                if n.lower() in low:return low[n.lower()]
            return None

        namec=col(["recipe_name","name","title","dish_name"])
        idc=col(["recipe_id","id","recipeid"])
        ingc=col(["ingredients","ingredient_names","ingredient_list","cleaned_ingredients","Cleaned-Ingredients","Ingredients"])
        mealc=col(["meal_type","meal","mealtype"])
        cuisinec=col(["cuisine","cuisine_type"])
        timec=col(["cooking_time_minutes","cooking_time","cook_time","total_time_minutes","total_time"])
        dietc=col(["diet","diet_type","diet_category"])
        regionc=[c for c in [col(["state","state_name"]),col(["region","region_name"]),cuisinec,col(["origin","origin_state","place"]),namec] if c]
        if not namec or not ingc: raise RuntimeError("Recipe dataset is missing recipe_name or ingredients.")

        citynorm=norm(ctx.get("location_city") or city)
        regionmap={
          "hyderabad":{"hyderabad","hyderabadi","telangana","telugu","andhra","deccan"},
          "secunderabad":{"hyderabad","hyderabadi","telangana","telugu","andhra","deccan"},
          "vijayawada":{"andhra","andhra pradesh","telugu","coastal andhra","vijayawada"},
          "visakhapatnam":{"andhra","andhra pradesh","telugu","vizag","coastal andhra"},
          "chennai":{"tamil","tamil nadu","chennai","chettinad"},
          "bengaluru":{"karnataka","kannada","udupi","mangalorean","bengaluru"},
          "kochi":{"kerala","malayalam","malabar","kochi"},
          "mumbai":{"maharashtra","marathi","konkani","mumbai"},
          "kolkata":{"bengal","bengali","west bengal","kolkata"},
          "delhi":{"delhi","punjabi","mughlai","north indian"},
          "ahmedabad":{"gujarat","gujarati","ahmedabad"}
        }
        terms=regionmap.get(citynorm,set())
        if not terms and "south" in str(ctx.get("cuisine","")).lower(): terms={"south indian"}

        def regional(row):
            if not terms:return 0
            txt=" ".join(str(row.get(c,"")) for c in regionc).lower()
            return 1.0 if any(t in txt for t in terms) else 0.0

        def role(name,ings,raw=""):
            raw=str(raw or "").lower(); txt=(str(name)+" "+" ".join(ings)).lower()
            if any(k in raw for k in ["breakfast","morning"]):return "breakfast_main"
            if any(k in raw for k in ["snack","starter","appetizer"]):return "snack"
            if any(k in txt for k in ["chutney","pachadi","thogayal","thuvayal","pickle"]):return "breakfast_side"
            if any(k in txt for k in ["sambar","sambhar","dal","pappu","kurma","korma"]):return "accompaniment"
            if any(k in txt for k in ["dosa","idli","upma","poha","pongal","pessarattu","uttapam","vada","puri","paratha","omelette","omelet","appam","puttu","toast"]):return "breakfast_main"
            if any(k in txt for k in ["rice","biryani","pulao","fried rice","khichdi"]):return "rice_main"
            if any(k in txt for k in ["chapati","roti","naan","paratha"]):return "flatbread_main"
            if any(k in txt for k in ["curry","poriyal","kootu","sabzi","masala","fish","chicken","mutton","prawn","prawns","egg","paneer","rasam"]):return "curry_side"
            if any(k in txt for k in ["tea","coffee","pakora","pakoda","bajji","bonda","samosa","sandwich","cutlet","murukku","mixture"]):return "snack"
            return "other"

        rows=[]
        for i,row in df.iterrows():
            name=str(row[namec]).strip()
            if not name or name.lower()=="nan":continue
            ings=parse(row[ingc])
            if not ings:continue
            ps,core,ex,matched=pscore(ings)
            if core<.25 and ps<.35:continue
            rg=regional(row)
            cuis=1.0 if cuisine.lower() not in ("auto","automatic","any","mixed","none") and cuisine.lower() in " ".join(str(row.get(c,"")) for c in [cuisinec] if c).lower() else 0.0
            ct=None
            if timec:
                try:ct=float(row[timec])
                except:ct=None
            tf=1 if ct is None else float(np.clip(1-max(ct-max_time,0)/max(max_time,1),0,1))
            ml=.5
            for c in ["hgb_suitability_score","rf_suitability_score","ml_suitability_score","han_score"]:
                if c in row.index:
                    try:ml=max(ml,float(np.clip(float(row[c]),0,1)))
                    except:pass
            total=.38*core+.15*ps+.12*ex+.16*rg+.06*cuis+.06*tf+.07*ml
            rows.append({"recipe_id":str(row[idc]) if idc else str(i),"recipe_name":name,
                         "ingredients":ings,"role":role(name,ings,row[mealc] if mealc else ""),
                         "pantry":ps,"core_pantry":core,"expiry":ex,"regional":rg,"cuisine_fit":cuis,
                         "time_fit":tf,"ml":ml,"total":total,"matched":matched,
                         "missing":[x for x in ings if not match(x)]})
        r=pd.DataFrame(rows)
        if r.empty:return jsonify({"success":False,"error":"No pantry-compatible recipes found."}),400
        r["_key"]=r.recipe_name.map(lambda x:re.sub(r"[^a-z0-9]+"," ",str(x).lower()).strip())
        r=r.sort_values(["total","core_pantry","regional"],ascending=False).drop_duplicates("_key").reset_index(drop=True)

        regional_pool=r[r.regional>=1].copy()
        base=regional_pool if len(regional_pool)>=12 else r

        def pool(role_name,limit=24):
            q=base[base.role==role_name].sort_values(["total","core_pantry","expiry"],ascending=False).head(limit)
            if len(q)<4:q=base.sort_values(["total","core_pantry","expiry"],ascending=False).head(limit)
            return q.reset_index(drop=True)

        bm=pool("breakfast_main"); bs=pool("breakfast_side"); ba=pool("accompaniment")
        lm=pool("rice_main")
        if len(lm)<4:lm=pool("flatbread_main")
        ls=pool("curry_side"); dm=pool("curry_side")
        if len(dm)<4:dm=pool("rice_main")
        ds=pool("curry_side"); sn=pool("snack")

        # Keep only a small candidate table for optimization.
        qs=[bm,bs,ba,lm,ls,dm,ds,sn]
        cand=pd.concat([q for q in qs if not q.empty],ignore_index=True).drop_duplicates("_key").reset_index(drop=True)
        cand=cand.sort_values(["total","core_pantry","regional"],ascending=False).head(100).reset_index(drop=True)

        def ci(q):
            return [int(cand.index[cand._key==k][0]) for k in q._key if not cand.index[cand._key==k].empty][:18]
        role_names=["bm","bs","ba","lm","ls","dm","ds"]
        domains=[ci(bm),ci(bs),ci(ba),ci(lm),ci(ls),ci(dm),ci(ds)]
        if any(not x for x in domains):return jsonify({"success":False,"error":"Not enough meal-compatible pantry recipes."}),400
        domains7=domains*7

        def fit(ch):
            rr=[cand.iloc[int(i)] for i in ch]
            pantry=float(np.mean([x.core_pantry for x in rr])); reg=float(np.mean([x.regional for x in rr]))
            exp=float(np.mean([x.expiry for x in rr])); ml=float(np.mean([x.ml for x in rr]))
            unique=len(set(x._key for x in rr))/len(rr)
            return np.array([pantry,reg,exp,ml,unique])
        def dom(a,b):return np.all(a>=b) and np.any(a>b)
        def fronts(v):
            n=len(v); ds=[[] for _ in range(n)];cnt=[0]*n;f=[[]]
            for i in range(n):
                for j in range(n):
                    if i==j:continue
                    if dom(v[i],v[j]):ds[i].append(j)
                    elif dom(v[j],v[i]):cnt[i]+=1
                if cnt[i]==0:f[0].append(i)
            k=0
            while k<len(f) and f[k]:
                nxt=[]
                for i in f[k]:
                    for j in ds[i]:
                        cnt[j]-=1
                        if cnt[j]==0:nxt.append(j)
                if nxt:f.append(nxt)
                k+=1
            return f
        def crowd(fr,v):
            d={i:0 for i in fr}
            if len(fr)<=2:
                for i in fr:d[i]=999
                return d
            for o in range(v.shape[1]):
                s=sorted(fr,key=lambda i:v[i,o]);d[s[0]]=d[s[-1]]=999
                lo,hi=v[s[0],o],v[s[-1],o]
                if hi==lo:continue
                for k in range(1,len(s)-1):d[s[k]]+=(v[s[k+1],o]-v[s[k-1],o])/(hi-lo)
            return d

        random.seed(42);np.random.seed(42)
        P=20;G=8
        pop=[[random.choice(domains7[j]) for j in range(len(domains7))] for _ in range(P)]
        for _ in range(G):
            val=np.array([fit(x) for x in pop]);fs=fronts(val);rank=np.zeros(P,int);cd=np.zeros(P)
            for rr,fr in enumerate(fs):
                for i in fr:rank[i]=rr
                for i,z in crowd(fr,val).items():cd[i]=z
            off=[]
            while len(off)<P:
                def pick():
                    a,b=random.randrange(P),random.randrange(P)
                    return pop[a][:] if rank[a]<rank[b] or (rank[a]==rank[b] and cd[a]>cd[b]) else pop[b][:]
                a,b=pick(),pick();cut=random.randint(1,len(a)-1);off += [a[:cut]+b[cut:],b[:cut]+a[cut:]]
            comb=pop+off[:P];cv=np.array([fit(x) for x in comb]);fs=fronts(cv);new=[]
            for fr in fs:
                if len(new)+len(fr)<=P:new += [comb[i] for i in fr]
                else:
                    cc=crowd(fr,cv);new += [comb[i] for i in sorted(fr,key=lambda i:cc[i],reverse=True)[:P-len(new)]];break
            pop=new
        val=np.array([fit(x) for x in pop]);best=max(range(len(pop)),key=lambda i:np.dot(val[i],[.42,.20,.15,.08,.15]))
        selected=pop[best]

        used=set()
        def recipe(idx):
            x=cand.iloc[int(idx)]
            return {"recipe_id":str(x.recipe_id),"recipe_name":str(x.recipe_name),
                    "suitability_score":round(float(x.ml),3),"pantry_score":round(float(x.core_pantry),3),
                    "expiry_score":round(float(x.expiry),3),"context_score":round(float(max(x.regional,x.cuisine_fit)),3),
                    "regional_score":round(float(x.regional),3),"nutrition_score":.5,
                    "purchase_score":round(1-min(len(x.missing)/max(1,len(x.ingredients)),1),3),
                    "matched_pantry":list(x.matched)[:12],"missing_ingredients":list(x.missing)[:8],
                    "ingredients":list(x.ingredients)}
        def choose(domain,offset):
            preferred=cand.iloc[int(selected[offset])]
            if preferred._key not in used:
                used.add(preferred._key);return recipe(int(selected[offset]))
            for i in domain:
                if cand.iloc[i]._key not in used:
                    used.add(cand.iloc[i]._key);return recipe(i)
            used.add(preferred._key);return recipe(int(selected[offset]))

        days=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];plan=[]
        for d,day in enumerate(days):
            o=d*7
            b=choose(domains[0],o); side=choose(domains[1],o+1); acc=choose(domains[2],o+2)
            lmain=choose(domains[3],o+3); lside=choose(domains[4],o+4)
            dmain=choose(domains[5],o+5); dside=choose(domains[6],o+6)
            snack=None
            if include_snacks and not sn.empty:
                for k in ci(sn):
                    if cand.iloc[k]._key not in used:
                        used.add(cand.iloc[k]._key);snack=recipe(k);break
            plan.append({"day":day,"theme":"Weekend special / higher-variety household meal" if day==special_day and special_day!="None" else "Pantry-first Indian household day",
                         "is_special_day":day==special_day and special_day!="None",
                         "breakfast":{"main":b,"side":side,"accompaniment":acc},
                         "lunch":{"main":lmain,"side":lside},"dinner":{"main":dmain,"side":dside},"snack":snack})

        allm=[]
        for d in plan:
            allm += [d["breakfast"]["main"],d["breakfast"]["side"],d["breakfast"]["accompaniment"],d["lunch"]["main"],d["lunch"]["side"],d["dinner"]["main"],d["dinner"]["side"]]
            if d["snack"]:allm.append(d["snack"])
        return jsonify({"success":True,"context":ctx,"plan":plan,
          "summary":{"meals_planned":len(allm),"unique_recipes":len(set(x["recipe_id"] for x in allm)),
          "pantry_utilization":round(float(np.mean([x["pantry_score"] for x in allm]))*100,1),
          "expiry_utilization":round(float(np.mean([x["expiry_score"] for x in allm]))*100,1),
          "nutrition_score":50.0,"context_score":round(float(np.mean([x["context_score"] for x in allm]))*100,1),
          "regional_fit":round(float(np.mean([x["regional_score"] for x in allm]))*100,1),
          "purchase_burden":round((1-float(np.mean([x["pantry_score"] for x in allm])))*100,1),
          "algorithm":"Pantry-first candidate generation + RF/HAN signals + compact NSGA-II",
          "candidate_recipes_evaluated":len(r),"pantry_grounded_candidates":len(base),"pantry_items_received":len(pantry_names),
          "regional_location":ctx.get("location_city")}})
    except Exception as e:
        import traceback;traceback.print_exc()
        return jsonify({"success":False,"error":str(e)}),500
# ===== MEALORA WEEKLY PLANNER FINAL END =====
"""

source=BACKEND.read_text(encoding="utf-8")
shutil.copy2(BACKEND,BACKUP)
source=re.sub(re.escape(START)+r".*?"+re.escape(END),"",source,flags=re.S)
source=re.sub(r"# ===== MEALORA WEEKLY PLANNER V[23] START =====.*?# ===== MEALORA WEEKLY PLANNER V[23] END =====","",source,flags=re.S)
source=re.sub(r'@app\.route\("/api/weekly-plan".*?(?=@app\.route\(|if __name__ == "__main__":)',"",source,flags=re.S)
anchor='@app.route("/api/recipe-assistant", methods=["POST"])'
if anchor not in source: anchor='if __name__ == "__main__":'
if anchor not in source: raise SystemExit("Could not find insertion point in backend/app.py.")
source=source.replace(anchor,ROUTE+"\n\n"+anchor,1)
if "\nimport re\n" not in source: source=source.replace("import json\n","import json\nimport re\n",1)
BACKEND.write_text(source,encoding="utf-8")
print("Installed FINAL weekly planner backend.")
print("Backend:",BACKEND)
print("Backup:",BACKUP)
