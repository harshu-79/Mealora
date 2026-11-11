#!/usr/bin/env python3
import math
from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parent
D=BASE/"datasets"/"processed"
INPUT=D/"recipe_master_dataset_nutrition_IFCT_V5.csv"
OUTPUT=D/"recipe_master_dataset_nutrition_CLEAN.csv"
AUDIT=D/"Mealora_nutrition_CLEAN_audit.json"
N=["calories","carbohydrates","protein","fat","fibre","sodium","calcium","iron","vitamin_c","folate"]

def num(x):
    try:
        v=float(x); return v if math.isfinite(v) else None
    except: return None

def check(r):
    v={c:num(r.get(c)) for c in N}
    if not any(x is not None for x in v.values()): return "unavailable",False,"no_nutrition"
    if any(x<0 for x in v.values() if x is not None): return "invalid",False,"negative_value"
    cal,p,c,f=v["calories"],v["protein"],v["carbohydrates"],v["fat"]
    if all(x is not None for x in (cal,p,c,f)):
        mk=4*p+4*c+9*f
        if mk>1 and cal>0:
            ratio=cal/mk
            if ratio<0.50 or ratio>1.80: return "suspicious",False,f"calorie_macro_mismatch:{ratio:.2f}"
    if v["sodium"] is not None and v["sodium"]>10000:
        return "suspicious",False,"very_high_sodium"
    if all(v[c] is not None for c in N): return "verified",True,"complete_and_consistent"
    return "partial",False,"incomplete_nutrition"

def score(r):
    if r["nutrition_quality"]!="verified": return pd.NA
    p=min(max(float(r["protein"])/25,0),1)
    fi=min(max(float(r["fibre"])/8,0),1)
    c=min(max(float(r["carbohydrates"])/60,0),1)
    fat=min(max(float(r["fat"])/25,0),1)
    s=min(max(float(r["sodium"])/900,0),1)
    return round(max(0,min(1,.35*p+.25*fi+.20*c+.20*(1-.5*fat-.5*s))),3)

def main():
    if not INPUT.exists(): raise SystemExit(f"Missing input: {INPUT}")
    df=pd.read_csv(INPUT,low_memory=False)
    if len(df)!=18568: raise SystemExit(f"Unexpected row count: {len(df)}")
    if not df["recipe_id"].is_unique: raise SystemExit("recipe_id is not unique")
    x=df.apply(check,axis=1,result_type="expand")
    x.columns=["nutrition_quality","nutrition_valid","nutrition_validation_reason"]
    df=x.join(df)
    df["nutrition_match_score"]=df.apply(score,axis=1)
    df.to_csv(OUTPUT,index=False)
    audit={
      "input":str(INPUT),"output":str(OUTPUT),"rows":len(df),
      "verified_complete":int((df.nutrition_quality=="verified").sum()),
      "suspicious_excluded_from_ml":int((df.nutrition_quality=="suspicious").sum()),
      "invalid_excluded_from_ml":int((df.nutrition_quality=="invalid").sum()),
      "partial_excluded_from_ml":int((df.nutrition_quality=="partial").sum()),
      "unavailable":int((df.nutrition_quality=="unavailable").sum()),
      "valid_for_ml":int(df.nutrition_valid.sum())
    }
    AUDIT.write_text(__import__("json").dumps(audit,indent=2))
    print(__import__("json").dumps(audit,indent=2)); print("OUTPUT:",OUTPUT)

if __name__=="__main__": main()
