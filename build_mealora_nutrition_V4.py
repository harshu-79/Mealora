#!/usr/bin/env python3
"""Mealora nutrition enrichment.

Keeps the original dataset untouched. Downloads an IFCT-derived JSON dataset,
parses recipe ingredient quantities, matches ingredients to IFCT foods, computes
recipe nutrition, and writes a NEW dataset plus validation/audit files.

Source basis: ICMR-NIN Indian Food Composition Tables (IFCT) 2017.
"""
from __future__ import annotations
import ast, json, math, os, re, shutil, sys, urllib.request
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
# Mealora's master dataset lives in datasets/processed. DO NOT move or overwrite it.
DATA_DIR = BASE / 'datasets' / 'processed'
INPUT = DATA_DIR / 'recipe_master_dataset.csv'
OUTPUT = DATA_DIR / 'recipe_master_dataset_nutrition_IFCT.csv'
AUDIT = DATA_DIR / 'Mealora_nutrition_IFCT_audit.json'
IFCT_JSON = DATA_DIR / 'IndianFoodCompositionTable_In_JSON_Format.json'
IFCT_URL = 'https://raw.githubusercontent.com/nithyamani/IndianFoodComposition/main/IndianFoodCompositionTable_In_JSON_Format.json'
NUTRIENTS = ['calories','carbohydrates','protein','fat','fibre','sodium','calcium','iron','vitamin_c','folate']

# Conservative household-weight conversions. Used only when a recipe explicitly
# gives a household unit; values are assumptions and are recorded as estimates.
UNIT_G = {
    'kg':1000,'kilogram':1000,'kilograms':1000,
    'g':1,'gram':1,'grams':1,
    'mg':0.001,
    'lb':453.592,'lbs':453.592,'pound':453.592,
    'oz':28.3495,'ounce':28.3495,
    'cup':240,'cups':240,
    'tbsp':15,'tablespoon':15,'tablespoons':15,'tbs':15,
    'tsp':5,'teaspoon':5,'teaspoons':5,
    'ml':1,'milliliter':1,'milliliters':1,'millilitre':1,'millilitres':1,
    'litre':1000,'liter':1000,'litres':1000,'liters':1000,
    'sprig':3,'sprigs':3,
    'clove':3,'cloves':3,
    'inch':8,'inches':8,
    'slice':25,'slices':25,
    'piece':50,'pieces':50,
    'small':70,'medium':100,'large':150,
}

STOP = set('and with for the a an of or to in on as from fresh finely chopped cut sliced diced grated boiled cooked soaked optional taste salt water recipe style'.split())

def norm(s):
    s = str(s or '').lower().replace('&amp;','and').replace('&',' and ')
    s = re.sub(r'\([^)]*\)', ' ', s)
    s = re.sub(r'[^a-z0-9\u0900-\u097f\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def frac(s):
    s=s.strip()
    # mixed number: 1 1/2, 2 3/4, etc.
    m=re.fullmatch(r'(\d+(?:\.\d+)?)\s+(\d+)\s*/\s*(\d+)', s)
    if m:
        return float(m.group(1)) + float(m.group(2))/float(m.group(3))
    # simple fraction
    m=re.fullmatch(r'(\d+)\s*/\s*(\d+)', s)
    if m:
        den=float(m.group(2))
        return float(m.group(1))/den if den else None
    try:return float(s)
    except:return None

def parse_amount(text):
    """Return grams, confidence, or None. Handles the common formats in Mealora."""
    s=str(text).strip().lower().replace('–','-').replace('—','-')
    # ranges: use midpoint
    m=re.search(r'(\d+(?:\s+\d+/\d+)?|\d+/\d+)\s*(?:-|to)\s*(\d+(?:\s+\d+/\d+)?|\d+/\d+)\s*(kg|g|grams?|mg|ml|l|liters?|litres?|cups?|tbsp|tablespoons?|tsp|teaspoons?|cloves?|sprigs?|inches?|pieces?)',s)
    if m:
        a,b=frac(m.group(1)),frac(m.group(2)); u=m.group(3)
        if a is not None and b is not None:
            return ((a+b)/2)*unit_factor(u), 0.85
    # mixed number + unit
    m=re.search(r'(\d+(?:\s+\d+/\d+)?|\d+/\d+)\s*(kg|g|grams?|mg|ml|l|liters?|litres?|cups?|tbsp|tablespoons?|tsp|teaspoons?|cloves?|sprigs?|inches?|pieces?)\b',s)
    if m:
        n=frac(m.group(1));
        if n is not None:return n*unit_factor(m.group(2)),0.95
    # number followed by common item-count words
    m=re.search(r'\b(\d+(?:\.\d+)?)\s*(?:small|medium|large)?\s*(?:pieces?|nos?\.?|items?)\b',s)
    if m:return float(m.group(1))*50,0.60
    # bare count: 1 apple, 2 onions, 4 cloves garlic, etc.
    m=re.match(r'^\s*(\d+(?:\.\d+)?)\s+',s)
    if m:
        n=float(m.group(1));
        # ingredient-specific piece weights
        if re.search(r'\b(egg|eggs)\b',s): return n*50,0.90
        if re.search(r'\b(onion|onions|tomato|tomatoes|apple|banana|mango|potato|potatoes|lemon|lime|guava|carrot|carrots|beetroot|cucumber|brinjal|eggplant|capsicum|pepper)\b',s): return n*100,0.70
        if re.search(r'\b(clove|cloves)\b',s): return n*3,0.90
        if re.search(r'\b(chicken leg|legs)\b',s): return n*120,0.75
        if re.search(r'\b(sprig|sprigs)\b',s): return n*3,0.80
    return None,0.0

def unit_factor(u):
    u=u.lower().rstrip('.')
    if u in UNIT_G:return UNIT_G[u]
    if u.startswith('cup'):return 240
    if u.startswith('tbsp') or u.startswith('tablespoon'):return 15
    if u.startswith('tsp') or u.startswith('teaspoon'):return 5
    if u.startswith('clove'):return 3
    if u.startswith('sprig'):return 3
    if u.startswith('inch'):return 8
    if u.startswith('piece'):return 50
    if u in ('l','liter','liters','litre','litres'):return 1000
    return 1

def ingredient_texts(cell):
    if pd.isna(cell):return []
    s=str(cell).strip()
    try:
        x=ast.literal_eval(s)
        if isinstance(x,list): return [str(v) for v in x]
    except Exception: pass
    # Some rows are comma-separated ingredient strings.
    return [p.strip() for p in re.split(r'\s*,\s*',s) if p.strip()]

def load_ifct():
    if not IFCT_JSON.exists():
        print('Downloading IFCT-derived JSON...')
        urllib.request.urlretrieve(IFCT_URL, IFCT_JSON)
    with open(IFCT_JSON,'r',encoding='utf-8') as f:data=json.load(f)
    records=[]
    def walk(x):
        if isinstance(x,dict):
            # Accept multiple known naming conventions.
            keys={norm(k):k for k in x.keys()}
            name=None
            for k in ('food_name','food name','name','foodname','item','food'):
                if norm(k) in keys and isinstance(x[keys[norm(k)]],str): name=x[keys[norm(k)]];break
            if name:
                records.append(x)
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(data)
    # de-duplicate by normalized name
    uniq={}
    for r in records:
        name=next((r[k] for k in r if norm(k) in ('food_name','food name','name','foodname','item','food') and isinstance(r[k],str)),None)
        if name:uniq[norm(name)]=r
    return list(uniq.values())

def numeric_value(v):
    if v is None or (isinstance(v,float) and not math.isfinite(v)): return None
    if isinstance(v,(int,float)): return float(v)
    s=str(v).strip()
    if not s or s.lower() in {'na','n','tr','trace','-','—','nan','none'}: return None
    # Values may be written as 518±33, 518 +/- 33, <0.1, etc.
    m=re.search(r'[-+]?\d+(?:\.\d+)?', s.replace(',',''))
    return float(m.group(0)) if m else None

def flatten_scalars(obj, path=''):
    """Flatten nested JSON into (key path, scalar value) pairs."""
    out=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            kp=f'{path} {k}'.strip()
            if isinstance(v,(dict,list)):
                out.extend(flatten_scalars(v,kp))
            else:
                out.append((kp,v))
    elif isinstance(obj,list):
        for i,v in enumerate(obj):
            out.extend(flatten_scalars(v,f'{path} {i}'.strip()))
    else:
        out.append((path,obj))
    return out

def _key_tokens(key):
    return set(norm(key).split())

def _find_scalar(fields, include_terms, exclude_terms=()):
    """Find the first numeric scalar whose flattened key contains the terms."""
    candidates=[]
    for key,val in fields:
        nk=norm(key)
        if any(ex in nk for ex in exclude_terms):
            continue
        if all(term in nk for term in include_terms):
            num=numeric_value(val)
            if num is not None:
                # Prefer the shortest matching key (usually the actual nutrient field).
                candidates.append((len(nk),key,num))
    if candidates:
        candidates.sort(key=lambda x:x[0])
        return candidates[0][2]
    return None

def extract_profile(r):
    """Extract IFCT values from the actual nested IFCT JSON structure.

    Each record has:
      nutrients -> category -> composition -> {name, code, value}
    The old parser flattened this structure and lost the relationship between
    a nutrient name/code and its value. This parser reads the composition rows
    directly.
    """
    out={n:None for n in NUTRIENTS}
    nutrients=r.get('nutrients',[]) if isinstance(r,dict) else []
    if not isinstance(nutrients,list):
        return out

    # IFCT codes observed in the published JSON. Names are used as fallback.
    code_map={
        'enerc':'calories',
        'choavldf':'carbohydrates',
        'protcnt':'protein',
        'fatce':'fat',
        'fibtg':'fibre',
        'na':'sodium',
        'sodium':'sodium',
        'ca':'calcium',
        'calcium':'calcium',
        'fe':'iron',
        'iron':'iron',
        'vitc':'vitamin_c',
        'folsum':'folate',
    }
    for category in nutrients:
        if not isinstance(category,dict):
            continue
        comps=category.get('composition',[])
        if not isinstance(comps,list):
            continue
        for item in comps:
            if not isinstance(item,dict):
                continue
            code=norm(item.get('code','')).replace(' ','')
            name=norm(item.get('name',''))
            value=numeric_value(item.get('value'))
            if value is None:
                continue
            target=code_map.get(code)
            if target is None:
                if 'energy' in name and 'kj' in name: target='calories'
                elif 'carbohydrate' in name: target='carbohydrates'
                elif 'protein' in name: target='protein'
                elif 'total fat' in name: target='fat'
                elif 'dietary fibre total' in name or 'dietary fiber total' in name: target='fibre'
                elif name.startswith('sodium'): target='sodium'
                elif name.startswith('calcium'): target='calcium'
                elif name.startswith('iron'): target='iron'
                elif 'ascorbic acid' in name: target='vitamin_c'
                elif 'total folates' in name or 'folate' in name: target='folate'
            if target is None:
                continue
            if target=='calories' and 'kj' in name:
                value=value/4.184
            # Keep the first valid value for each nutrient.
            if out[target] is None:
                out[target]=value
    return out

def name_from(r):
    # Support both flat records and nested structures.
    for k,v in flatten_scalars(r):
        nk=norm(k)
        if isinstance(v,str) and (
            nk.endswith('food name') or nk.endswith('foodname') or
            nk.endswith('food name en') or nk.endswith('name') or
            nk.endswith('item') or nk.endswith('food')
        ):
            return v
    # Fallback: any short string field whose key contains name/food.
    for k,v in flatten_scalars(r):
        if isinstance(v,str) and ('name' in norm(k) or 'food' in norm(k)) and len(v.strip())<200:
            return v
    return ''

def build_index(records):
    idx=[]
    inverted={}
    for r in records:
        name=name_from(r); prof=extract_profile(r)
        if not name or prof['calories'] is None:
            continue
        nn=norm(name)
        tokens=set(t for t in nn.split() if len(t)>2 and t not in STOP)
        entry=(name,nn,tokens,prof)
        pos=len(idx); idx.append(entry)
        for t in tokens:
            inverted.setdefault(t,set()).add(pos)
    return idx,inverted

def match_ingredient(ing, idx, inverted):
    q=norm(ing)
    q=re.sub(r'^\s*\d+(?:\s+\d+/\d+|\s*[-–—]\s*\d+)?\s*(?:kg|g|grams?|mg|ml|l|liters?|litres?|cups?|tbsp|tablespoons?|tsp|teaspoons?|cloves?|sprigs?|inches?|pieces?)?\s*',' ',q)
    q=re.sub(r'\b(chopped|finely|roughly|thinly|sliced|diced|grated|boiled|cooked|soaked|roasted|fried|optional|as required|to taste|for cooking|for dusting)\b',' ',q)
    q=re.sub(r'\s+',' ',q).strip()
    qt=set(t for t in q.split() if len(t)>2 and t not in STOP)
    candidate_ids=set()
    for t in qt:
        candidate_ids.update(inverted.get(t,()))
    # If token lookup finds nothing, use all profiles as a conservative fallback.
    candidates=candidate_ids if candidate_ids else range(len(idx))
    best=None; bs=0
    for j in candidates:
        name,nn,toks,prof=idx[j]
        inter=len(qt&toks); union=len(qt|toks) or 1
        jacc=inter/union
        seq=SequenceMatcher(None,q,nn).ratio()
        score=0.72*jacc+0.28*seq
        if name.lower() in q or q in nn: score+=0.12
        if score>bs: bs=score; best=(name,prof)
    if best and bs>=0.52:return best[0],best[1],bs
    return None,None,bs

def main():
    if not INPUT.exists():raise SystemExit(f'Missing {INPUT}')
    df=pd.read_csv(INPUT,low_memory=False)
    original_cols=list(df.columns)
    # Preserve existing values. New metadata only.
    for c in ['nutrition_source','nutrition_confidence','nutrition_servings_assumption','nutrition_matched_ingredients','nutrition_unmatched_ingredients']:
        if c not in df.columns:df[c]=''
    if 'nutrition_source' not in df.columns:df['nutrition_source']=''
    records=load_ifct(); idx,inverted=build_index(records)
    print(f'IFCT records loaded: {len(records)}; usable profiles: {len(idx)}')
    if len(idx)==0:
        print('DIAGNOSTIC: first IFCT record keys:', list(records[0].keys())[:30] if records else 'NONE')
        if records:
            print('DIAGNOSTIC: flattened scalar fields:')
            for k,v in flatten_scalars(records[0])[:80]: print(' ',k,'=',v)
        raise SystemExit('STOP: IFCT file was loaded but no usable nutrient profiles were detected. No output CSV was written.')
    filled=0; partial=0; unmatched_total=0; qty_total=0; match_total=0
    details=[]
    for i,row in df.iterrows():
        existing={c:row[c] for c in NUTRIENTS}
        if all(pd.notna(existing[c]) for c in NUTRIENTS):
            df.at[i,'nutrition_source']='existing_complete'
            df.at[i,'nutrition_confidence']='high'
            continue
        total={n:0.0 for n in NUTRIENTS}; matched=[]; unmatched=[]; confs=[]; qcount=0; total_weight_g=0.0
        for raw in ingredient_texts(row.get('ingredients')):
            g,uc=parse_amount(raw)
            if g is None or g<=0:continue
            qcount+=1; qty_total+=1; total_weight_g += g
            name,prof,ms=match_ingredient(raw,idx,inverted)
            if prof is None:
                unmatched.append(raw[:120]); continue
            matched.append(name); match_total+=1; confs.append(ms*uc)
            for n in NUTRIENTS:
                if prof.get(n) is not None:total[n]+=prof[n]*g/100.0
        if qcount and matched and len(matched)/(len(matched)+len(unmatched))>=0.60:
            # Recipe serving count is not present in this master dataset. To avoid
            # inventing a serving count, store nutrition normalized to 100 g of
            # ingredient-derived recipe mass.
            divisor = max(total_weight_g, 1.0) / 100.0
            for n in NUTRIENTS:
                if pd.isna(row[n]):df.at[i,n]=round(total[n]/divisor,2)
            df.at[i,'nutrition_source']=('existing_partial+IFCT_ingredient_estimate' if any(pd.notna(existing[c]) for c in NUTRIENTS) else 'IFCT_ingredient_estimate')
            conf=sum(confs)/len(confs) if confs else 0
            coverage=len(matched)/(len(matched)+len(unmatched))
            c=0.5*conf+0.5*coverage
            df.at[i,'nutrition_confidence']=f'{c:.2f}'
            df.at[i,'nutrition_servings_assumption']='Per 100 g ingredient-derived recipe mass; recipe serving count unavailable'
            df.at[i,'nutrition_matched_ingredients']=len(matched)
            df.at[i,'nutrition_unmatched_ingredients']=len(unmatched)
            if len(matched)==len(matched)+len(unmatched):filled+=1
            else:partial+=1
            unmatched_total+=len(unmatched)
        else:
            if qcount:
                df.at[i,'nutrition_source']='insufficient_IFCT_coverage'
                df.at[i,'nutrition_confidence']='low'
                df.at[i,'nutrition_matched_ingredients']=len(matched)
                df.at[i,'nutrition_unmatched_ingredients']=len(unmatched)
        if i%1000==0:print('processed',i)
    # Derive nutrition_match_score only where nutrition exists. This is a suitability
    # score, not a medical quality score: balanced macros + lower sodium/fat is rewarded.
    def score(r):
        vals={n:pd.to_numeric(r[n],errors='coerce') for n in NUTRIENTS}
        if pd.isna(vals['calories']):return r.get('nutrition_match_score',0.0)
        # Broad meal-level heuristic, bounded and deliberately not clinical.
        p=min(max((vals['protein'] or 0)/25,0),1)
        f=min(max((vals['fibre'] or 0)/8,0),1)
        c=min(max((vals['carbohydrates'] or 0)/60,0),1)
        fat=min(max((vals['fat'] or 0)/25,0),1)
        sodium=min(max((vals['sodium'] or 0)/900,0),1)
        return round(max(0,min(1,0.35*p+0.25*f+0.20*c+0.20*(1-0.5*fat-0.5*sodium))),3)
    df['nutrition_match_score']=df.apply(score,axis=1)
    # Validation
    assert len(df)==18568, len(df)
    assert df['recipe_id'].is_unique
    assert list(df['recipe_id'])==list(pd.read_csv(INPUT,usecols=['recipe_id'],low_memory=False)['recipe_id'])
    df.to_csv(OUTPUT,index=False)
    audit={
      'input':str(INPUT),'output':str(OUTPUT),'rows':len(df),'columns':len(df.columns),
      'ifct_records_loaded':len(records),'ifct_usable_profiles':len(idx),
      'existing_complete_before':int(pd.read_csv(INPUT,low_memory=False)[NUTRIENTS].notna().all(axis=1).sum()),
      'recipes_with_any_nutrition_after':int(df[NUTRIENTS].notna().any(axis=1).sum()),
      'recipes_complete_nutrition_after':int(df[NUTRIENTS].notna().all(axis=1).sum()),
      'recipes_ifct_estimated':int((df.nutrition_source=='IFCT_ingredient_estimate').sum()),
      'recipes_existing_complete':int((df.nutrition_source=='existing_complete').sum()),
      'recipes_insufficient_coverage':int((df.nutrition_source=='insufficient_IFCT_coverage').sum()),
      'fully_matched_estimates':filled,'partially_matched_estimates':partial,
      'note':'Ingredient-derived values are estimates because recipe serving counts and cooking retention are not always available. Existing complete values are preserved. Original CSV is not overwritten.'
    }
    AUDIT.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
    print('OUTPUT:',OUTPUT)

if __name__=='__main__':main()
