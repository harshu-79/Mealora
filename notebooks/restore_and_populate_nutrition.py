import os
import shutil
import re
import numpy as np
import pandas as pd

BASE = 'datasets/processed'
RAW = 'datasets/raw'
MASTER = os.path.join(BASE, 'recipe_master_dataset.csv')
BACKUP = os.path.join(BASE, 'recipe_master_dataset_before_nutrition.csv')
NUTRITION = os.path.join(RAW, 'Indian_Food_Ingredients_Nutrition_CookingMethods.csv')

NUTRI_COLS = {
    'calories': 'calories',
    'carbohydrates': 'carbohydrates',
    'protein': 'protein',
    'fat': 'fat',
    'fibre': 'fibre',
    'sodium': 'sodium',
    'calcium': 'calcium',
    'iron': 'iron',
    'vitamin_c': 'vitamin_c',
    'folate': 'folate',
}

def norm(x):
    x = str(x or '').lower().strip()
    x = x.replace('&', ' and ')
    x = re.sub(r"[^a-z0-9]+", ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x

def find_col(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

print('=' * 70)
print('MEALORA: RESTORE MASTER DATASET + POPULATE NUTRITION')
print('=' * 70)

if not os.path.exists(BACKUP):
    raise FileNotFoundError(
        f'Missing backup: {BACKUP}\n'
        'Do NOT continue. Restore the original master dataset first.'
    )
if not os.path.exists(NUTRITION):
    raise FileNotFoundError(f'Missing nutrition source: {NUTRITION}')

# Keep the current file safe before restoring the known pre-nutrition copy.
if os.path.exists(MASTER):
    safety = MASTER + '.before_restore.csv'
    shutil.copy2(MASTER, safety)
    print('Safety backup:', safety)

shutil.copy2(BACKUP, MASTER)
print('Restored master dataset from:', BACKUP)

recipes = pd.read_csv(MASTER, low_memory=False)
nut = pd.read_csv(NUTRITION, low_memory=False)

name_col = find_col(recipes, ['recipe_name', 'final_food_name', 'name'])
source_name_col = find_col(nut, ['final_food_name', 'recipe_original', 'food_name', 'recipe_name'])
if not name_col or not source_name_col:
    raise RuntimeError(f'Could not identify recipe name columns: master={recipes.columns.tolist()}, nutrition={nut.columns.tolist()}')

# Ensure the master has the expected nutrition columns, but leave unmatched rows empty.
for c in NUTRI_COLS.values():
    if c not in recipes.columns:
        recipes[c] = np.nan

# Nutrition source columns.
source_cols = {
    'calories': find_col(nut, ['Calories (kcal)', 'calories', 'Calories']),
    'carbohydrates': find_col(nut, ['Carbohydrates (g)', 'carbohydrates', 'Carbohydrates']),
    'protein': find_col(nut, ['Protein (g)', 'protein', 'Protein']),
    'fat': find_col(nut, ['Fats (g)', 'Fat (g)', 'fat', 'Fats']),
    'fibre': find_col(nut, ['Fibre (g)', 'Fiber (g)', 'fibre', 'fiber']),
    'sodium': find_col(nut, ['Sodium (mg)', 'sodium', 'Sodium']),
    'calcium': find_col(nut, ['Calcium (mg)', 'calcium', 'Calcium']),
    'iron': find_col(nut, ['Iron (mg)', 'iron', 'Iron']),
    'vitamin_c': find_col(nut, ['Vitamin C (mg)', 'vitamin_c', 'Vitamin C']),
    'folate': find_col(nut, ['Folate (µg)', 'Folate (ug)', 'folate']),
}

# Conservative matching: exact normalized name first, then use the already-computed
# matching columns from the nutrition source when they are present and trustworthy.
recipe_norm = recipes[name_col].map(norm)
nut_norm = nut[source_name_col].map(norm)
exact_map = {}
for i, n in enumerate(nut_norm):
    if n and n not in exact_map:
        exact_map[n] = i

matched = 0
for ri, rn in enumerate(recipe_norm):
    if not rn or rn not in exact_map:
        continue
    ni = exact_map[rn]
    row = nut.iloc[ni]
    valid_core = True
    for key in ['calories', 'protein', 'carbohydrates', 'fat', 'fibre']:
        c = source_cols[key]
        if c is None or pd.isna(pd.to_numeric(row[c], errors='coerce')):
            valid_core = False
            break
    if not valid_core:
        continue
    for key, dest in NUTRI_COLS.items():
        src = source_cols[key]
        if src:
            recipes.at[ri, dest] = pd.to_numeric(row[src], errors='coerce')
    matched += 1

# Use the source's verified best-match mapping only when the target name is explicitly
# present in the nutrition reference and the match score is strong.
if {'best_match_clean', 'composite_score'}.issubset(nut.columns):
    master_norm = recipes[name_col].map(norm)
    best_map = {}
    for _, row in nut.iterrows():
        best = norm(row.get('best_match_clean', ''))
        score = pd.to_numeric(row.get('composite_score'), errors='coerce')
        if best and pd.notna(score) and score >= 0.90 and best not in best_map:
            best_map[best] = row
    for ri, rn in enumerate(master_norm):
        if rn in exact_map or rn not in best_map:
            continue
        row = best_map[rn]
        valid_core = all(
            source_cols[k] and pd.notna(pd.to_numeric(row[source_cols[k]], errors='coerce'))
            for k in ['calories', 'protein', 'carbohydrates', 'fat', 'fibre']
        )
        if not valid_core:
            continue
        for key, dest in NUTRI_COLS.items():
            src = source_cols[key]
            if src:
                recipes.at[ri, dest] = pd.to_numeric(row[src], errors='coerce')
        matched += 1

recipes.to_csv(MASTER, index=False)

core = ['calories', 'protein', 'carbohydrates', 'fat', 'fibre']
coverage = recipes[core].notna().all(axis=1)
print('Master recipes:', len(recipes))
print('Recipes with complete core nutrition:', int(coverage.sum()))
print(f'Nutrition coverage: {coverage.mean()*100:.2f}%')
print('Nutrition columns:', ', '.join(NUTRI_COLS.values()))
print('\nDONE. Master dataset restored first, then nutrition values populated.\n')
