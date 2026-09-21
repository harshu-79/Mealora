#!/usr/bin/env python3
# Mealora_ML_Integration_OneGo.py
# Run from /Users/harshitha/mealora with: python3 Mealora_ML_Integration_OneGo.py

from pathlib import Path
import sys, re, shutil, json, importlib.util

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / 'datasets' / 'processed'
RESULTS = ROOT / 'Mealora_ML_Results'
MODEL_DIR = PROCESSED / 'mealora_ml_models'
RECIPES = PROCESSED / 'recipe_master_dataset.csv'
INTERACTIONS = PROCESSED / 'user_recipe_interactions.csv'
RANKER = ROOT / 'notebooks' / 'final_ranker.py'
FEATURES = ['pantry_match_pct','missing_ingredient_count','expiry_score','meal_match','diet_match','cuisine_match','time_match','context_score','ingredient_count','cooking_time_minutes']

def stop(msg):
    print('\nERROR:', msg)
    sys.exit(1)

def check_packages():
    needed = ['pandas','numpy','joblib','sklearn','matplotlib','xgboost']
    missing = [p for p in needed if importlib.util.find_spec(p) is None]
    if missing:
        print('\nMissing:', ', '.join(missing))
        print(f'Install with: {sys.executable} -m pip install ' + ' '.join(missing))
        stop('Install missing packages and rerun this script.')

check_packages()
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

def make_features(df):
    out = pd.DataFrame(index=df.index)
    for col in FEATURES:
        out[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0) if col in df.columns else 0.0
    return out[FEATURES].replace([np.inf,-np.inf],0).fillna(0.0)

def build_training_set(recipes, interactions):
    pos = {'cooked','liked','saved'}
    neg = {'skipped','disliked'}
    rows = []
    for _, event in interactions.iterrows():
        rid = pd.to_numeric(event.get('recipe_id'), errors='coerce')
        if pd.isna(rid): continue
        match = recipes[recipes['recipe_id'].astype(str) == str(int(rid))]
        if match.empty: continue
        action = str(event.get('interaction','')).strip().lower()
        if action in pos: label = 1
        elif action in neg: label = 0
        else:
            rating = pd.to_numeric(event.get('rating'), errors='coerce')
            if pd.isna(rating): continue
            label = int(float(rating) >= 4)
        row = match.iloc[0].copy()
        row['_label'] = label
        rows.append(row)
    data = pd.DataFrame(rows)
    if data.empty: stop('No usable interaction examples found.')
    seen = set(pd.to_numeric(interactions['recipe_id'], errors='coerce').dropna().astype(int).tolist())
    unseen = recipes[~recipes['recipe_id'].astype(str).isin({str(x) for x in seen})].copy()
    if len(unseen):
        n = min(max(12, len(data)*2), len(unseen))
        unseen = unseen.sample(n=n, random_state=42)
        unseen['_label'] = 0
        data = pd.concat([data, unseen], ignore_index=True)
    if data['_label'].nunique() < 2: stop('Training data contains only one class.')
    return data

def patch_ranker():
    if not RANKER.exists():
        print('WARNING: notebooks/final_ranker.py not found.')
        return None
    text = RANKER.read_text(encoding='utf-8')
    backup = RANKER.with_name(RANKER.name + '.before_ml_integration_backup')
    if not backup.exists(): shutil.copy2(RANKER, backup)
    loader = """
# ============================================================
# MEALORA SELECTED ML MODEL
# ============================================================
ML_MODEL_PATH = os.path.join(BASE, 'mealora_ml_models', 'mealora_selected_ml_model.joblib')
try:
    _mealora_ml_artifact = joblib.load(ML_MODEL_PATH)
    selected_ml_model = _mealora_ml_artifact['model']
    selected_ml_scaler = _mealora_ml_artifact.get('scaler')
    selected_ml_features = _mealora_ml_artifact.get('features', ['pantry_match_pct','missing_ingredient_count','expiry_score','meal_match','diet_match','cuisine_match','time_match','context_score','ingredient_count','cooking_time_minutes'])
except Exception as _mealora_ml_error:
    selected_ml_model = None
    selected_ml_scaler = None
    selected_ml_features = ['pantry_match_pct','missing_ingredient_count','expiry_score','meal_match','diet_match','cuisine_match','time_match','context_score','ingredient_count','cooking_time_minutes']
    print('Selected ML model load failed:', _mealora_ml_error)
"""
    if 'MEALORA SELECTED ML MODEL' not in text:
        marker = '\n# ============================================================\n# RECOMMENDATION FUNCTION'
        text = text.replace(marker, '\n' + loader + marker, 1) if marker in text else loader + '\n' + text
    prediction = """
    # ========================================================
    # SELECTED SUPERVISED ML MODEL SUITABILITY
    # ========================================================
    if selected_ml_model is not None:
        ml_input = pd.DataFrame(index=df.index)
        for _feature in selected_ml_features:
            ml_input[_feature] = pd.to_numeric(df[_feature], errors='coerce').fillna(0.0) if _feature in df.columns else 0.0
        ml_input = ml_input[selected_ml_features]
        try:
            if selected_ml_scaler is not None:
                ml_probability = selected_ml_model.predict_proba(selected_ml_scaler.transform(ml_input))[:,1]
            else:
                ml_probability = selected_ml_model.predict_proba(ml_input)[:,1]
            df['ml_suitability_score'] = np.clip(np.asarray(ml_probability,dtype=float),0.0,1.0)
        except Exception as _e:
            print('Selected ML prediction failed:', _e)
            df['ml_suitability_score'] = 0.0
    else:
        df['ml_suitability_score'] = 0.0
"""
    if 'SELECTED SUPERVISED ML MODEL SUITABILITY' not in text:
        marker = re.search(r'(?m)^\s*#\s*=+\s*\n\s*#\s*8\.\s*FINAL WEIGHTED RANKING', text)
        if marker: text = text[:marker.start()] + prediction + text[marker.start():]
        else:
            marker = re.search(r'(?m)^\s*df\["final_score"\]\s*=\s*\(', text)
            if marker: text = text[:marker.start()] + prediction + text[marker.start():]
    if 'ML_INTEGRATED_FINAL_SCORE' not in text:
        start = re.search(r'(?m)^\s*df\["final_score"\]\s*=\s*\(', text)
        if not start: print('WARNING: final_score assignment not found.'); return backup
        open_pos = text.find('(', start.start())
        depth = 0; close_pos = None
        for i in range(open_pos, len(text)):
            if text[i] == '(': depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0: close_pos = i + 1; break
        if close_pos is None: print('WARNING: Could not parse final_score.'); return backup
        old = text[start.start():close_pos]
        expression = old.split('=',1)[1].strip()
        new_block = """    # ========================================================
    # 8. FINAL WEIGHTED RANKING — ML INTEGRATED
    # ========================================================
    df['base_final_score'] = %s
    df['final_score'] = (
        0.85 * df['base_final_score']
        + 0.15 * df['ml_suitability_score']
    )
    # ML_INTEGRATED_FINAL_SCORE
""" % expression
        text = text[:start.start()] + new_block + text[close_pos:]
    RANKER.write_text(text, encoding='utf-8')
    print('LIVE RANKER UPDATED:', RANKER)
    print('BACKUP:', backup)
    return backup

def main():
    print('='*72)
    print('MEALORA — NAIVE BAYES vs RANDOM FOREST vs XGBOOST')
    print('='*72)
    if not RECIPES.exists(): stop(f'Missing: {RECIPES}')
    if not INTERACTIONS.exists(): stop(f'Missing: {INTERACTIONS}')
    RESULTS.mkdir(exist_ok=True); MODEL_DIR.mkdir(parents=True, exist_ok=True)
    recipes = pd.read_csv(RECIPES, low_memory=False)
    interactions = pd.read_csv(INTERACTIONS, low_memory=False)
    print('Recipes:', len(recipes)); print('Interactions:', len(interactions))
    if 'interaction' in interactions.columns: print(interactions['interaction'].value_counts().to_string())
    train = build_training_set(recipes, interactions)
    X = make_features(train); y = train['_label'].astype(int)
    print('Training examples:', len(train), '| Positive:', int((y==1).sum()), '| Negative:', int((y==0).sum()))
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)
    scaler = StandardScaler(); nb = GaussianNB(); nb.fit(scaler.fit_transform(X_train), y_train)
    rf = RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced',n_jobs=-1); rf.fit(X_train,y_train)
    xgb = XGBClassifier(n_estimators=250,max_depth=4,learning_rate=0.05,subsample=0.85,colsample_bytree=0.85,objective='binary:logistic',eval_metric='logloss',random_state=42,n_jobs=2); xgb.fit(X_train,y_train)
    models = {'Naive Bayes':(nb,scaler.transform(X_test)),'Random Forest':(rf,X_test),'XGBoost':(xgb,X_test)}
    rows=[]; preds={}
    for name,(model,test_data) in models.items():
        pred=model.predict(test_data); preds[name]=pred
        rows.append({'Model':name,'Accuracy':accuracy_score(y_test,pred),'Precision':precision_score(y_test,pred,zero_division=0),'Recall':recall_score(y_test,pred,zero_division=0),'F1':f1_score(y_test,pred,zero_division=0)})
    results=pd.DataFrame(rows).sort_values(['F1','Precision','Recall','Accuracy'],ascending=False).reset_index(drop=True)
    print('\n'+results.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    winner=str(results.iloc[0]['Model']); print('\nWINNER:',winner); print('Selection: highest F1-score')
    results_csv=RESULTS/'model_comparison_results.csv'; results.to_csv(results_csv,index=False)
    winner_model={'Naive Bayes':nb,'Random Forest':rf,'XGBoost':xgb}[winner]
    winner_scaler=scaler if winner=='Naive Bayes' else None
    model_path=MODEL_DIR/'mealora_selected_ml_model.joblib'
    joblib.dump({'model_name':winner,'model':winner_model,'scaler':winner_scaler,'features':FEATURES,'evaluation':results.to_dict(orient='records')},model_path)
    joblib.dump({'Naive Bayes':nb,'Random Forest':rf,'XGBoost':xgb,'scaler':scaler,'features':FEATURES},MODEL_DIR/'mealora_three_model_comparison.joblib')
    plt.figure(figsize=(10,6)); pos=np.arange(len(results)); width=0.18
    for i,m in enumerate(['Accuracy','Precision','Recall','F1']): plt.bar(pos+(i-1.5)*width,results[m].values,width,label=m)
    plt.xticks(pos,results['Model'].values); plt.ylim(0,1.05); plt.ylabel('Score'); plt.title('Mealora — ML Model Comparison'); plt.legend(); plt.tight_layout()
    comparison_png=RESULTS/'model_comparison.png'; plt.savefig(comparison_png,dpi=200); plt.close()
    cm=confusion_matrix(y_test,preds[winner]); plt.figure(figsize=(6,5)); plt.imshow(cm,interpolation='nearest'); plt.title(f'Confusion Matrix — {winner}'); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.xticks([0,1],['Not Suitable','Suitable']); plt.yticks([0,1],['Not Suitable','Suitable'])
    for (r,c),v in np.ndenumerate(cm): plt.text(c,r,str(v),ha='center',va='center')
    plt.tight_layout(); confusion_png=RESULTS/'confusion_matrix_winner.png'; plt.savefig(confusion_png,dpi=200); plt.close()
    feature_png=None
    if hasattr(winner_model,'feature_importances_'):
        imp=pd.DataFrame({'Feature':FEATURES,'Importance':winner_model.feature_importances_}).sort_values('Importance',ascending=False); imp.to_csv(RESULTS/'feature_importance.csv',index=False)
        plt.figure(figsize=(9,6)); plt.barh(imp['Feature'].iloc[::-1],imp['Importance'].iloc[::-1]); plt.xlabel('Importance'); plt.title(f'Feature Importance — {winner}'); plt.tight_layout(); feature_png=RESULTS/'feature_importance_winner.png'; plt.savefig(feature_png,dpi=200); plt.close()
    backup=patch_live_ranker()
    report={'winner':winner,'selection_metric':'F1','results':results.to_dict(orient='records'),'features':FEATURES,'training_examples':int(len(train)),'positive_examples':int((y==1).sum()),'negative_examples':int((y==0).sum()),'model_path':str(model_path),'comparison_csv':str(results_csv),'comparison_png':str(comparison_png),'confusion_matrix_png':str(confusion_png),'feature_importance_png':str(feature_png) if feature_png else None,'ranker':str(RANKER),'ranker_backup':str(backup) if backup else None}
    (RESULTS/'ML_integration_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\nCOMPLETE'); print('Winner:',winner); print('Comparison PNG:',comparison_png); print('Confusion Matrix:',confusion_png); print('Model:',model_path); print('Ranker:',RANKER)
    print('\nRestart Flask and test Get Recommendations. Look for ml_suitability_score.')
    print('NOTE: current interaction data is small; describe metrics as initial/prototype evaluation.')

if __name__ == '__main__':
    main()