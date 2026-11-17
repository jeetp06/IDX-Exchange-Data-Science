"""Weeks 5/6 local experiments; April selects, May only reports frozen winners."""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from src import avm

GRIDS = {
    'Linear Regression': [{}],
    'Decision Tree': [
        dict(max_depth=None, min_samples_split=2, min_samples_leaf=1),
        dict(max_depth=12, min_samples_split=10, min_samples_leaf=10),
        dict(max_depth=18, min_samples_split=20, min_samples_leaf=5)],
    'Random Forest': [
        dict(n_estimators=60, max_depth=None, min_samples_leaf=1, max_features=1.0),
        dict(n_estimators=100, max_depth=22, min_samples_leaf=3, max_features=.8),
        dict(n_estimators=100, max_depth=16, min_samples_leaf=5, max_features=.6)]}

def load_splits(root):
    raw, files = avm.load_raw(Path(root)/'data/raw')
    clean, audit = avm.clean_properties(raw)
    splits, chronology = avm.chronological_split(clean, '2026-10-12')
    bounded, bounds = avm.price_outlier_filter(splits)
    assert chronology['validation_month'] == '2026-04' and chronology['test_month'] == '2026-05'
    assert {k: len(v) for k,v in bounded.items()} == dict(train=37277, validation=11913, test=11919)
    return splits, bounded, dict(chronology=chronology, bounds=bounds, cleaning=audit)

def inputs(frame, numeric, categorical):
    features = numeric + categorical
    assert not set(features) & avm.FORBIDDEN_FEATURES
    x = frame[features].copy()
    x[numeric] = x[numeric].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf],np.nan)
    for c in categorical:
        x[c] = x[c].astype(object).where(x[c].notna(),np.nan)
    return x

def pipeline(name, numeric, categorical, params):
    p = avm.build_pipeline(numeric, categorical)
    if name == 'Decision Tree':
        p.set_params(model=DecisionTreeRegressor(random_state=42, **params))
    elif name == 'Random Forest':
        p.set_params(model=RandomForestRegressor(random_state=42, n_jobs=2, **params))
    return p

def score(y, pred):
    return {**avm.metrics(y,pred), 'RMSE_USD':float(np.sqrt(np.mean((np.asarray(y)-pred)**2)))}

def fit_candidates(chunks, numeric, categorical, grids=GRIDS):
    x = {s:inputs(chunks[s],numeric,categorical) for s in ('train','validation')}
    records, winners = [], {}
    for name, grid in grids.items():
        best = -np.inf
        for i, params in enumerate(grid):
            start = time.monotonic()
            model = pipeline(name,numeric,categorical,params)
            model.fit(x['train'],chunks['train'][avm.TARGET])
            row = dict(model=name,candidate=i,parameters=json.dumps(params,sort_keys=True),
                       predictors=len(numeric+categorical),
                       transformed_predictors=len(model['preprocess'].get_feature_names_out()))
            for s in x:
                row.update({f'{s}_{k}':v for k,v in score(chunks[s][avm.TARGET],model.predict(x[s])).items()})
            row['fit_and_score_seconds'] = time.monotonic()-start
            records.append(row)
            print(name, i, 'April R2',round(row['validation_R2'],6),flush=True)
            if row['validation_R2'] > best:
                best = row['validation_R2']; winners[name]=(model,row.copy())
    return pd.DataFrame(records), winners

def importance(model, path, title):
    names = model['preprocess'].get_feature_names_out()
    values = model['model'].feature_importances_
    table = pd.DataFrame({'transformed_feature':names,'impurity_importance':values}).sort_values('impurity_importance',ascending=False)
    table.to_csv(path.with_suffix('.csv'),index=False)
    top=table.head(18).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,7)); ax.barh(top.transformed_feature,top.impurity_importance)
    ax.set(title=title+' (impurity; not causal)',xlabel='Training impurity importance')
    fig.tight_layout(); fig.savefig(path,dpi=140); plt.close(fig)

def comparison_charts(table, prefix, figs, labels=None):
    labels = labels if labels is not None else table.model
    for metric in ('R2','MdAPE_pct'):
        fig,ax=plt.subplots(figsize=(12,5))
        pos=np.arange(len(table))
        ax.bar(pos-.18,table['train_'+metric],.36,label='Train')
        ax.bar(pos+.18,table['validation_'+metric],.36,label='April validation')
        ax.set_xticks(pos,labels,rotation=25,ha='right'); ax.set(ylabel=metric,title=prefix+' chronological comparison'); ax.legend()
        fig.tight_layout(); fig.savefig(figs/f'{prefix}_{metric}.png',dpi=140); plt.close(fig)

def evaluate_winner(model, numeric, categorical, splits, bounded, out, prefix):
    result={}
    for pop,frame in [('bounded',bounded['test']),('all_cleaned',splits['test'])]:
        pred=model.predict(inputs(frame,numeric,categorical))
        result[pop]=dict(rows=len(frame),**score(frame[avm.TARGET],pred))
        if pop == 'bounded':
            actual=frame[avm.TARGET].to_numpy()
            fig,ax=plt.subplots(figsize=(6,6)); ax.scatter(actual,pred,s=3,alpha=.2)
            ax.plot([actual.min(),actual.max()],[actual.min(),actual.max()],color='red')
            ax.set(xlabel='Actual sale price ($)',ylabel='Predicted price ($)',title=prefix+' frozen winner: bounded May')
            fig.tight_layout(); fig.savefig(out/'figures'/f'{prefix}_actual_predicted.png',dpi=140); plt.close(fig)
            fig,ax=plt.subplots(figsize=(8,4)); ax.hist((pred-actual)/actual*100,bins=100,range=(-150,150))
            ax.set(title='Bounded May signed percentage errors (display -150% to 150%)',xlabel='100 × (prediction - actual) / actual',ylabel='Sales')
            fig.tight_layout(); fig.savefig(out/'figures'/f'{prefix}_errors.png',dpi=140); plt.close(fig)
    avm._json_dump(out/f'{prefix}_test_metrics.json',result)
    return result

def run_week5(root):
    root=Path(root); out=root/'outputs'; (out/'models').mkdir(exist_ok=True); (out/'figures').mkdir(exist_ok=True)
    splits,bounded,audit=load_splits(root)
    numeric,categorical=avm.choose_features(bounded['train'])
    trials,winners=fit_candidates(bounded,numeric,categorical)
    trials.to_csv(out/'week05_tuning.csv',index=False)
    table=pd.DataFrame([row for model,row in winners.values()])
    table.to_csv(out/'week05_model_comparison.csv',index=False)
    historical=json.loads((out/'baseline_metrics.json').read_text())
    baseline=winners['Linear Regression'][1]
    for metric,tolerance in [('R2',2e-6),('MAE_USD',2),('MAPE_pct',.01),('MdAPE_pct',.01)]:
        assert abs(baseline['validation_'+metric]-historical['validation'][metric]) < tolerance
    for name,(model,row) in winners.items():
        joblib.dump(dict(pipeline=model,numeric=numeric,categorical=categorical,parameters=row['parameters']),out/'models'/f'week05_{name.replace(" ","_")}.joblib')
        if name != 'Linear Regression': importance(model,out/'figures'/f'week05_{name.replace(" ","_")}_importance.png',name)
    best=table.sort_values('validation_R2',ascending=False).iloc[0]['model']
    avm._json_dump(out/'week05_selection.json',dict(winner=best,criterion='Maximum April R2; grid order breaks exact ties',audit=audit,may_previously_inspected=True))
    test=evaluate_winner(winners[best][0],numeric,categorical,splits,bounded,out,'week05')
    comparison_charts(table,'week05',out/'figures')
    return table,test

def run_week6(root):
    import geopandas as gpd
    from sklearn.impute import SimpleImputer
    from src.features import engineer, spatial_join, ENGINEERED, DISTRICTS
    root=Path(root); out=root/'outputs'
    splits,bounded,audit=load_splits(root)
    numeric,categorical=avm.choose_features(bounded['train'])
    districts=gpd.read_file(root/'data/external/school_districts/districts_2024_25.geojson')
    enriched={}; coverage={}
    for split,frame in splits.items():
        enriched[split],coverage[split]=spatial_join(engineer(frame),districts)
    avm._json_dump(out/'week06_spatial_audit.json',coverage)
    low,high=audit['bounds']['train_price_min'],audit['bounds']['train_price_max']
    bounded_enriched={s:f.loc[f[avm.TARGET].between(low,high)].copy() for s,f in enriched.items()}
    original=pd.read_csv(out/'week05_model_comparison.csv')
    rows=[]; models={}
    for _,row in original.iterrows():
        name=row['model']; r=row.to_dict(); r['feature_set']='original'; rows.append(r)
        models[(name,'original')]=(joblib.load(out/'models'/f'week05_{name.replace(" ","_")}.joblib')['pipeline'],numeric,categorical)
    for feature_set in ('engineered','engineered_districts'):
        num=numeric+ENGINEERED; cat=categorical+(DISTRICTS if feature_set.endswith('districts') else [])
        for _,base in original.iterrows():
            name=base['model']; params=json.loads(base['parameters'])
            model=pipeline(name,num,cat,params)
            # Preserve unknown district membership as an explicit unknown category,
            # never substitute a modal real district for an unmatched property.
            if DISTRICTS[0] in cat:
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import OneHotEncoder
                district_pipe=Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='UNKNOWN',keep_empty_features=True)),
                    ('encode',OneHotEncoder(handle_unknown='ignore',max_categories=100,min_frequency=25))])
                pre=model['preprocess']
                pre.transformers=[('numeric',pre.transformers[0][1],num),
                                  ('categorical',pre.transformers[1][1],categorical),
                                  ('districts',district_pipe,DISTRICTS)]
            start=time.monotonic()
            model.fit(inputs(bounded_enriched['train'],num,cat),bounded_enriched['train'][avm.TARGET])
            row=dict(model=name,feature_set=feature_set,parameters=base['parameters'],predictors=len(num+cat),
                     transformed_predictors=len(model['preprocess'].get_feature_names_out()))
            for s in ('train','validation'):
                f=bounded_enriched[s]
                row.update({f'{s}_{k}':v for k,v in score(f[avm.TARGET],model.predict(inputs(f,num,cat))).items()})
            row['fit_and_score_seconds']=time.monotonic()-start
            rows.append(row); models[(name,feature_set)]=(model,num,cat)
            joblib.dump(dict(pipeline=model,numeric=num,categorical=cat,valuation_year=2026,feature_set=feature_set,
                             feature_builder='src.features.engineer then src.features.spatial_join; boundaries local'),
                        out/'models'/f'week06_{name.replace(" ","_")}_{feature_set}.joblib')
            if name!='Linear Regression': importance(model,out/'figures'/f'week06_{name.replace(" ","_")}_{feature_set}_importance.png',name+' '+feature_set)
            print(name,feature_set,'April R2',row['validation_R2'],flush=True)
    table=pd.DataFrame(rows)
    baseline=float(original.loc[original.model.eq('Linear Regression'),'validation_R2'].iloc[0])
    table['R2_delta_vs_linear_baseline']=table.validation_R2-baseline
    per_model=original.set_index('model').validation_R2
    table['R2_delta_vs_same_model_original']=table.validation_R2-table.model.map(per_model)
    table.to_csv(out/'week06_feature_comparison.csv',index=False)
    paired=table.loc[table.feature_set.eq('engineered_districts'),['model','validation_R2','validation_MAE_USD','validation_MAPE_pct','validation_MdAPE_pct']].set_index('model')-table.loc[table.feature_set.eq('engineered'),['model','validation_R2','validation_MAE_USD','validation_MAPE_pct','validation_MdAPE_pct']].set_index('model')
    paired.add_suffix('_district_minus_no_district').to_csv(out/'week06_district_ablation.csv')
    best=table.sort_values('validation_R2',ascending=False,kind='stable').iloc[0]
    model,num,cat=models[(best['model'],best['feature_set'])]
    avm._json_dump(out/'week06_selection.json',dict(winner=best['model'],feature_set=best['feature_set'],criterion='Maximum April R2; stable row order breaks ties',may_previously_inspected=True))
    test=evaluate_winner(model,num,cat,enriched,bounded_enriched,out,'week06')
    comparison_charts(table,'week06',out/'figures',table.model+' / '+table.feature_set)
    return table,test
