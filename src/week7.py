"""April-only diagnostics extending the frozen Weeks 5/6 experiment protocol."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from src import avm, experiments
from src.features import engineer, spatial_join, ENGINEERED, DISTRICTS
import matplotlib.pyplot as plt

PROPERTY = ENGINEERED[:5]
INTERACTIONS = ENGINEERED[5:]


def feature_groups(numeric, categorical):
    original = [c for c in numeric + categorical if c not in ENGINEERED + DISTRICTS]
    return {
        'original_property': [c for c in original if c in avm.NUMERIC_CANDIDATES[:5]],
        'districts': DISTRICTS,
        'property_transformations': PROPERTY,
        'geographic_interactions': INTERACTIONS,
        'all_engineered': ENGINEERED + DISTRICTS,
        'all_geography': ['Latitude', 'Longitude', 'PostalCode', 'CountyOrParish'] + INTERACTIONS + DISTRICTS,
        'living_area_family': ['LivingArea', 'LogLivingArea', 'LivingAreaPerBedroom'],
        'lot_size_family': ['LotSizeSquareFeet', 'LogLotSize'],
        'age_family': ['YearBuilt', 'PropertyAge'],
        'bed_bath_family': ['BedroomsTotal', 'BathroomsTotalInteger', 'BedBathRatio', 'LivingAreaPerBedroom'],
    }


def permute_columns(x, columns, order):
    """One shared row order preserves within-group associations, including missingness."""
    changed = x.copy()
    changed.loc[:, columns] = x.iloc[order][columns].to_numpy()
    return changed


def permutation(model, x, y, groups, repeats=5, seed=42):
    base = experiments.score(y, model.predict(x))
    rows = []
    # Identical shuffles across groups give paired, reproducible comparisons.
    orders = [np.random.default_rng(seed + r).permutation(len(x)) for r in range(repeats)]
    for name, columns in groups.items():
        for r, order in enumerate(orders):
            m = experiments.score(y, model.predict(permute_columns(x, columns, order)))
            rows.append(dict(feature=name, repeat=r, R2_drop=base['R2']-m['R2'],
                             MAE_increase=m['MAE_USD']-base['MAE_USD'],
                             MdAPE_increase=m['MdAPE_pct']-base['MdAPE_pct']))
        print('Permutation:', name, flush=True)
    return pd.DataFrame(rows)


def summarize_permutation(raw):
    raw = raw.copy()
    raw['rank'] = raw.groupby('repeat').R2_drop.rank(ascending=False, method='min')
    raw['negative'] = raw.R2_drop < 0
    return raw.groupby('feature').agg(R2_drop=('R2_drop', 'mean'),
        shuffle_sd=('R2_drop', 'std'), min_drop=('R2_drop', 'min'), max_drop=('R2_drop', 'max'),
        best_rank=('rank', 'min'), worst_rank=('rank', 'max'), negative_shuffle_fraction=('negative', 'mean'),
        MAE_increase=('MAE_increase', 'mean'), MdAPE_increase=('MdAPE_increase', 'mean')).sort_values('R2_drop', ascending=False)


def plot_ranking(table, path, title):
    top = table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index, top.R2_drop, xerr=top.shuffle_sd.fillna(0))
    ax.axvline(0, color='black', linewidth=.6)
    ax.set(title=title, xlabel='April R2 decrease (mean +/- shuffle SD; not a confidence interval)')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def error_metrics(frame):
    y = frame.actual.to_numpy(); p = frame.predicted.to_numpy()
    return dict(n=len(frame), **experiments.score(y, p),
        bias_USD=float(np.mean(p-y)), median_signed_pct=float(np.median((p-y)/y*100)),
        overprediction_pct=float(np.mean(p>y)*100),
        p95_absolute_error_USD=float(np.quantile(np.abs(p-y), .95)))


def subgroup_table(errors, key, minimum=50):
    rows = []
    for name, frame in errors.groupby(key, observed=True, dropna=False):
        if len(frame) >= minimum:
            rows.append(dict(group=str(name), **error_metrics(frame)))
    return pd.DataFrame(rows)


def diagnose(model, frame, train, numeric, categorical, out):
    errors = frame[numeric+categorical].copy()
    errors['actual'] = frame[avm.TARGET].to_numpy()
    errors['predicted'] = model.predict(experiments.inputs(frame, numeric, categorical))
    errors['residual'] = errors.predicted-errors.actual
    errors['absolute_error'] = errors.residual.abs()
    errors['signed_pct'] = errors.residual/errors.actual*100
    errors['price_band'] = pd.cut(errors.actual, [0, 500000, 1000000, 2000000, 5000000, np.inf],
        right=False, labels=['<500k', '500k-1m', '1m-2m', '2m-5m', '>=5m'])
    bands = subgroup_table(errors, 'price_band', 2)
    bands.to_csv(out/'price_band_metrics.csv', index=False)
    for col in DISTRICTS + ['CountyOrParish']:
        errors[col] = errors[col].fillna('UNKNOWN')
        subgroup_table(errors, col).to_csv(out/f'errors_by_{col}.csv', index=False)
    # Compare genuinely unseen raw IDs, not merely encoder rare-category buckets.
    for col in DISTRICTS + ['PostalCode']:
        known = set(train[col].dropna())
        status = np.where(frame[col].isna(), 'missing', np.where(frame[col].isin(known), 'seen', 'unseen'))
        errors['geography_status'] = status
        subgroup_table(errors, 'geography_status', 2).to_csv(out/f'coverage_{col}.csv', index=False)
    errors.to_csv(out/'april_predictions_private.csv', index=False)
    errors.nlargest(25, 'absolute_error').to_csv(out/'extreme_errors_private.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0,0].hexbin(errors.actual, errors.predicted, gridsize=65, mincnt=1, bins='log')
    limits = [errors.actual.min(), errors.actual.max()]
    axes[0,0].plot(limits, limits, 'r--'); axes[0,0].set(xlabel='Actual USD', ylabel='Predicted USD')
    axes[0,1].hist(errors.residual/1000, bins=100); axes[0,1].axvline(0, color='red')
    axes[0,1].set(xlabel='Prediction - actual (thousand USD)', ylabel='Sales')
    axes[1,0].hist(errors.absolute_error/1000, bins=100)
    axes[1,0].set(xlabel='Absolute error (thousand USD)', ylabel='Sales')
    axes[1,1].hexbin(errors.actual, errors.signed_pct, gridsize=65, mincnt=1, bins='log')
    axes[1,1].axhline(0, color='red'); axes[1,1].set(xlabel='Actual USD', ylabel='Signed percentage error')
    fig.suptitle('Week 6 reference forest: bounded April validation (all rows)')
    fig.tight_layout(); fig.savefig(out/'error_diagnostics.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(bands.group, bands.MdAPE_pct); axes[0].set(ylabel='MdAPE %')
    axes[1].bar(bands.group, bands.bias_USD); axes[1].axhline(0, color='black'); axes[1].set(ylabel='Mean prediction - actual USD')
    fig.suptitle('April error by actual price band (descriptive, not prediction-time groups)')
    fig.tight_layout(); fig.savefig(out/'price_band_errors.png', dpi=150); plt.close(fig)
    district = pd.read_csv(out/'errors_by_SchoolDistrictUnified.csv').sort_values('MAE_USD').tail(20)
    fig, ax = plt.subplots(figsize=(10, 7)); ax.barh(district.group, district.MAE_USD)
    ax.set(xlabel='April MAE USD', title='Unified districts: largest MAE, minimum 50 April sales')
    fig.tight_layout(); fig.savefig(out/'district_errors.png', dpi=150); plt.close(fig)
    return error_metrics(errors)


def historical_hashes(root):
    paths = list((root/'outputs').glob('week0[56]*')) + list((root/'outputs/models').glob('week0[56]*'))
    paths += list((root/'outputs/figures').glob('week0[56]*')) + list((root/'notebooks').glob('*.ipynb'))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def run(root, repeats=5):
    import geopandas as gpd
    root = Path(root); out = root/'outputs/week07'; out.mkdir(parents=True, exist_ok=True)
    before = historical_hashes(root)
    artifact_path = root/'outputs/models/week06_Random_Forest_engineered_districts.joblib'
    artifact = joblib.load(artifact_path); reference = artifact['pipeline']
    numeric = artifact['numeric']; categorical = artifact['categorical']
    assert set(numeric+categorical) == set(avm.NUMERIC_CANDIDATES+avm.CATEGORICAL_CANDIDATES+ENGINEERED+DISTRICTS)
    provenance = json.loads((root/'data/external/school_districts/provenance.json').read_text())
    assert hashlib.sha256((root/'data/external/school_districts/districts_2024_25.geojson').read_bytes()).hexdigest() == provenance['sha256']
    splits, bounded, audit = experiments.load_splits(root)
    polygons = gpd.read_file(root/'data/external/school_districts/districts_2024_25.geojson')
    enriched = {}; coverage = {}
    # May is loaded by the shared split loader but never enriched, predicted, or selected on.
    for split in ('train', 'validation'):
        enriched[split], coverage[split] = spatial_join(engineer(splits[split]), polygons)
    low, high = audit['bounds']['train_price_min'], audit['bounds']['train_price_max']
    chunks = {s: f.loc[f[avm.TARGET].between(low, high)].copy() for s, f in enriched.items()}
    assert chunks['train'][avm.DATE_COL].max() < chunks['validation'][avm.DATE_COL].min()
    x = {s: experiments.inputs(f, numeric, categorical) for s, f in chunks.items()}
    assert len(x['train']) == 37277 and len(x['validation']) == 11913
    historical = pd.read_csv(root/'outputs/week06_feature_comparison.csv')
    expected = historical.loc[(historical.model=='Random Forest') & (historical.feature_set=='engineered_districts')].iloc[0]
    ref_scores = {s: experiments.score(chunks[s][avm.TARGET], reference.predict(x[s])) for s in x}
    for s in x:
        for key in ('R2', 'MAE_USD', 'MdAPE_pct'):
            np.testing.assert_allclose(ref_scores[s][key], expected[f'{s}_{key}'], rtol=1e-9, atol=1e-7)
    print('Historical Week 6 reference reproduced:', ref_scores, flush=True)
    groups = feature_groups(numeric, categorical)
    records = [dict(experiment='reference', removed='', **{f'{s}_{k}': v for s, m in ref_scores.items() for k,v in m.items()})]
    best_model = reference; best_num = numeric; best_cat = categorical
    best_score = ref_scores['validation']['R2']; best_name = 'reference'
    for name in ['original_property', 'districts', 'property_transformations', 'geographic_interactions', 'all_engineered', 'all_geography']:
        removed = groups[name]
        num = [c for c in numeric if c not in removed]; cat = [c for c in categorical if c not in removed]
        candidate = clone(reference)
        candidate['preprocess'].transformers = [(n,t,[c for c in cols if c not in removed])
            for n,t,cols in candidate['preprocess'].transformers if any(c not in removed for c in cols)]
        start = time.monotonic()
        candidate.fit(experiments.inputs(chunks['train'], num, cat), chunks['train'][avm.TARGET])
        row = dict(experiment='without_'+name, removed='|'.join(removed), seconds=time.monotonic()-start)
        for s in x:
            row.update({f'{s}_{k}':v for k,v in experiments.score(chunks[s][avm.TARGET], candidate.predict(experiments.inputs(chunks[s],num,cat))).items()})
        records.append(row)
        pd.DataFrame(records).to_csv(out/'ablation_progress.csv', index=False)
        print(row['experiment'], row['validation_R2'], flush=True)
        if row['validation_R2'] > best_score:
            best_score = row['validation_R2']; best_model = candidate; best_num = num; best_cat = cat; best_name = row['experiment']
    table = pd.DataFrame(records)
    for key in ('R2', 'MAE_USD', 'MdAPE_pct'):
        table[f'delta_{key}_vs_reference'] = table[f'validation_{key}']-ref_scores['validation'][key]
    table.to_csv(out/'feature_ablation.csv', index=False)
    joblib.dump(dict(pipeline=best_model,numeric=best_num,categorical=best_cat,feature_builder=artifact.get('feature_builder'),valuation_year=2026),out/'selected_model_private.joblib')
    raw = permutation(reference,x['validation'],chunks['validation'][avm.TARGET],{c:[c] for c in numeric+categorical},repeats)
    raw.to_csv(out/'permutation_repeats.csv',index=False)
    ranking = summarize_permutation(raw); ranking.to_csv(out/'permutation_importance.csv')
    plot_ranking(ranking,out/'top20_permutation.png','Week 6 forest: individual-column April permutation')
    grouped_raw = permutation(reference,x['validation'],chunks['validation'][avm.TARGET],groups,repeats)
    grouped_raw.to_csv(out/'grouped_permutation_repeats.csv',index=False)
    grouped = summarize_permutation(grouped_raw); grouped.to_csv(out/'grouped_permutation_importance.csv')
    assert np.isfinite(raw[['R2_drop','MAE_increase','MdAPE_increase']]).all().all()
    assert np.isfinite(grouped_raw[['R2_drop','MAE_increase','MdAPE_increase']]).all().all()
    plot_ranking(grouped,out/'grouped_permutation.png','Joint-column permutation; groups overlap, do not sum')
    experiments.importance(reference,out/'impurity_importance.png','Week 6 reference forest')
    errors = diagnose(reference,chunks['validation'],chunks['train'],numeric,categorical,out)
    all_april = enriched['validation']
    sensitivity = experiments.score(all_april[avm.TARGET],reference.predict(experiments.inputs(all_april,numeric,categorical)))
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax,key in zip(axes,['R2','MAE_USD','MdAPE_pct']):
        ax.barh(table.experiment,table['validation_'+key]); ax.set(xlabel=key)
    fig.suptitle('Fixed RF settings, same bounded April rows; removal experiments')
    fig.tight_layout(); fig.savefig(out/'ablation_comparison.png',dpi=150); plt.close(fig)
    assert historical_hashes(root) == before, 'Historical artifacts changed'
    manifest = dict(status='passed', historical_files_unchanged=True, historical_hashes=before,
        reference_scores=ref_scores, april_all_cleaned=dict(n=len(all_april),**sensitivity), error_summary=errors,
        selected=best_name, criterion='Maximum bounded April R2; reference wins exact ties',selected_R2=best_score,
        train_rows=len(x['train']),validation_rows=len(x['validation']),parameters=reference['model'].get_params(),
        permutation_repeats=repeats,permutation_seed=42,coverage=coverage,python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','geopandas','matplotlib']},
        source_hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((root/'data/raw').glob('CRMLSSold*.csv'))+[root/'data/external/school_districts/districts_2024_25.geojson',root/'src/week7.py']},
        may_evaluated=False)
    avm._json_dump(out/'verification.json',manifest)
    write_report(root, table, ranking, grouped, manifest)
    return manifest


def markdown(frame):
    # Avoid adding tabulate to the existing environment.
    return '| '+' | '.join(map(str,frame.columns))+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None))


def write_report(root, table, ranking, grouped, manifest):
    summary = table[['experiment','validation_R2','validation_MAE_USD','validation_MdAPE_pct','delta_R2_vs_reference']].round(6)
    bands = pd.read_csv(root/'outputs/week07/price_band_metrics.csv')
    text = f'''# Week 7 results

Executed locally against the existing CRMLS snapshot. No May predictions or tuning were performed. May has already been examined in earlier work and is not an untouched holdout.

## Method

Reused `avm`, `experiments.inputs`, deterministic Week 6 features, local official polygons and the saved Week 6 forest. Reproduced historical training/April metrics before experimentation. December–March training: 37,277 bounded sales; April: 11,913. Training-derived price bounds remain $179,265–$9,150,000. Every removal refits a cloned pipeline on identical training rows with fixed seed 42, 60 trees, unrestricted depth, leaf size 1 and max_features 1.0; imputation, scaling and category grouping fit training only. Six removal fits, no hyperparameter search. Saved historical files/notebooks are hash-checked unchanged.

## Controlled ablation

{markdown(summary)}

Selected by April R2: **{manifest['selected']}**, R2 **{manifest['selected_R2']:.6f}**. This is exploratory selection on a repeatedly used validation month, not established future improvement. Error metrics can disagree with R2. Original property removal means the five raw size/room/year columns, retaining engineered descendants and geography. All engineered removal includes all eight Week 6 numeric additions and all three district IDs, leaving the original nine predictors. The extra all-geography removal removes coordinates, ZIP, county, districts and coordinate interactions together, testing dependence beyond district IDs alone. Removing a column family does not establish causality; redundant columns can substitute for it.

## Importance

Full bounded April permutation, {manifest['permutation_repeats']} deterministic shuffles per input/group; importance is baseline R2 minus shuffled R2. Joint group permutations share one row order, preserving internal associations. Top individual inputs:

{markdown(ranking.head(10).reset_index().round(6))}

Grouped results:

{markdown(grouped.reset_index().round(6))}

Error bars show shuffle SD, not confidence intervals, retraining stability, or geographic uncertainty. Raw repeat tables record sign/rank variability. Correlated inputs (age/year, size/log size, district/coordinates) can dilute individual importance; joint families address substitution partly. Groups overlap and cannot be summed. Permuting columns independently can create physically inconsistent records; group values remain associated but cross-group combinations can still be unrealistic. Impurity rankings are supplementary and biased by split opportunities. These diagnostics describe the saved Week 6 reference model, not a refitted ablation winner.

## Prediction errors and robustness

Residual convention is prediction minus actual; positive means overprediction. All bounded April rows appear in the error plots; hexbin color uses log counts. Minimum 50 April sales for district/county reporting; small groups are excluded, not treated as reliable estimates. Raw district IDs are kept separate by level. Narrow subgroup R2 is unstable and should be read with counts, dollar and percentage errors.

{markdown(bands.round(4))}

Overall April bias: **${manifest['error_summary']['bias_USD']:,.0f}**; 95th percentile absolute error: **${manifest['error_summary']['p95_absolute_error_USD']:,.0f}**. Training R2 **{manifest['reference_scores']['train']['R2']:.6f}** versus April **{manifest['reference_scores']['validation']['R2']:.6f}** reveals a substantial fit/generalization gap. Extreme records and full predictions are private local CSVs. Seen/unseen/missing district and ZIP tables compare chronological April coverage against December–March raw memberships; they are diagnostics of the existing model, not spatial holdout tests. Rare unseen groups do not support strong geographic claims. All-cleaned April sensitivity (12,024 sales): R2 **{manifest['april_all_cleaned']['R2']:.6f}**, MAE **${manifest['april_all_cleaned']['MAE_USD']:,.0f}**, MdAPE **{manifest['april_all_cleaned']['MdAPE_pct']:.3f}%**. Target-bounded scores cannot be applied as a production-time selection rule.

## Limits and conclusions

Use the ablation deltas to distinguish useful engineered families from redundant ones; a positive removal delta favors removal only on this April sample. Avoid adding features or declaring a robust improvement from tiny single-month gains. Large joint geographic importance indicates dependence on location information, not proof of geographic transportability. No random temporal splits were introduced. Geographic transfer, model-seed importance stability and rolling-month confirmation remain unverified; use later unseen months and spatially excluded training regions before deployment.

Predictors are explicitly whitelisted; target, closing date, listing prices and DOM are excluded. Engineering does not consume target/date; preprocessing fits training only. Sold snapshots cannot establish historical property-attribute availability. Retrospectively retrieved 2024–25 district boundaries have a vintage mismatch and March 2026 service revision, so strict point-in-time leakage freedom cannot be claimed. District IDs indicate location, not causal school quality.

## Reproduction and artifacts

Run `.\\.venv\\Scripts\\python.exe scripts/run_week7.py`, then `.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v`. No new dependencies. `outputs/week07/` contains individual/group rankings and repeats, impurity comparison, ablation metrics and chart, actual/predicted and residual/absolute-error charts, price-band and district charts/tables, coverage diagnostics, private predictions/extremes, selected model and verification manifest with source hashes, versions and fixed parameters. These outputs remain ignored and local, following the existing licensed-data policy. Existing Week 5/6 results and notebooks are preserved.
'''
    (root/'docs/WEEK7_RESULTS.md').write_text(text,encoding='utf8')
