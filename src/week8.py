"""Evaluate existing frozen AVMs; preserve prior weeks and select only on April."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform

import joblib
import numpy as np
import pandas as pd

from src import avm, experiments
from src.features import engineer, spatial_join, ENGINEERED, DISTRICTS
from src.evaluation import (METRICS, training_price_bands, metric_rows,
                            assign_price_bands, validation_assessment)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def historical_hashes(root):
    """Protect previous results, source, docs and notebooks, excluding Week 8."""
    root = Path(root)
    paths = [p for p in (root/'outputs').rglob('*') if p.is_file()
             and 'week08' not in p.relative_to(root/'outputs').parts
             and not any(part in ('ipython','jupyter_runtime','local_kernel') for part in p.parts)]
    paths += [p for p in (root/'notebooks').glob('*.ipynb') if p.name != '06_evaluation.ipynb']
    paths += [p for p in (root/'docs').glob('*') if p.is_file() and p.name != 'WEEK8_RESULTS.md']
    paths += [root/'src'/name for name in ['avm.py','features.py','experiments.py','verification.py','week7.py']]
    paths += list(root.glob('requirements*.txt'))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(paths))}


def model_specs(root):
    """All persisted pipelines discovered in the existing output tree."""
    paths = sorted(p for p in (root/'outputs').rglob('*.joblib')
                   if 'week08' not in p.relative_to(root/'outputs').parts)
    specs = []
    for p in paths:
        if p.name == 'baseline_pipeline.joblib':
            model_id, family, feature_set = 'W04 Linear Regression / original', 'Linear Regression', 'original'
        elif p.name == 'selected_model_private.joblib' and p.parent.name == 'week07':
            selection = json.loads((p.parent/'verification.json').read_text())
            feature_set = selection['selected']
            model_id, family = 'W07 Random Forest / '+feature_set, 'Random Forest'
        else:
            stem = p.stem
            week = stem.split('_', 1)[0]
            if week not in ('week05','week06'):
                raise ValueError(f'Unrecognized saved artifact requires explicit inventory: {p}')
            family = next((f for f in ['Linear Regression','Decision Tree','Random Forest']
                           if stem.startswith(week+'_'+f.replace(' ','_'))), None)
            if family is None:
                raise ValueError(f'Unrecognized model family: {p}')
            feature_set = stem[len(week+'_'+family.replace(' ','_')):].lstrip('_') or 'original'
            model_id = 'W'+week[-2:]+' '+family+' / '+feature_set
        specs.append(dict(model_id=model_id, family=family, feature_set=feature_set,
                          artifact=p.relative_to(root).as_posix(), sha256=sha256(p)))
    if not specs:
        raise FileNotFoundError('No saved models to evaluate')
    return specs


def load_model(root, spec):
    artifact = joblib.load(root/spec['artifact'])
    numeric = artifact.get('numeric', artifact.get('numeric_features'))
    categorical = artifact.get('categorical', artifact.get('categorical_features'))
    columns = numeric+categorical
    allowed = set(avm.NUMERIC_CANDIDATES+avm.CATEGORICAL_CANDIDATES+ENGINEERED+DISTRICTS)
    assert set(columns) <= allowed and not set(columns) & avm.FORBIDDEN_FEATURES
    pipeline = artifact['pipeline']
    assert list(pipeline.feature_names_in_) == columns
    used = {c for _, transformer, cols in pipeline['preprocess'].transformers
            if transformer != 'drop' for c in cols}
    assert used == set(columns)
    return artifact, numeric, categorical


def historical_reference(root, spec):
    if spec['artifact'] == 'outputs/baseline_pipeline.joblib':
        report = json.loads((root/'outputs/baseline_metrics.json').read_text())
        return {split: report[split] for split in ('train','validation','test')}
    if spec['artifact'].startswith('outputs/week07/'):
        table = pd.read_csv(root/'outputs/week07/feature_ablation.csv')
        row = table.loc[table.experiment == spec['feature_set']].iloc[0]
    else:
        table = pd.read_csv(root/'outputs/week06_feature_comparison.csv')
        row = table.loc[(table.model == spec['family']) & (table.feature_set == spec['feature_set'])].iloc[0]
    return {split: {key: float(row[f'{split}_{key}']) for key in METRICS}
            for split in ('train','validation')}


def archived_experiments(root):
    tuning = pd.read_csv(root/'outputs/week05_tuning.csv')
    winners = pd.read_csv(root/'outputs/week05_model_comparison.csv').set_index('model').candidate
    rows = []
    for _, row in tuning.iterrows():
        rows.append(dict(week=5, experiment=f'{row.model} candidate {row.candidate}',
                         saved_pipeline=bool(row.candidate == winners[row.model]),
                         validation_R2=row.validation_R2, source='outputs/week05_tuning.csv',
                         note='Only each family winner was persisted'))
    ablations = pd.read_csv(root/'outputs/week07/feature_ablation.csv')
    selected = json.loads((root/'outputs/week07/verification.json').read_text())['selected']
    aliases = {'reference': 'W06 engineered_districts',
               'without_districts': 'W06 engineered', 'without_all_engineered': 'W05 original'}
    for _, row in ablations.iterrows():
        rows.append(dict(week=7, experiment=row.experiment,
                         saved_pipeline=bool(row.experiment == selected or row.experiment in aliases),
                         validation_R2=row.validation_R2, source='outputs/week07/feature_ablation.csv',
                         note=aliases.get(row.experiment, 'Selected artifact' if row.experiment == selected
                                          else 'Metrics only; no saved pipeline for full-population/test evaluation')))
    return pd.DataFrame(rows)


def run(root):
    import geopandas as gpd
    from src.week8_report import write_figures, write_report
    root = Path(root).resolve()
    out = root/'outputs/week08'
    (out/'predictions_private').mkdir(parents=True, exist_ok=True)
    before = historical_hashes(root)
    specs = model_specs(root)
    splits, bounded, audit = experiments.load_splits(root)
    assert audit['cleaning']['raw'] == 124404 and sum(map(len,splits.values())) == 61698
    assert splits['train'][avm.DATE_COL].max() < splits['validation'][avm.DATE_COL].min()
    assert splits['validation'][avm.DATE_COL].max() < splits['test'][avm.DATE_COL].min()
    band_spec = training_price_bands(splits['train'][avm.TARGET])
    avm._json_dump(out/'price_bands.json', band_spec)
    distribution = pd.DataFrame({'quantile': list(band_spec['quantiles']),
                                 'ClosePrice_USD': list(band_spec['quantiles'].values())})
    distribution.to_csv(out/'training_price_distribution.csv', index=False)
    provenance_path = root/'data/external/school_districts/provenance.json'
    geography_path = provenance_path.parent/'districts_2024_25.geojson'
    provenance = json.loads(provenance_path.read_text())
    assert sha256(geography_path) == provenance['sha256']
    polygons = gpd.read_file(geography_path)
    rows, inventory, comparisons, tail_rows, coverage = [], [], [], [], {}
    selection = None
    # This phase order freezes every assessment before even predicting May.
    for phase, split_names in [('assessment', ('train','validation')), ('test', ('test',))]:
        frames = {}
        for split in split_names:
            frames[split], coverage[split] = spatial_join(engineer(splits[split], valuation_year=2026), polygons)
        for index, spec in enumerate(specs):
            artifact, numeric, categorical = load_model(root, spec)
            model = artifact['pipeline']
            if phase == 'assessment':
                inventory.append(dict(**spec, estimator=type(model['model']).__name__,
                                      predictors=len(numeric+categorical),
                                      numeric=json.dumps(numeric), categorical=json.dumps(categorical),
                                      parameters=json.dumps(model['model'].get_params(), sort_keys=True)))
            expected = historical_reference(root, spec)
            for split in split_names:
                frame = frames[split]
                actual = frame[avm.TARGET].to_numpy(dtype=float)
                predicted = model.predict(experiments.inputs(frame,numeric,categorical))
                low, high = audit['bounds']['train_price_min'], audit['bounds']['train_price_max']
                inside = frame[avm.TARGET].between(low, high).to_numpy()
                assert int(inside.sum()) == len(bounded[split])
                np.savez_compressed(out/'predictions_private'/f'model_{index:02d}_{split}.npz',
                                    actual=actual, predicted=predicted, bounded=inside,
                                    row_index=frame.index.to_numpy())
                for population, mask in [('all_cleaned', np.ones(len(frame),dtype=bool)), ('bounded',inside)]:
                    calculated = metric_rows(actual[mask],predicted[mask],band_spec)
                    assert calculated[0]['n_invalid_actual'] == 0
                    assert calculated[0]['n_excluded_percentage'] == 0
                    for metrics in calculated:
                        rows.append(dict(model_id=spec['model_id'], family=spec['family'],
                                         feature_set=spec['feature_set'], split=split,
                                         period='2025-12 to 2026-03' if split=='train' else
                                         '2026-04' if split=='validation' else '2026-05',
                                         population=population, **metrics))
                    if population == 'bounded' and split in expected:
                        for metric, value in expected[split].items():
                            if metric in METRICS:
                                np.testing.assert_allclose(calculated[0][metric],value,rtol=1e-8,atol=1e-6)
                                comparisons.append(dict(model_id=spec['model_id'],split=split,metric=metric,
                                                        expected=value,observed=calculated[0][metric],
                                                        difference=calculated[0][metric]-value))
                squared = (predicted-actual)**2
                absolute = np.abs(predicted-actual)
                tail_rows.append(dict(model_id=spec['model_id'],split=split,n_all=len(frame),
                                      n_outside=int((~inside).sum()),
                                      outside_SSE_share_pct=float(squared[~inside].sum()/squared.sum()*100),
                                      outside_absolute_error_share_pct=float(absolute[~inside].sum()/absolute.sum()*100)))
            print(phase, spec['model_id'], 'evaluated', flush=True)
            del artifact, model
        if phase == 'assessment':
            ranks, selection = validation_assessment(pd.DataFrame(rows))
            ranks.to_csv(out/'validation_comparison.csv',index=False)
            avm._json_dump(out/'selection_before_may.json',selection)
            selection_hash = sha256(out/'selection_before_may.json')
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'metrics_summary.csv',index=False)
    pd.DataFrame(inventory).to_csv(out/'model_inventory.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'historical_reproduction.csv',index=False)
    pd.DataFrame(tail_rows).to_csv(out/'tail_sensitivity.csv',index=False)
    archived_experiments(root).to_csv(out/'archived_experiments.csv',index=False)
    assert sha256(out/'selection_before_may.json') == selection_hash
    _, repeated = validation_assessment(summary)
    assert repeated == selection
    assert historical_hashes(root) == before, 'Prior weekly artifacts changed'
    manifest = dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),
                    refitted_models=0, model_count=len(specs), metric_rows=len(summary),
                    historical_reproduction_checks=len(comparisons),
                    historical_hashes=before, historical_files_unchanged=True,
                    cleaning=audit['cleaning'],chronology=audit['chronology'],bounds=audit['bounds'],
                    price_bands=band_spec,selection=selection,selection_file_sha256=selection_hash,
                    phase_order=['train','validation','freeze assessment','test'],
                    may_previously_inspected=True,may_used_for_selection=False,
                    coverage=coverage,python=platform.python_version(),
                    packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','geopandas','matplotlib','nbformat']},
                    source_hashes={p.relative_to(root).as_posix():sha256(p) for p in
                                   avm.find_files(root/'data/raw')+[geography_path,provenance_path]+
                                   [root/'src'/n for n in ['avm.py','experiments.py','features.py','evaluation.py','week8.py','week8_report.py']]})
    write_figures(root,summary,specs,selection,band_spec)
    write_report(root,summary,pd.DataFrame(inventory),manifest)
    artifacts = [p for p in out.rglob('*') if p.is_file() and p.name != 'verification.json']
    artifacts += [root/'docs/WEEK8_RESULTS.md']
    manifest['output_hashes']={p.relative_to(root).as_posix():sha256(p) for p in artifacts}
    avm._json_dump(out/'verification.json',manifest)
    print('Week 8 complete:',len(specs),'frozen pipelines;',len(summary),'metric rows',flush=True)
    return summary, manifest
