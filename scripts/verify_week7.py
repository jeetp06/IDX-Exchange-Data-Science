"""Validate completed Week 7 artifacts without refitting or accessing May."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import joblib
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import avm
from src.week7 import summarize_permutation


def main():
    out = ROOT/'outputs/week07'
    manifest = json.loads((out/'verification.json').read_text())
    assert manifest['status'] == 'passed' and manifest['may_evaluated'] is False
    for path, expected in {**manifest['historical_hashes'], **manifest['source_hashes']}.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    table = pd.read_csv(out/'feature_ablation.csv')
    assert len(table) == 7 and table.experiment.nunique() == 7
    keys = ['validation_R2','validation_MAE_USD','validation_MdAPE_pct']
    assert np.isfinite(table[keys]).all().all()
    selected = table.loc[table.experiment == manifest['selected']].iloc[0]
    np.testing.assert_allclose(selected.validation_R2, table.validation_R2.max(), atol=1e-12)
    np.testing.assert_allclose(selected.validation_R2, manifest['selected_R2'], atol=1e-12)
    reference = table.loc[table.experiment == 'reference'].iloc[0]
    for metric in ('R2', 'MAE_USD', 'MdAPE_pct'):
        np.testing.assert_allclose(table[f'delta_{metric}_vs_reference'],
                                   table[f'validation_{metric}']-reference[f'validation_{metric}'], atol=1e-10)
    for prefix, count in [('',20),('grouped_',10)]:
        repeats = pd.read_csv(out/f'{prefix}permutation_repeats.csv')
        assert len(repeats) == count*manifest['permutation_repeats']
        assert repeats.groupby('feature').size().eq(manifest['permutation_repeats']).all()
        assert not repeats.duplicated(['feature', 'repeat']).any()
        assert np.isfinite(repeats[['R2_drop','MAE_increase','MdAPE_increase']]).all().all()
        assert repeats.groupby('feature')['repeat'].apply(
            lambda values: set(values) == set(range(manifest['permutation_repeats']))).all()
        saved = pd.read_csv(out/f'{prefix}permutation_importance.csv', index_col='feature')
        pd.testing.assert_frame_equal(saved, summarize_permutation(repeats),
                                      check_dtype=False, rtol=1e-10, atol=1e-10)
    pred = pd.read_csv(out/'april_predictions_private.csv')
    assert len(pred) == 11913
    for key, value in avm.metrics(pred.actual,pred.predicted).items():
        np.testing.assert_allclose(value, manifest['reference_scores']['validation'][key], rtol=1e-10)
    model = joblib.load(out/'selected_model_private.joblib')
    assert not set(model['numeric']+model['categorical']) & avm.FORBIDDEN_FEATURES
    columns = model['numeric']+model['categorical']
    assert list(model['pipeline'].feature_names_in_) == columns
    reference_model = joblib.load(ROOT/'outputs/models/week06_Random_Forest_engineered_districts.joblib')
    removed = set() if pd.isna(selected.removed) else set(selected.removed.split('|'))
    assert set(columns) == set(reference_model['numeric']+reference_model['categorical'])-removed
    assert len(list(out.glob('*.png'))) == 7
    assert (ROOT/'docs/WEEK7_RESULTS.md').stat().st_size > 1000
    print('PASS: historical/source hashes, selection, permutation counts, prediction metrics, schema and 7 figures')


if __name__ == '__main__':
    main()
