"""Independently recompute all Week 8 CSV metrics from saved prediction arrays."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.evaluation import METRICS, metric_rows, training_price_bands, validation_assessment
from src.week8 import sha256, historical_hashes
from src.week8_report import CHARTS


def verify(root=ROOT):
    root = Path(root)
    out = root/'outputs/week08'
    manifest=json.loads((out/'verification.json').read_text(encoding='utf-8'))
    assert manifest['status']=='passed' and manifest['refitted_models']==0
    assert manifest['may_used_for_selection'] is False and manifest['may_previously_inspected'] is True
    assert manifest['phase_order']==['train','validation','freeze assessment','test']
    for key in ('historical_hashes','source_hashes','output_hashes'):
        for name, expected in manifest[key].items():
            assert sha256(root/name)==expected, f'Changed {key}: {name}'
    assert historical_hashes(root)==manifest['historical_hashes']
    summary=pd.read_csv(out/'metrics_summary.csv',float_precision='round_trip')
    inventory=pd.read_csv(out/'model_inventory.csv')
    assert len(inventory)==manifest['model_count']
    assert len(summary)==manifest['metric_rows']==len(inventory)*3*2*5
    identity=['model_id','split','population','price_band']
    assert not summary.duplicated(identity).any()
    assert set(summary.split)=={'train','validation','test'}
    assert set(summary.population)=={'bounded','all_cleaned'}
    assert summary.n_invalid_actual.eq(0).all() and summary.n_excluded_percentage.eq(0).all()
    recalculated=[]
    baseline_arrays={}
    specification=json.loads((out/'price_bands.json').read_text())
    for index,spec in inventory.iterrows():
        assert sha256(root/spec.artifact)==spec.sha256
        for split in ('train','validation','test'):
            path=out/'predictions_private'/f'model_{index:02d}_{split}.npz'
            with np.load(path) as data:
                actual,predicted,inside=data['actual'],data['predicted'],data['bounded']
                assert len(actual)==manifest['chronology']['sizes_before_outlier_filter'][split]
                assert np.isfinite(actual).all() and (actual>0).all()
                assert int(inside.sum())==manifest['bounds']['rows_after_outlier_filter'][split]
                bounds=manifest['bounds']
                np.testing.assert_array_equal(inside,(actual>=bounds['train_price_min'])&(actual<=bounds['train_price_max']))
                if index==0:
                    baseline_arrays[split]=(actual.copy(),data['row_index'].copy())
                    if split=='train': assert training_price_bands(actual)==specification
                else:
                    np.testing.assert_array_equal(actual,baseline_arrays[split][0])
                    np.testing.assert_array_equal(data['row_index'],baseline_arrays[split][1])
                for population,mask in [('all_cleaned',np.ones(len(actual),dtype=bool)),('bounded',inside)]:
                    for row in metric_rows(actual[mask],predicted[mask],specification):
                        recalculated.append(dict(model_id=spec.model_id,split=split,population=population,**row))
    actual_frame=pd.DataFrame(recalculated).set_index(identity).sort_index()
    saved=summary.set_index(identity).sort_index()[actual_frame.columns]
    pd.testing.assert_frame_equal(saved,actual_frame,check_dtype=False,rtol=1e-10,atol=1e-8)
    for _,group in summary.groupby(['model_id','split','population']):
        assert group.loc[group.price_band!='All','n'].sum()==group.loc[group.price_band=='All','n'].iloc[0]
    ranks,selection=validation_assessment(summary)
    assert selection==manifest['selection']
    assert json.loads((out/'selection_before_may.json').read_text())==selection
    assert sha256(out/'selection_before_may.json')==manifest['selection_file_sha256']
    pd.testing.assert_frame_equal(pd.read_csv(out/'validation_comparison.csv',float_precision='round_trip'),ranks,
                                  check_dtype=False,rtol=1e-10,atol=1e-8)
    reproduction=pd.read_csv(out/'historical_reproduction.csv')
    assert len(reproduction)==manifest['historical_reproduction_checks']
    np.testing.assert_allclose(reproduction.observed,reproduction.expected,rtol=1e-8,atol=1e-6)
    # Also check earlier reported May winners, without using those scores in selection.
    for week in ('week05','week06'):
        prior_selection=json.loads((root/'outputs'/f'{week}_selection.json').read_text())
        prior_test=json.loads((root/'outputs'/f'{week}_test_metrics.json').read_text())
        family=prior_selection['winner']
        features=prior_selection.get('feature_set','original')
        model_id='W'+week[-2:]+' '+family+' / '+features
        for population,expected in prior_test.items():
            current=summary.loc[(summary.model_id==model_id)&(summary.split=='test')&
                                (summary.population==population)&(summary.price_band=='All')].iloc[0]
            assert current.n==expected['rows']
            for metric in METRICS:
                np.testing.assert_allclose(current[metric],expected[metric],rtol=1e-8,atol=1e-6)
    figure_names=['training_price_distribution']+[f'{split}_{population}_{chart}'
        for split in ('validation','test') for population in ('all_cleaned','bounded') for chart in CHARTS]
    assert set(p.stem for p in (out/'figures').glob('*.png'))==set(figure_names)
    assert set(p.stem for p in (out/'figures').glob('*.pdf'))==set(figure_names)
    for name in figure_names:
        with Image.open(out/'figures'/(name+'.png')) as im:
            assert im.width>=1000 and im.height>=700
            im.verify()
        with (out/'figures'/(name+'.pdf')).open('rb') as pdf:
            assert pdf.read(5)==b'%PDF-'
    assert (root/'docs/WEEK8_RESULTS.md').stat().st_size>4000
    print(f'PASS: {len(inventory)} pipelines, {len(summary)} metric rows, {len(figure_names)} PNG/PDF pairs; '
          'historical/source/output hashes, historical scores, row coverage and April-only selection')
    return manifest


if __name__=='__main__':
    verify()
