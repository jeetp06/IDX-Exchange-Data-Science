"""Only synthetic, generated test cases; not source MLS performance estimates."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import avm


def synthetic_sales(seed=42):
    rng=np.random.default_rng(seed)
    dates=pd.date_range('2026-01-02','2026-08-27',freq='3D')
    n=len(dates)
    area=rng.integers(950,3800,n)
    bedrooms=rng.integers(2,6,n)
    baths=rng.integers(1,5,n)
    rows=pd.DataFrame({
        'PropertyType':'Residential',
        'PropertySubType':'SingleFamilyResidence',
        'CloseDate':dates,
        'ListingContractDate':dates-pd.Timedelta(days=35),
        'LivingArea':area,
        'BedroomsTotal':bedrooms,
        'BathroomsTotalInteger':baths,
        'LotSizeSquareFeet':rng.integers(2400,16000,n),
        'YearBuilt':rng.integers(1950,2025,n),
        'Latitude':rng.uniform(33.5,34.7,n),
        'Longitude':rng.uniform(-119,-117,n),
        'PostalCode':rng.choice(['90001','90210','91311'],n),
        'CountyOrParish':rng.choice(['Los Angeles','Ventura'],n),
        'ListingKey':[str(i) for i in range(n)],
        'ParcelNumber':[f'P{i}' for i in range(n)],
        'ClosePrice':(220000+area*230+bedrooms*17000+baths*14000+rng.normal(0,45000,n)).round(),
    })
    # A leakage trap: asking price tracks the target very closely but must not be a feature.
    rows['ListPrice']=rows['ClosePrice']*1.01
    rows['OriginalListPrice']=rows['ClosePrice']*1.1
    rows['DaysOnMarket']=25
    rows.loc[0,'LotSizeSquareFeet']=np.nan
    return rows


class PipelineTests(unittest.TestCase):
    def test_clean_split_pipeline_and_outputs(self):
        data=synthetic_sales()
        bad=data.iloc[[0]].copy(); bad['ClosePrice']=0; bad['ListingKey']='broken'
        data=pd.concat([data,data.iloc[[5]],bad],ignore_index=True)
        clean,counts=avm.clean_properties(data)
        self.assertEqual(len(clean),len(synthetic_sales()))
        self.assertGreater(counts['raw'],counts['deduplicated_transactions'])
        self.assertFalse(clean['ClosePrice'].le(0).any())
        splits,info=avm.chronological_split(clean,'2026-10-12')
        self.assertEqual(info['test_month'],'2026-08')
        self.assertEqual(info['validation_month'],'2026-07')
        self.assertTrue(splits['train']['CloseDate'].max() < splits['validation']['CloseDate'].min())
        bounded,policy=avm.price_outlier_filter(splits)
        self.assertLessEqual(policy['train_price_min'],policy['train_price_max'])
        num,cat=avm.choose_features(bounded['train'])
        self.assertFalse((set(num+cat) & avm.FORBIDDEN_FEATURES))
        with tempfile.TemporaryDirectory() as d:
            model=avm.train_baseline(bounded,d,evaluate_test=False)
            self.assertIn('validation',model)
            self.assertNotIn('test',model)
            self.assertTrue((Path(d)/'baseline_pipeline.joblib').exists())
            self.assertTrue(np.isfinite(model['validation']['R2']))
            full=avm.train_baseline(bounded,d,evaluate_test=True)
            self.assertIn('test',full)
            avm.explore(clean,d)
            self.assertTrue((Path(d)/'figures'/'ClosePrice_distribution.png').exists())
            self.assertTrue((Path(d)/'eda_missingness.csv').exists())

    def test_staged_cli_with_synthetic_csv(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'raw'; raw.mkdir()
            synthetic_sales().to_csv(raw/'CRMLSSold_SYNTHETIC_TEST_ONLY.csv',index=False)
            result=avm.run(raw,Path(d)/'out',as_of='2026-10-12',through='train')
            self.assertIn('model',result)
            self.assertTrue((Path(d)/'out'/'california_single_family_cleaned.csv').is_file())
            self.assertTrue((Path(d)/'out'/'data_split_audit.json').is_file())

    def test_missing_data(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError): avm.find_files(Path(d)/'none')
        df=synthetic_sales().copy()
        df.loc[:,'ClosePrice']=None
        with self.assertRaisesRegex(ValueError,'No valid'):
            avm.clean_properties(df)

if __name__=='__main__':
    unittest.main()
