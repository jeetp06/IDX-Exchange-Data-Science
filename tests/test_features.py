"""Synthetic calculation and spatial edge cases; not real-data accuracy tests."""
import unittest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from src.features import engineer, spatial_join, DISTRICTS, ENGINEERED
from src import experiments, avm

class FeatureTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(YearBuilt=[2000,2030,np.nan],BedroomsTotal=[4,0,2],BathroomsTotalInteger=[2,0,-1],LivingArea=[2000,-2,np.inf],LotSizeSquareFeet=[0,-1,100],Latitude=[34,34,100],Longitude=[-118,-118,-118]))
    def test_calculations_and_leakage(self):
        frame=self.frame(); out=engineer(frame)
        self.assertEqual(out.PropertyAge.iloc[0],26)
        self.assertTrue(out.PropertyAge.iloc[1:].isna().all())
        self.assertEqual(out.BedBathRatio.iloc[0],2)
        self.assertTrue(out.BedBathRatio.iloc[1:].isna().all())
        self.assertEqual(out.LivingAreaPerBedroom.iloc[0],500)
        self.assertTrue(out.LivingAreaPerBedroom.iloc[1:].isna().all())
        self.assertAlmostEqual(out.LogLivingArea.iloc[0],np.log1p(2000))
        self.assertEqual(out.LogLotSize.iloc[0],0)
        self.assertTrue(np.isnan(out.LatitudeSquared.iloc[2]))
        changed=frame.assign(ClosePrice=999,CloseDate='2040-01-01',ListPrice=1,DaysOnMarket=400)
        pd.testing.assert_frame_equal(out[ENGINEERED],engineer(changed)[ENGINEERED])
        self.assertFalse(set(ENGINEERED+DISTRICTS)&avm.FORBIDDEN_FEATURES)

    def polygons(self):
        return gpd.GeoDataFrame(dict(DistrictType=['Elementary','Secondary','Unified'],CDSCode=['E','S','U']),geometry=[box(-119,33,-117,35)]*3,crs=4326)
    def test_overlap_levels_crs_and_unmatched(self):
        frame=self.frame(); frame.index=[9,9,2]
        frame.loc[2,'Latitude']=40
        out,audit=spatial_join(frame,self.polygons().to_crs(3857))
        self.assertEqual(len(out),3); self.assertTrue(out.index.equals(frame.index))
        self.assertEqual(out[DISTRICTS].iloc[0].tolist(),['E','S','U'])
        self.assertTrue(out[DISTRICTS].iloc[2].isna().all())
        self.assertEqual(audit['no_district_match'],1)
    def test_same_level_ambiguity_and_boundary(self):
        polygons=self.polygons()
        extra=polygons.iloc[[0]].assign(CDSCode='E2')
        polygons=pd.concat([polygons,extra],ignore_index=True)
        frame=self.frame().iloc[[0]].copy(); frame.Longitude=-119
        out,audit=spatial_join(frame,polygons)
        self.assertTrue(out[DISTRICTS[0]].isna().all())
        self.assertEqual(audit[DISTRICTS[0]]['ambiguous'],1)
        self.assertEqual(out[DISTRICTS[1]].iloc[0],'S')
    def test_missing_crs_and_unknown_schema_rejected(self):
        p=self.polygons().set_crs(None,allow_override=True)
        with self.assertRaisesRegex(ValueError,'CRS'): spatial_join(self.frame(),p)
        p=self.polygons(); p.loc[0,'DistrictType']='Unexpected'
        with self.assertRaisesRegex(ValueError,'Unrecognized'): spatial_join(self.frame(),p)
    def test_high_type_and_invalid_polygon_repair(self):
        from shapely.geometry import Polygon
        p=self.polygons(); p.loc[1,'DistrictType']='High'
        p.loc[0,'geometry']=Polygon([(-119,33),(-117,35),(-117,33),(-119,35),(-119,33)])
        out,audit=spatial_join(self.frame(),p)
        self.assertEqual(audit['repaired_polygon_ids'],['E'])
        self.assertEqual(out[DISTRICTS[1]].iloc[0],'S')
    def test_preprocessor_train_only_and_whitelist(self):
        frame=self.frame().assign(PostalCode=['a','b','c'],ListPrice=10,ClosePrice=100)
        num=['LivingArea']; cat=['PostalCode']
        x=experiments.inputs(frame,num,cat)
        self.assertEqual(list(x),num+cat)
        model=experiments.pipeline('Decision Tree',num,cat,dict(max_depth=2))
        model.fit(x.iloc[:2],[100,200])
        before=model['preprocess'].named_transformers_['numeric']['impute'].statistics_.copy()
        model.predict(x.iloc[[2]])
        np.testing.assert_equal(before,model['preprocess'].named_transformers_['numeric']['impute'].statistics_)

if __name__=='__main__': unittest.main()
