"""Deterministic intrinsic features and cardinality-preserving school geography."""
from pathlib import Path
import hashlib
import json
from urllib.request import urlopen
from urllib.parse import urlencode
import numpy as np
import pandas as pd

SERVICE='https://services3.arcgis.com/fdvHcZVgB2QSRNkL/arcgis/rest/services/SchoolDistrictAreas2425/FeatureServer/0'
SOURCE='https://data.ca.gov/dataset/california-school-district-areas-2024-25/resource/7dfaf005-58eb-45db-93b1-7aff091b2172'
ENGINEERED=['PropertyAge','BedBathRatio','LivingAreaPerBedroom','LogLivingArea','LogLotSize','LatitudeLongitude','LatitudeSquared','LongitudeSquared']
DISTRICTS=['SchoolDistrictElementary','SchoolDistrictSecondary','SchoolDistrictUnified']

def engineer(frame, valuation_year=2026):
    """Fixed valuation year is an explicit prediction-time input, never CloseDate."""
    out=frame.copy()
    def num(c):
        return pd.to_numeric(frame[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
    year=num('YearBuilt'); beds=num('BedroomsTotal'); baths=num('BathroomsTotalInteger')
    area=num('LivingArea'); lot=num('LotSizeSquareFeet')
    out['PropertyAge']=(valuation_year-year).where(year.between(1600,valuation_year))
    out['BedBathRatio']=(beds/baths.where(baths>0)).where(beds>=0)
    out['LivingAreaPerBedroom']=(area/beds.where(beds>0)).where(area>0)
    out['LogLivingArea']=np.log1p(area.where(area>0))
    out['LogLotSize']=np.log1p(lot.where(lot>=0))
    lat=num('Latitude').where(num('Latitude').between(-90,90))
    lon=num('Longitude').where(num('Longitude').between(-180,180))
    # Center on fixed California reference; no learned geographic statistics.
    out['LatitudeLongitude']=(lat-37)*(lon+120)
    out['LatitudeSquared']=(lat-37)**2
    out['LongitudeSquared']=(lon+120)**2
    return out

def download_districts(directory):
    """Only public boundary requests are sent; never send property coordinates."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    def get(url):
        with urlopen(url,timeout=180) as response: return json.load(response)
    metadata=get(SERVICE+'?f=pjson')
    ids=get(SERVICE+'/query?'+urlencode(dict(where='1=1',returnIdsOnly='true',f='json')))['objectIds']
    features=[]
    for start in range(0,len(ids),100):
        batch=get(SERVICE+'/query?'+urlencode(dict(objectIds=','.join(map(str,ids[start:start+100])),outFields='OBJECTID,Year,CDSCode,DistrictName,DistrictType',returnGeometry='true',outSR=4326,f='geojson')))
        if 'error' in batch: raise RuntimeError(batch['error'])
        features.extend(batch['features'])
    if len(features)!=len(ids): raise ValueError('Incomplete district download')
    path=directory/'districts_2024_25.geojson'
    path.write_text(json.dumps(dict(type='FeatureCollection',features=features)),encoding='utf8')
    provenance=dict(source=SOURCE,service=SERVICE,native_crs=metadata['extent']['spatialReference'],download_crs='EPSG:4326',
                    rows=len(features),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    retrieved_utc=pd.Timestamp.now(tz='UTC').isoformat(),last_edit=metadata.get('editingInfo'),
                    warning='2024-25 boundaries retrieved retrospectively; historical availability not established')
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    (directory/'service_schema.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    return path

def spatial_join(frame, districts):
    """Separate district levels; same-level ambiguities stay unknown, not arbitrary."""
    import geopandas as gpd
    if districts.crs is None: raise ValueError('District CRS must be declared')
    required={'DistrictType','CDSCode','geometry'}
    if not required.issubset(districts.columns): raise ValueError('Missing district schema fields')
    districts=districts.to_crs('EPSG:4326').copy()
    if districts.geometry.isna().any() or districts.geometry.is_empty.any():
        raise ValueError('Empty district polygons; inspect source before joining')
    invalid=~districts.geometry.is_valid
    repair_ids=districts.loc[invalid,'CDSCode'].astype(str).tolist()
    if invalid.any():
        # Source has seven self-intersecting polygons. Explicit GEOS structure
        # repair preserves polygonal pieces, dropping collapsed lines only.
        districts.loc[invalid,'geometry']=districts.loc[invalid].geometry.make_valid(method='structure',keep_collapsed=False)
    if not districts.geometry.is_valid.all() or districts.geometry.is_empty.any():
        raise ValueError('District topology repair failed')
    # Source labels are checked explicitly instead of guessing classifications.
    districts['DistrictType']=districts.DistrictType.replace({'High':'Secondary'})
    types={'Elementary':DISTRICTS[0],'Secondary':DISTRICTS[1],'Unified':DISTRICTS[2]}
    unknown=set(districts.DistrictType.dropna().unique())-set(types)
    if unknown: raise ValueError(f'Unrecognized district types: {unknown}')
    lat=pd.to_numeric(frame.Latitude,errors='coerce'); lon=pd.to_numeric(frame.Longitude,errors='coerce')
    valid=lat.between(-90,90)&lon.between(-180,180)
    positions=np.flatnonzero(valid.to_numpy())
    points=gpd.GeoDataFrame({'_row':positions},geometry=gpd.points_from_xy(lon.iloc[positions],lat.iloc[positions]),crs=4326)
    out=frame.copy(); audit=dict(rows=len(frame),invalid_coordinates=int((~valid).sum()),district_crs=str(districts.crs),types=districts.DistrictType.value_counts().to_dict(),repaired_polygon_ids=repair_ids,repair_method='GEOS make_valid structure; keep_collapsed=False')
    for kind,col in types.items():
        polygons=districts.loc[districts.DistrictType.eq(kind),['CDSCode','geometry']].copy()
        polygons['CDSCode']=polygons.CDSCode.astype('string')
        joined=gpd.sjoin(points,polygons,how='inner',predicate='intersects')
        matches=joined.groupby('_row').CDSCode.agg(lambda s: sorted(set(s.dropna())))
        values=np.full(len(frame),np.nan,dtype=object)
        for row,ids in matches.items():
            if len(ids)==1: values[row]=ids[0]
        out[col]=values
        audit[col]=dict(matched=int(pd.notna(values).sum()),ambiguous=int(matches.map(len).gt(1).sum()))
    audit['no_district_match']=int(out[DISTRICTS].isna().all(axis=1).sum())
    assert len(out)==len(frame) and out.index.equals(frame.index)
    return out,audit
