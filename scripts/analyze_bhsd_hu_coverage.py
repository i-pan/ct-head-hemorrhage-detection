"""Annotation-only HU coverage audit; never reads teacher predictions."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/bhsd_hu_coverage_20261110'
MANIFEST=ROOT/'data/bhsd/bhsd_segmentation_slices.csv'
LOWERS=[0,20,30,35,40,45,50]
UPPERS=[80,100,120]
BANDS=[(lo,hi) for hi in UPPERS for lo in LOWERS]
# Integer HU boundaries inclusive on both ends, also correct for fractional HU.
EDGES=np.array(sorted(set([-np.inf,np.inf]+LOWERS+[float(np.nextafter(np.float32(hi),np.float32(np.inf))) for hi in UPPERS])),dtype=np.float64)
NB=len(EDGES)-1

def band_counts(hist):
    return np.stack([hist[..., (EDGES[:-1]>=lo)&(EDGES[1:]<=float(np.nextafter(np.float32(hi),np.float32(np.inf))))].sum(-1) for lo,hi in BANDS],axis=-1)

def group_hist(group,hu,n):
    bins=np.searchsorted(EDGES,hu,side='right')-1
    return np.bincount(group*NB+bins,minlength=n*NB).reshape(n,NB)

def main():
    OUT.mkdir(exist_ok=True)
    df=pd.read_csv(MANIFEST)
    whole=np.zeros(NB,np.int64);positive_images=np.zeros(NB,np.int64)
    bleed=np.zeros(NB,np.int64);subtypes=np.zeros((6,NB),np.int64)
    slices=[];lesions=[];studies=[]
    for number,(uid,rows) in enumerate(df.groupby('series_uid'),1):
        hu=np.load(ROOT/'data/bhsd/hu_volumes'/f'{uid}.npy',mmap_mode='r')
        label=np.ascontiguousarray(np.load(rows.iloc[0].mask_volume_path,mmap_mode='r').transpose(2,0,1))
        target=label>0
        assert hu.shape==target.shape and np.isfinite(hu).all()
        assert np.array_equal(target.any((1,2)),rows.sort_values('slice_index').has_mask.to_numpy())
        whole+=np.histogram(hu,bins=EDGES)[0]
        positive_images+=np.histogram(hu[target.any((1,2))],bins=EDGES)[0]
        z=np.nonzero(target)[0];values=hu[target]
        poshist=group_hist(z,values,len(hu));bleed+=poshist.sum(0)
        subtypes+=group_hist(label[target],values,6)
        pixel_mm2=float(rows.iloc[0].row_spacing_mm*rows.iloc[0].col_spacing_mm)
        voxel_ml=pixel_mm2*float(rows.iloc[0].slice_spacing_mm)/1000
        counts=band_counts(poshist)
        for i,n in enumerate(poshist.sum(1)):
            if n:
                r=dict(series_uid=uid,slice_index=i,pixels=int(n),area_mm2=float(n*pixel_mm2))
                r.update({f'coverage_{lo}_{hi}':float(counts[i,j]/n) for j,(lo,hi) in enumerate(BANDS)})
                slices.append(r)
        cc,ncomp=ndimage.label(target,structure=np.ones((3,3,3),bool))
        component_hist=group_hist(cc[target]-1,values,ncomp)
        component_counts=band_counts(component_hist)
        for i,n in enumerate(component_hist.sum(1)):
            r=dict(series_uid=uid,component=i+1,pixels=int(n),volume_ml=float(n*voxel_ml))
            r.update({f'coverage_{lo}_{hi}':float(component_counts[i,j]/n) for j,(lo,hi) in enumerate(BANDS)})
            lesions.append(r)
        sh=poshist.sum(0);sc=band_counts(sh)
        r=dict(series_uid=uid,hemorrhage_ml=float(sh.sum()*voxel_ml),positive_pixels=int(sh.sum()))
        r.update({f'coverage_{lo}_{hi}':float(sc[j]/sh.sum()) for j,(lo,hi) in enumerate(BANDS)})
        studies.append(r)
        if number%32==0: print(f'{number}/192 series',flush=True)
    sl=pd.DataFrame(slices);le=pd.DataFrame(lesions);st=pd.DataFrame(studies)
    sl.to_csv(OUT/'positive_slices.csv',index=False);le.to_csv(OUT/'components_3d.csv',index=False);st.to_csv(OUT/'series.csv',index=False)
    rows=[]
    for j,(lo,hi) in enumerate(BANDS):
        key=f'coverage_{lo}_{hi}'
        rows.append(dict(lower=lo,upper=hi,bleed_pixel_coverage=float(band_counts(bleed)[j]/bleed.sum()),all_image_fraction=float(band_counts(whole)[j]/whole.sum()),positive_image_fraction=float(band_counts(positive_images)[j]/positive_images.sum()),positive_slice_macro_coverage=float(sl[key].mean()),series_macro_coverage=float(st[key].mean()),lesion_macro_coverage=float(le[key].mean()),lesion_zero_coverage_fraction=float((le[key]==0).mean())))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'bands.csv',index=False)
    strata=[]
    for name,data,size,bins in [('component',le,'volume_ml',[0,.1,1,5,np.inf]),('positive_slice',sl,'area_mm2',[0,10,50,200,np.inf]),('series',st,'hemorrhage_ml',[0,1,5,20,np.inf])]:
        for low,high in zip(bins[:-1],bins[1:]):
            selected=data[(data[size]>low)&(data[size]<=high)]
            for lo,hi in BANDS:
                key=f'coverage_{lo}_{hi}'
                strata.append(dict(unit=name,size_lower=low,size_upper=high,n=len(selected),hu_lower=lo,hu_upper=hi,macro_coverage=float(selected[key].mean()),median_coverage=float(selected[key].median()),p10_coverage=float(selected[key].quantile(.1)),zero_coverage_fraction=float((selected[key]==0).mean()),below50pct_fraction=float((selected[key]<.5).mean())))
    pd.DataFrame(strata).to_csv(OUT/'size_strata.csv',index=False)
    subtype_rows=[]
    for c,name in enumerate(['background','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']):
        if c==0:continue
        for j,(lo,hi) in enumerate(BANDS):
            subtype_rows.append(dict(subtype=name,pixels=int(subtypes[c].sum()),lower=lo,upper=hi,coverage=float(band_counts(subtypes[c])[j]/subtypes[c].sum())))
    pd.DataFrame(subtype_rows).to_csv(OUT/'subtypes.csv',index=False)
    result=dict(series=len(st),patients=df.patient_id.nunique(),slices=len(df),positive_slices=len(sl),components_3d=len(le),positive_pixels=int(bleed.sum()),total_pixels=int(whole.sum()),manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),method='Original reoriented NIfTI HU and manual binary BHSD masks; no teacher predictions. Inclusive bands. Whole-image region size includes extracranial tissue/air; no brain mask. Lesions are 26-connected 3D binary components, not adjudicated individual lesions. Coverage macro averaged within size bins; pooled pixel coverage separately. All BHSD used descriptively; not an independent evaluation of a chosen HU range.')
    (OUT/'provenance.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result));print(summary.to_string(index=False))

if __name__=='__main__':main()
