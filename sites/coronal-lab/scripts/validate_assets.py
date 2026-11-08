#!/usr/bin/env python3
"""Validate physical indexing, observed-sample preservation, and exported assets."""
import gzip,json,os
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates
ROOT=Path(__file__).resolve().parents[1];ASSETS=Path(os.environ.get('CORONAL_ASSETS',ROOT/'dist/assets512'));WORK=Path(os.environ.get('CORONAL_WORK',ROOT/'work/full512'))
m=json.loads((ASSETS/'manifest.json').read_text());rng=np.random.default_rng(88);report=[]
for c in m['cases']:
    cid=c['id'];aff=np.array(c['source_affine']);target=np.array(c['target_affine']);inv=np.array(c['inverse_source_affine'])
    assert np.isfinite(aff).all() and abs(np.linalg.det(aff[:3,:3]))>0
    assert np.allclose(aff@inv,np.eye(4))
    source=np.load(WORK/f'{cid}-native.npy');data={}
    original=nib.as_closest_canonical(nib.load(f"/mnt/stor/datasets/BHSD/label_192/images/{c['source_id']}.nii.gz"))
    assert source.shape[1:]==(512,512)
    assert np.array_equal(source,original.get_fdata(dtype=np.float32).transpose(2,1,0))
    assert np.array_equal(aff,original.affine)
    assert c['shape'][2]==512
    assert np.allclose(c['spacing_xyz'][:2],nib.affines.voxel_sizes(original.affine)[:2])
    for method in c['methods']+['source']:
        with gzip.open(ASSETS/f'{cid}-{method}.bin.gz','rb') as f:buf=f.read()
        shape=c['source_shape'] if method=='source' else c['shape']
        assert len(buf)==np.prod(shape)*2,(cid,method,'length')
        data[method]=np.frombuffer(buf,dtype='<i2').reshape(shape)
    assert np.max(np.abs(data['source']-source))<=.501
    # Verify serialized Z,Y,X world voxels against native indexing, independently
    # of the preprocessing coordinate grid and its reshape/transpose operations.
    checked=0;sampled=[]
    for _ in range(200):
        z,y,x=[rng.integers(n) for n in c['shape']]
        src=(inv@target@np.array([x,y,z,1.]))[:3]
        if np.all(src>=0) and np.all(src<np.array(source.shape[::-1])-1):
            sx,sy,sz=np.floor(src+.5).astype(int)
            assert data['nearest'][z,y,x]==round(float(source[sz,sy,sx]))
            checked+=1;sampled.append((z,y,x,src))
    assert checked>20
    errs=[]
    for seed in [88,123]:
        pred=np.load(WORK/f'{cid}-seed{seed}.npy')
        assert np.isfinite(pred).all()
        error=float(np.abs(pred[::4]-source).max());assert error<.001,(cid,seed,error);errs.append(error)
    # Compare both exported ensemble and disagreement to the independently
    # transformed neural seed outputs at physical sample points.
    pts=np.array([v[3][::-1] for v in sampled]).T;pts[0]*=4
    seed_values=[]
    for seed in [88,123]:
        seed_values.append(map_coordinates(np.load(WORK/f'{cid}-seed{seed}.npy'),pts,order=1,mode='nearest'))
    for i,(z,y,x,_) in enumerate(sampled):
        assert abs(float(data['ensemble'][z,y,x])-float((seed_values[0][i]+seed_values[1][i])/2))<=.501
        assert abs(float(data['disagreement'][z,y,x])-float(abs(seed_values[0][i]-seed_values[1][i])))<=.501
    assert data['disagreement'].min()>=0
    assert (ASSETS/f'{cid}-thumb.png').exists()
    delta=np.abs(data['ensemble'].astype(float)-data['linear'].astype(float))
    assert delta.max()>0
    report.append(dict(study=cid,world_index_checks=checked,source_projection_max_error=errs,ai_linear_mean_absolute_difference=float(delta.mean()),ai_seed_mean_absolute_difference=float(data['disagreement'].mean())))
print(json.dumps({'status':'passed','cases':report},indent=2))
(WORK/'asset-validation.json').write_text(json.dumps(report,indent=2))
