#!/usr/bin/env python3
"""Verify physical samples, leakage boundaries, metrics, and actual model outputs."""
import gzip,json
import numpy as np
from scipy.ndimage import map_coordinates
from run_pretrained import ROOT,WORK,ASSETS,metrics,evaluate_setup
m=json.loads((ASSETS/'manifest.json').read_text());rng=np.random.default_rng(1988);report=[]
assert sum(c['evaluation']['heldout_count'] for c in m['cases'])==59
for c in m['cases']:
    cid=c['id'];native,known,held,truth,lesion=evaluate_setup(c)
    assert set(held).isdisjoint(known) and held.min()>known.min() and held.max()<known.max()
    assert c['evaluation']['input_indices']==known.tolist()
    assert c['evaluation']['heldout_indices']==held.tolist()
    aff=np.array(c['source_affine']);target=np.array(c['target_affine'])
    xyz=rng.random((3,50))*np.array(native.shape[::-1])[:,None]*.8
    world=aff[:3,:3]@xyz+aff[:3,3,None]
    targetxyz=(np.linalg.inv(target)[:3,:3]@world+np.linalg.inv(target)[:3,3,None])
    # Verify source-to-world-to-source round trips before testing exported arrays.
    recovered=np.linalg.inv(aff)[:3,:3]@world+np.linalg.inv(aff)[:3,3,None]
    assert np.max(np.abs(recovered-xyz))<1e-8
    per_method={}
    for method in ['qdm','esrgan','esrnet']:
        if method in m.get('pretrained_pending',[]):continue
        full=np.load(WORK/f'{cid}-{method}-full.npy',mmap_mode='r');raw=np.load(WORK/f'{cid}-{method}-full-raw.npy',mmap_mode='r')
        assert np.isfinite(full).all() and np.isfinite(raw).all()
        error=float(np.max(np.abs(full[::4]-native)));assert error<.002
        conditioned=np.load(WORK/f'{cid}-{method}-withheld.npy',mmap_mode='r')
        assert np.max(np.abs(conditioned[::4]-native[known]))<.002
        score=metrics(conditioned[2*held],truth,lesion)
        for k,v in score.items():assert np.isclose(v,c['evaluation']['methods'][method][k]),(cid,method,k)
        for suffix,pred in [('',full),('_raw',raw)]:
            with gzip.open(ASSETS/f'{cid}-{method}{suffix}.bin.gz','rb') as f:
                data=np.frombuffer(f.read(),dtype='<i2').reshape(c['shape'])
            # Sample exact integer display voxels, then transform to model coordinates.
            checked=0
            for _ in range(50):
                z,y,x=[rng.integers(n) for n in c['shape']]
                q=(np.linalg.inv(aff)@target@np.array([x,y,z,1]))[:3]
                if np.all(q>=0) and np.all(q<np.array(native.shape[::-1])-1):
                    modelq=np.array([[q[2]*4],[q[1]],[q[0]]]);expected=map_coordinates(pred,modelq,order=1,mode='nearest')[0]
                    assert abs(float(data[z,y,x])-float(expected))<.502,(cid,method,suffix)
                    checked+=1
            assert checked>10
        with gzip.open(ASSETS/f'{cid}-eval-{method}.bin.gz','rb') as f:export=np.frombuffer(f.read(),dtype='<i2').reshape(truth.shape)
        assert np.max(np.abs(export-conditioned[held*2]))<=.501
        per_method[method]=dict(source_max_error=error,withheld_metrics_recomputed=True,exported_physical_coordinates_verified=True)
    report.append(dict(study=cid,withheld=len(held),hemorrhage_voxels=int(lesion.sum()),models=per_method))
(WORK/'pretrained-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':'passed','cases':report},indent=2))
