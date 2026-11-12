"""Equal-probability OOF ensembles of final BHSD segmentation teachers."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/bhsd_teacher_ensemble_20261110'
ROOTS={'convnext':ROOT/'experiments/bhsd_convnext_s16_binary_ablations_20261110/p100_batch',**{name:ROOT/'experiments/bhsd_teacher_backbones_20261110'/folder for name,folder in [('maxvit','maxvit_tiny'),('coatnet','coatnet0'),('effnet','effnetv2m')]}}
ENSEMBLES={'three':['convnext','maxvit','coatnet'],'four':['convnext','maxvit','coatnet','effnet']}
THRESHOLDS=np.arange(1,10)/10


def evaluate_counts(counts,df,fold,model):
    positive=counts[:,0,2]>0
    assert np.array_equal(positive,df.has_mask.to_numpy())
    slice_dice=2*counts[:,:,0]/np.maximum(counts[:,:,1]+counts[:,:,2],1)
    volume=[];rows=[]
    for uid,indices in df.groupby('series_uid').indices.items():
        raw=counts[indices].sum(0)
        gated=counts[indices].copy();gated[~positive[indices],:,1]=0;gated=gated.sum(0)
        assert (raw[:,2]>0).all()
        ungated_dice=2*raw[:,0]/(raw[:,1]+raw[:,2])
        gated_dice=2*gated[:,0]/(gated[:,1]+gated[:,2])
        assert (gated_dice>=ungated_dice-1e-12).all()
        volume.append(np.stack([ungated_dice,gated_dice],axis=1))
        for j,t in enumerate(THRESHOLDS):
            rows.append(dict(model=model,fold=fold,patient_id=df.iloc[indices[0]].patient_id,series_uid=uid,threshold=t,volume_dice=ungated_dice[j],gated_volume_dice=gated_dice[j]))
    mean=np.mean(volume,axis=0)
    summaries=[dict(model=model,fold=fold,threshold=t,positive_slice_dice=slice_dice[positive,j].mean(),volume_dice=mean[j,0],gated_volume_dice=mean[j,1],negative_slice_fp_fraction=(counts[~positive,j,1]>0).mean()) for j,t in enumerate(THRESHOLDS)]
    return summaries,rows


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    summary=[];series=[];provenance=[]
    for fold in range(5):
        dest=OUT/f'fold{fold}';dest.mkdir(exist_ok=True)
        arrays={};manifest=None
        for name,root in ROOTS.items():
            run=root/f'fold{fold}'
            df=pd.read_csv(run/'validation_manifest.csv')
            train=pd.read_csv(run/'train_manifest.csv')
            assert set(train.patient_id).isdisjoint(df.patient_id)
            meta=json.loads((run/'validation_predictions_final.json').read_text())
            status=json.loads((run/'status.json').read_text())
            assert status['status']=='complete' and meta['epoch']==25 and meta['step']==meta['total_steps']
            assert meta['checkpoint']=='last.pt'
            if manifest is None:manifest=df
            else:pd.testing.assert_frame_equal(df,manifest)
            path=run/'validation_predictions_final.npy'
            arrays[name]=np.load(path,mmap_mode='r')
            assert arrays[name].shape==(len(df),512,512)
            provenance.append(dict(model=name,fold=fold,predictions=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,metadata_sha256=hashlib.sha256((run/'validation_predictions_final.json').read_bytes()).hexdigest()))
        manifest.to_csv(dest/'validation_manifest.csv',index=False)
        counts={name:np.zeros((len(manifest),9,3),np.int64) for name in [*ROOTS,*ENSEMBLES]}
        outputs={name:np.lib.format.open_memmap(dest/f'{name}_probabilities.npy',mode='w+',dtype=np.float16,shape=(len(manifest),512,512)) for name in ENSEMBLES}
        masks={uid:np.load(rows.iloc[0].mask_volume_path,mmap_mode='r') for uid,rows in manifest.groupby('series_uid')}
        for start in range(0,len(manifest),16):
            end=min(start+16,len(manifest))
            rows=manifest.iloc[start:end]
            target=torch.from_numpy(np.stack([masks[r.series_uid][...,int(r.slice_index)]>0 for r in rows.itertuples()])).cuda()
            predictions={k:torch.from_numpy(np.array(a[start:end],dtype=np.float32)).cuda() for k,a in arrays.items()}
            assert all(torch.isfinite(p).all() and p.min()>=0 and p.max()<=1 for p in predictions.values())
            for name,members in ENSEMBLES.items():
                predictions[name]=sum(predictions[m] for m in members)/len(members)
                outputs[name][start:end]=predictions[name].cpu().numpy().astype(np.float16)
            target_sum=target.sum((1,2))
            for name,p in predictions.items():
                for j,t in enumerate(THRESHOLDS):
                    pred=p>=float(t)
                    counts[name][start:end,j]=torch.stack(((pred&target).sum((1,2)),pred.sum((1,2)),target_sum),dim=1).cpu().numpy()
        for name,out in outputs.items():out.flush()
        outputs.clear()
        for name,c in counts.items():
            np.save(dest/f'{name}_counts.npy',c)
            s,v=evaluate_counts(c,manifest,fold,name);summary+=s;series+=v
            if name in ROOTS:
                meta=json.loads((ROOTS[name]/f'fold{fold}'/'validation_predictions_final.json').read_text())
                assert abs(s[4]['positive_slice_dice']-meta['positive_slice_dice_05'])<.002
                assert abs(s[4]['volume_dice']-meta['volume_dice_05'])<.002
        print(f'Completed fold {fold}',flush=True)
    folds=pd.DataFrame(summary);volumes=pd.DataFrame(series)
    folds.to_csv(OUT/'fold_metrics.csv',index=False);volumes.to_csv(OUT/'series_metrics.csv',index=False)
    metrics=['positive_slice_dice','volume_dice','gated_volume_dice','negative_slice_fp_fraction']
    means=folds.groupby(['model','threshold'])[metrics].mean().reset_index()
    means.to_csv(OUT/'summary.csv',index=False)
    # Paired bootstrap by patient, retaining the one patient's two series.
    ci=[]
    for t in [.5,.6,.7,.8]:
        v=volumes[np.isclose(volumes.threshold,t)]
        pivot=v.pivot(index=['patient_id','series_uid'],columns='model',values='gated_volume_dice')
        assert len(pivot)==192 and not pivot.isna().any().any()
        for a,b in [('three','four'),('three','convnext'),('four','convnext'),('three','maxvit')]:
            diff=pivot[a]-pivot[b];sums=diff.groupby(level=0).sum();sizes=diff.groupby(level=0).size()
            draws=np.random.default_rng(88).integers(0,len(sums),size=(10000,len(sums)))
            boot=sums.to_numpy()[draws].sum(1)/sizes.to_numpy()[draws].sum(1)
            lo,hi=np.quantile(boot,[.025,.975]);ci.append(dict(threshold=t,a=a,b=b,pooled_series_delta=diff.mean(),ci_low=lo,ci_high=hi))
    pd.DataFrame(ci).to_csv(OUT/'paired_bootstrap.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(sources=provenance,ensembles=ENSEMBLES,method='Equal arithmetic mean of saved float16 probabilities, computed in float32. Same held-out fold per family; no in-fold teacher predictions. 5-fold macro means. Gating empties known-negative center slices. Threshold sweep is descriptive, not independent threshold validation. Bootstrap paired by patient,10000 seed88,pooled series means; no multiple-comparison adjustment. Individual models recomputed from same float16 exports for fairness; metrics checked against original float32-derived logs within0.002. No inference or training.',thresholds=THRESHOLDS.tolist()),indent=2))
    print(means[np.isclose(means.threshold,.5)].round(5).to_string(index=False))
    print('Best gated-volume thresholds (descriptive):')
    print(means.loc[means.groupby('model').gated_volume_dice.idxmax()].round(5).to_string(index=False))
    print(pd.DataFrame(ci).round(5).to_string(index=False))


if __name__=='__main__':main()
