"""Final-checkpoint OOF volume Dice with known-negative slices gated to empty."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'diagnostics/bhsd_negative_gating_20261110'
BASE = ROOT / 'experiments/bhsd_convnext_s16_binary_schedules_20261110/passes_25'
ABL = ROOT / 'experiments/bhsd_convnext_s16_binary_ablations_20261110'
ARMS = {'p100_slice': BASE, **{a: ABL/a for a in ['p100_batch','p075_slice','p075_batch','p050_slice','p050_batch']}}
THRESHOLDS = np.arange(1,10)/10


def main():
    OUT.mkdir(exist_ok=True)
    volume_rows=[]; fold_rows=[]; sources=[]; reference={}
    for arm, path in ARMS.items():
        for fold in range(5):
            run=path/f'fold{fold}'
            df=pd.read_csv(run/'validation_manifest.csv')
            meta=json.loads((run/'validation_predictions_final.json').read_text())
            counts=np.load(run/'validation_counts_final.npy')
            assert meta['epoch']==25 and meta['step']==meta['total_steps']
            assert counts.shape==(len(df),9,3)
            assert np.all(counts[:,:,2]==counts[:,[0],2])
            positive=counts[:,0,2]>0
            assert np.array_equal(positive,df.has_mask.to_numpy())
            assert not counts[~positive,:,0].any()
            if fold not in reference:reference[fold]=df
            else:pd.testing.assert_frame_equal(df,reference[fold])
            chosen=int(np.argmin(abs(THRESHOLDS-meta['best_threshold'])))
            dice=2*counts[:,:,0]/np.maximum(counts[:,:,1]+counts[:,:,2],1)
            assert np.isclose(dice[positive,4].mean(),meta['positive_slice_dice_05'])
            gated=counts.copy();gated[~positive,:,1]=0
            raw_volumes=[];gated_volumes=[]
            for uid,indices in df.groupby('series_uid').indices.items():
                rows=df.iloc[indices]
                assert rows.patient_id.nunique()==1
                raw=counts[indices].sum(0);g=gated[indices].sum(0)
                assert (raw[:,2]>0).all()
                before=2*raw[:,0]/(raw[:,1]+raw[:,2])
                after=2*g[:,0]/(g[:,1]+g[:,2])
                assert (after>=before-1e-12).all()
                raw_volumes.append(before);gated_volumes.append(after)
                for j,t in enumerate(THRESHOLDS):
                    volume_rows.append(dict(arm=arm,fold=fold,patient_id=rows.iloc[0].patient_id,series_uid=uid,threshold=t,selected_threshold=bool(j==chosen),ungated_dice=before[j],gated_dice=after[j],negative_predicted_pixels=raw[j,1]-g[j,1]))
            before=np.mean(raw_volumes,axis=0);after=np.mean(gated_volumes,axis=0)
            assert np.isclose(before[4],meta['volume_dice_05'])
            assert np.isclose(before[chosen],meta['volume_dice_at_slice_best'])
            fold_rows.append(dict(arm=arm,fold=fold,epoch=25,step=meta['step'],positive_dice_05=meta['positive_slice_dice_05'],positive_dice_selected=meta['positive_slice_dice_best'],threshold=meta['best_threshold'],ungated_05=before[4],gated_05=after[4],ungated_selected=before[chosen],gated_selected=after[chosen]))
            sources.append(dict(arm=arm,fold=fold,run=str(run),counts_sha256=hashlib.sha256((run/'validation_counts_final.npy').read_bytes()).hexdigest()))
    folds=pd.DataFrame(fold_rows);volumes=pd.DataFrame(volume_rows)
    folds.to_csv(OUT/'fold_results.csv',index=False);volumes.to_csv(OUT/'series_threshold_results.csv',index=False)
    columns=['positive_dice_05','positive_dice_selected','ungated_05','gated_05','ungated_selected','gated_selected']
    summary=folds.groupby('arm')[columns].mean()
    summary['gating_gain_05']=summary.gated_05-summary.ungated_05
    summary['gated_delta_vs_baseline_05']=summary.gated_05-summary.loc['p100_slice','gated_05']
    summary.to_csv(OUT/'summary.csv')
    # Paired patient-cluster bootstrap of pooled (not equally fold-weighted) series means.
    # Includes the one patient with two studies; no independent-resampling of that pair.
    cis=[]
    for rule in ['0.5','selected']:
        v=volumes[np.isclose(volumes.threshold,.5)] if rule=='0.5' else volumes[volumes.selected_threshold]
        pivot=v.pivot(index=['patient_id','series_uid'],columns='arm',values='gated_dice')
        assert len(pivot)==192 and not pivot.isna().any().any()
        for comparison in ['p100_slice','p100_batch']:
            diff=pivot.subtract(pivot[comparison],axis=0)
            sums=diff.groupby(level='patient_id').sum();sizes=diff.groupby(level='patient_id').size()
            rng=np.random.default_rng(88);draws=rng.integers(0,len(sums),size=(10000,len(sums)))
            boot=sums.to_numpy()[draws].sum(1)/sizes.to_numpy()[draws].sum(1)[:,None]
            for j,arm in enumerate(diff.columns):
                low,high=np.quantile(boot[:,j],[.025,.975])
                cis.append(dict(threshold_rule=rule,arm=arm,reference=comparison,pooled_series_mean_delta=diff[arm].mean(),ci_low=low,ci_high=high))
    pd.DataFrame(cis).to_csv(OUT/'paired_patient_bootstrap.csv',index=False)
    (OUT/'provenance.json').write_text(json.dumps(dict(checkpoint='final, pass25, all30folds',source='Saved float32-derived overlap counts; no model inference or prediction alteration',gating='Set predicted pixel count to zero on manually labelled negative slices. Positive slices unchanged.',summary='Equal-weight five-fold mean, matching training reports. Each fold volume metric is macro mean over series.',thresholds='0.5 and original positive-slice-Dice-selected threshold per fold; no retuning',bootstrap='10000 paired patient-cluster resamples, seed88; pooled series macro mean; conditional descriptive intervals, not corrected for architecture selection or multiple comparisons.',sources=sources),indent=2))
    print(summary.round(5).to_string())
    print(pd.DataFrame(cis).round(5).to_string(index=False))


if __name__=='__main__':main()
