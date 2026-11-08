"""CPU-only descriptive analysis of saved fixed-validation classification logits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'diagnostics/ich_positive_slice_burden_20261107'
RUNS={'efficientnet':'rsna_ich_effv2m_sequence_bigru/seqsweep_01_sliceheavy_20260713', 'maxvit':'rsna_ich_maxvit_tiny_sequence_bigru/seqsweep_01_sliceheavy_maxvit_20260714'}
D={k:dict(np.load(ROOT/'experiments'/v/'validation_predictions.npz',allow_pickle=False)) for k,v in RUNS.items()}
a=D['maxvit']
for k in ['slice_targets','slice_series_index','series_targets','series_index']:
 np.testing.assert_array_equal(a[k],D['efficientnet'][k])
meta=pd.read_csv(ROOT/'data/features/rsna_ich_maxvit_tiny_9ch/val_series.csv').set_index('series_index').loc[a['series_index']].reset_index()
assert meta.study_id.nunique()==len(meta)
sl=pd.DataFrame({'series_index':a['slice_series_index'],'positive':a['slice_targets'][:,-1]})
burden=sl.groupby('series_index').positive.agg(positive_slices='sum',total_slices='size')
table=meta.merge(burden,on='series_index',validate='one_to_one')
np.testing.assert_array_equal(table.series_index,a['series_index'])
np.testing.assert_array_equal(table.total_slices,table.length)
table['target']=a['series_targets'][:,-1].astype(int)
np.testing.assert_array_equal(table.target,(table.positive_slices>0).astype(int))
table['positive_fraction']=table.positive_slices/table.total_slices
labels=['0–10%','>10–25%','>25–50%','>50%']
table['burden_bin']=pd.cut(table.positive_fraction,[0,.1,.25,.5,1],labels=labels).astype('object').fillna('Negative')
scores={k:v['series_logits'][:,-1].astype(float) for k,v in D.items()}
scores['equal_ensemble']=(scores['maxvit']+scores['efficientnet'])/2
for name,d in D.items():
 b=pd.Series(d['slice_base_logits'][:,-1]).groupby(d['slice_series_index']).max()
 scores[name+'_backbone_max']=b.loc[table.series_index].to_numpy(dtype=float)
def wilson(k,n):
 z=norm.ppf(.975);p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;r=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return float(c-r),float(c+r)
negative=table.target.eq(0).to_numpy();rows=[];operating=[]
for name,score in scores.items():
 table[name+'_probability']=expit(score)
 for specificity in [.95,.98,.99]:
  allowed=int(np.floor((1-specificity)*negative.sum()+1e-9))
  threshold=np.nextafter(np.sort(score[negative])[::-1][allowed],np.inf)
  calls=score>=threshold;fp=int(calls[negative].sum());assert fp<=allowed
  operating.append(dict(model=name,target_specificity=specificity,threshold_logit=float(threshold),threshold_probability=float(expit(threshold)),false_positive_studies=fp,negative_studies=int(negative.sum()),specificity=1-fp/negative.sum(),overall_auc=roc_auc_score(table.target,score)))
  table[f'{name}_positive_at_{int(specificity*100)}spec']=calls
  for label in labels:
   mask=table.burden_bin.eq(label).to_numpy();n=int(mask.sum());tp=int(calls[mask].sum());lo,hi=wilson(tp,n)
   rows.append(dict(model=name,target_specificity=specificity,burden_bin=label,n=n,detected=tp,missed=n-tp,sensitivity=tp/n,ci_low=lo,ci_high=hi,median_total_slices=float(table.loc[mask,'total_slices'].median()),median_positive_slices=float(table.loc[mask,'positive_slices'].median())))
OUT.mkdir(exist_ok=True)
results=pd.DataFrame(rows);results.to_csv(OUT/'sensitivity_by_burden.csv',index=False)
pd.DataFrame(operating).to_csv(OUT/'operating_points.csv',index=False)
table.to_csv(OUT/'study_predictions_and_burden.csv',index=False)
# A second description uses absolute positive-slice counts and study-length strata.
checks=[];positive=table.target.eq(1)
for grouping,groups in [('positive_slice_count',{'1–3':table.positive_slices.between(1,3),'4–10':table.positive_slices.between(4,10),'>10':table.positive_slices.gt(10)}),('study_length_and_burden',{f'{length}; {fraction}':lengthmask&fractionmask for length,lengthmask in [('≤35 slices',table.total_slices.le(35)),('>35 slices',table.total_slices.gt(35))] for fraction,fractionmask in [('≤10% positive',table.positive_fraction.le(.1)),('>10% positive',table.positive_fraction.gt(.1))]})]:
 for label,mask in groups.items():
  mask=mask&positive;n=int(mask.sum());tp=int(table.loc[mask,'maxvit_positive_at_99spec'].sum())
  checks.append(dict(grouping=grouping,group=label,n=n,detected=tp,sensitivity=tp/n if n else None))
pd.DataFrame(checks).to_csv(OUT/'proxy_sensitivity_checks.csv',index=False)
# Cluster bootstrap respects repeated studies from the same patient. Threshold fixed.
patients=table.patient_id.unique();pos_table=table.loc[positive].copy();rng=np.random.default_rng(88)
small=pos_table.positive_fraction.le(.1);large=~small
agg=[]
for mask in [small,large]:
 counts=pos_table.loc[mask].groupby('patient_id').maxvit_positive_at_99spec.agg(['sum','size']).reindex(patients,fill_value=0)
 agg.append(counts.to_numpy())
deltas=[]
for _ in range(5000):
 ix=rng.integers(0,len(patients),len(patients));sums=[x[ix].sum(axis=0) for x in agg]
 if all(x[1]>0 for x in sums):deltas.append(sums[1][0]/sums[1][1]-sums[0][0]/sums[0][1])
summary=dict(split='Fixed validation, not full OOF; selected runs had already used this validation set for model selection.',runs=RUNS,studies=len(table),patients=len(patients),positive_studies=int(positive.sum()),negative_studies=int(negative.sum()),proxy='Ground-truth any-positive axial slices / all axial slices per series; one series per study in this set. Measures axial extent, not bleed volume.',primary='Selected MaxViT contextual series head, any hemorrhage class (last column).',threshold_policy='Lowest threshold permitting no more than the target false-positive proportion on the same validation negatives. Descriptive, not independently validated specificity.',intervals='Bin intervals are Wilson 95% intervals (study-level, unclustered). Difference interval is patient-cluster bootstrap with fixed threshold, 5000 resamples; does not include model selection or threshold estimation uncertainty.',large_minus_small_sensitivity_gap_bootstrap_ci=np.quantile(deltas,[.025,.975]).tolist(),test_set_accessed=False,gpu_used=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(figsize=(8,4.5))
for name,offset,color in [('maxvit',-.12,'#087b7f'),('efficientnet',0,'#bc6b28'),('equal_ensemble',.12,'#5d5aa5')]:
 r=results[(results.model==name)&(results.target_specificity==.99)];x=np.arange(4)+offset;y=r.sensitivity.to_numpy()*100
 ax.errorbar(x,y,yerr=np.maximum(0,np.array([r.sensitivity-r.ci_low,r.ci_high-r.sensitivity]))*100,fmt='o-',capsize=3,label=name.replace('_',' '),color=color)
ax.set_xticks(range(4),[f'{label}\n(n={int((table.burden_bin==label).sum())})' for label in labels]);ax.set_ylim(0,103);ax.set_ylabel('Study sensitivity (%)');ax.set_xlabel('Ground-truth positive slices / study');ax.set_title('Sensitivity by positive-slice burden\nFixed validation; 5 false positives / 589 negative studies per model');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'sensitivity_by_burden.png',dpi=180);plt.close(fig)
print(results[(results.model.isin(['maxvit','efficientnet','equal_ensemble'])) & (results.target_specificity==.99)].to_string(index=False));print(pd.DataFrame(checks).to_string(index=False));print(json.dumps(summary,indent=2))
