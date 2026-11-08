#!/usr/bin/env python3
"""Run pretrained pilot and leakage-free even/odd slice withholding experiment."""
import argparse,gzip,json,time,hashlib,os
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates,affine_transform
from pretrained_models import ROOT,load_model,infer_coronal
import torch
from skimage.metrics import structural_similarity
WORK=Path(os.environ.get('CORONAL_WORK',ROOT/'work/full512'));ASSETS=Path(os.environ.get('CORONAL_ASSETS',ROOT/'dist/assets512'))

def write_array(path,a):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(tmp,'wb',compresslevel=6) as stream:stream.write(np.rint(a).clip(-32768,32767).astype('<i2').tobytes())
    tmp.replace(path)

def atomic_json(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def metrics(pred,reference,lesion):
    band=(reference>=-20)&(reference<=120)
    error=pred.astype(np.float64)-reference
    a=np.clip(pred,0,80)/80;b=np.clip(reference,0,80)/80
    return dict(soft_tissue_mae_hu=float(np.abs(error[band]).mean()),soft_tissue_rmse_hu=float(np.sqrt(np.square(error[band]).mean())),hemorrhage_mae_hu=float(np.abs(error[lesion]).mean()) if lesion.any() else None,brain_window_psnr_db=float(-10*np.log10(max(np.mean((a-b)**2),1e-12))),brain_window_ssim=float(np.mean([structural_similarity(x,y,data_range=1) for x,y in zip(a,b)])),soft_tissue_voxels=int(band.sum()),hemorrhage_voxels=int(lesion.sum()))

def project(pred,native):
    residual=native-pred[::4]
    correction=interp1d(np.arange(len(native))*4,residual,axis=0,kind='linear')(np.arange(len(pred)))
    out=(pred+correction).astype(np.float32)
    assert np.max(np.abs(out[::4]-native))<.002
    return out

def evaluate_setup(c):
    cid=c['id'];native=np.load(WORK/f'{cid}-native.npy');known=np.arange(0,len(native),2);held=np.arange(1,known[-1],2)
    image=nib.as_closest_canonical(nib.load(f"/mnt/stor/datasets/BHSD/label_192/labels/{c['source_id']}.nii.gz"))
    mapper=np.linalg.inv(image.affine)@np.array(c['source_affine'])
    masks=affine_transform(image.get_fdata(),mapper[:3,:3],mapper[:3,3],output_shape=native.shape[::-1],order=0,prefilter=False).transpose(2,1,0)>0
    truth=native[held];lesion=masks[held];return native,known,held,truth,lesion

def baselines(m):
    for c in m['cases']:
        cid=c['id'];native,known,held,truth,lesion=evaluate_setup(c)
        coords=np.indices(truth.shape,dtype=np.float32);coords[0]=held[:,None,None]/2
        results={}
        for name,order in [('nearest',0),('linear',1),('cubic',3)]:
            pred=map_coordinates(native[known],coords,order=order,mode='nearest')
            results[name]=metrics(pred,truth,lesion);write_array(ASSETS/f'{cid}-eval-{name}.bin.gz',pred)
        write_array(ASSETS/f'{cid}-eval-reference.bin.gz',truth);write_array(ASSETS/f'{cid}-eval-lesion.bin.gz',lesion.astype(np.int16))
        out=dict(shape=list(truth.shape),input_indices=known.tolist(),heldout_indices=held.tolist(),heldout_count=len(held),input_spacing_mm=2*c['source_spacing'][2],methods=results,protocol='Retain even native axial planes. Predict only interior odd planes withheld from every model input and residual correction. Last unbracketed odd plane excluded. No local training or tuning. All 512 coronal planes processed independently.',metrics_note='Soft-tissue mask: reference intensity −20 to 120 HU (not a brain segmentation). Hemorrhage mask: nonzero BHSD labels, nearest-resampled to the common grid. SSIM and PSNR: whole acquired axial images clipped to W80/L40, including background. Metrics are computed before int16 export.')
        atomic_json(WORK/f'{cid}-evaluation.json',out);print(cid,'baseline evaluation',results['linear'],flush=True)

def run(m,method,device,batch_size,cases):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    model=load_model(method,device)
    for c in m['cases']:
        if cases and c['id'] not in cases:continue
        cid=c['id'];native,known,held,truth,lesion=evaluate_setup(c);stats={};start_time=time.time()
        # Fixed seed per study/phase. No parameter selection from withheld targets.
        for phase,source in [('full',native),('withheld',native[known])]:
            torch.manual_seed(8800+int(cid[-2:])+(100 if phase=='withheld' else 0))
            raw=np.empty(((len(source)-1)*4+1,source.shape[1],source.shape[2]),dtype=np.float32)
            for start in range(0,source.shape[1],batch_size):
                indices=list(range(start,min(source.shape[1],start+batch_size)))
                raw[:,indices]=infer_coronal(model,source,device,indices,batch_size)
                if start%32==0:print(method,cid,phase,'coronal',start,'/',source.shape[1],'seconds',round(time.time()-start_time,1),flush=True)
            assert np.isfinite(raw).all();pred=project(raw,source)
            np.save(WORK/f'{cid}-{method}-{phase}-raw.npy',raw);np.save(WORK/f'{cid}-{method}-{phase}.npy',pred)
            if phase=='full':
                stats['source_projection_max_error_hu']=float(np.abs(pred[::4]-source).max())
                geo=np.load(WORK/f'{cid}-geometry.npz');coords=geo['coords'].copy();coords[2]*=4
                for key,a in [(method,pred),(method+'_raw',raw)]:
                    out=map_coordinates(a.transpose(2,1,0),coords,order=1,mode='nearest');out[~geo['valid']]=-1024
                    write_array(ASSETS/f'{cid}-{key}.bin.gz',out.transpose(2,1,0))
            else:
                # Original native index z corresponds to output index 2*z because
                # known axial samples are separated by two original slice intervals.
                stats['projected_metrics']=metrics(pred[held*2],truth,lesion)
                stats['raw_metrics']=metrics(raw[held*2],truth,lesion)
                write_array(ASSETS/f'{cid}-eval-{method}.bin.gz',pred[held*2]);write_array(ASSETS/f'{cid}-eval-{method}_raw.bin.gz',raw[held*2])
        stats.update(method=method,seconds=time.time()-start_time,parameters=model.parameters,batch_size=batch_size,coronal_planes=native.shape[1],local_training=False,normalization_hu=[-100,200])
        atomic_json(WORK/f'{cid}-{method}-results.json',stats)
        print('COMPLETE',method,cid,json.dumps(stats['projected_metrics']),flush=True)

def finish(m,allow_pending=False):
    model_info={
      'qdm':dict(name='QDM-L · medical CT ×4',family='Pretrained latent diffusion',domain='Head-and-neck CT (HaN-Seg / SegRap2023)',source='https://github.com/linYDTHU/QDM',checkpoint='qdm_medx4_l.pth',steps=15,parameters=691941580,notes='Official CT-specific model and medical VAE. Native anisotropic coronal inputs are outside its 2D isotropic SR training task. Upstream HU window was not specified in the released inference documentation; this pilot uses −100 to 200 HU.'),
      'esrgan':dict(name='Real-ESRGAN ×4plus',family='Pretrained GAN',domain='Natural images · out of domain',source='https://github.com/xinntao/Real-ESRGAN',checkpoint='RealESRGAN_x4plus.pth',parameters=16697987,notes='Adversarially trained RRDB generator. No medical fine-tuning. Grayscale repeated across RGB channels; output channels averaged.'),
      'esrnet':dict(name='Real-ESRNet ×4plus',family='Pretrained non-adversarial control',domain='Natural images · out of domain',source='https://github.com/xinntao/Real-ESRGAN',checkpoint='RealESRNet_x4plus.pth',parameters=16697987,notes='Pixel-loss-trained counterpart to Real-ESRGAN. Same architecture; no adversarial refinement or medical fine-tuning.')}
    pending=[key for key in model_info if not all((WORK/f"{c['id']}-{key}-results.json").exists() for c in m['cases'])]
    if pending and not allow_pending:raise RuntimeError(f'Models still pending: {pending}')
    m['pretrained_pending']=pending
    for c in m['cases']:
        cid=c['id'];evaluation=json.loads((WORK/f'{cid}-evaluation.json').read_text());c['pretrained']={}
        for key,info in model_info.items():
            if key in pending:continue
            path=WORK/f'{cid}-{key}-results.json'
            if not path.exists():raise RuntimeError(f'Missing completed result {path}')
            stats=json.loads(path.read_text());assert stats['coronal_planes']==c['source_shape'][1]==512;c['pretrained'][key]=stats;evaluation['methods'][key]=stats['projected_metrics'];evaluation.setdefault('raw_methods',{})[key]=stats['raw_metrics']
            for method in [key,key+'_raw']:
                if method not in c['methods']:c['methods'].append(method)
        c['evaluation']=evaluation
    m['pretrained_models']=model_info
    m['checkpoint_hashes']=json.loads((WORK/'checkpoint-hashes.json').read_text())
    m['pretrained_protocol']=dict(input_hu_range=[-100,200],adapter='Every one of 512 native-grid coronal planes processed by each fixed 2D ×4 model. RGB input repeats normalized grayscale; output channels averaged. Upsampled X is area-reduced to the common grid; Z is aligned to native voxel centers. No interpolated AP stand-ins.',projection='Piecewise-linear residual correction on observed axial planes only; raw predictions also retained. A point-sampling constraint, not a slice-profile model.',evaluation='59 interior native axial planes withheld across four studies. Doubled acquired spacing. Four selected positive studies; no external generalization or thin-slice-ground-truth claim.',source_commits={'QDM':'19461ceb2a73e39bb0c5945e176a4d3022e82db7','Real-ESRGAN':'a4abfb2979a7bbff3f69f58f58ae324608821e27'})
    m['provenance']['ai']='Custom fitted deep image priors plus official pretrained QDM-L medical CT diffusion, Real-ESRGAN, and Real-ESRNet. Raw and observed-sample-projected outputs; withholding evaluation for the three pretrained models.'
    atomic_json(ASSETS/'manifest.json',m)
    # Export machine-readable, case-averaged summary; no pooling across pixels/cases.
    summary={}
    for method in ['nearest','linear','cubic','qdm','esrgan','esrnet']:
        if method in pending:continue
        summary[method]={key:float(np.mean([c['evaluation']['methods'][method][key] for c in m['cases']])) for key in ['soft_tissue_mae_hu','hemorrhage_mae_hu','brain_window_ssim']}
    atomic_json(ASSETS/'benchmark-summary.json',dict(cases=4,withheld_planes=sum(c['evaluation']['heldout_count'] for c in m['cases']),aggregation='Equal weight per study; descriptive pilot only',methods=summary))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['baselines','run','finish']);p.add_argument('--method',choices=['qdm','esrgan','esrnet']);p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--cases',nargs='*');p.add_argument('--allow-pending',action='store_true');a=p.parse_args()
    m=json.loads((ASSETS/'manifest.json').read_text())
    if a.stage=='baselines':baselines(m)
    elif a.stage=='run':run(m,a.method,a.device,a.batch_size,a.cases)
    else:finish(m,a.allow_pending)
