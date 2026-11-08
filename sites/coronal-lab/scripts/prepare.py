#!/usr/bin/env python3
"""Prepare real BHSD volumes and fit reproducible 3D convolutional neural priors.

No external learned weights; source-image-only fitting, not clinical validation.
Run with the project's .venv Python. See --help for stages.
"""
import argparse, gzip, json, time, os
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates, affine_transform
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
WORK=Path(os.environ.get('CORONAL_WORK',ROOT/'work/full512')); ASSETS=Path(os.environ.get('CORONAL_ASSETS',ROOT/'dist/assets512'))
SOURCE=Path('/mnt/stor/datasets/BHSD/label_192/images')
CASES=[('case-01','ID_199b0a56_ID_09e623e993'),('case-02','ID_02f779fb_ID_c4d7f33559'),('case-03','ID_1cbe2853_ID_37afefdc7c'),('case-04','ID_47aa2b3a_ID_95a9953aba')]
NXY=512; FACTOR=4

def write_array(path, a):
    with gzip.open(path,'wb',compresslevel=6) as f: f.write(np.rint(a).clip(-32768,32767).astype('<i2').tobytes(order='C'))

def prepare():
    WORK.mkdir(exist_ok=True); ASSETS.mkdir(exist_ok=True)
    labels=pd.read_csv('/mnt/champaca/rsna-intracranial-hemorrhage-detection-16bit-png/slice_labels.csv')
    label_cols=['epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
    cases=[]
    for cid,uid in CASES:
        img=nib.as_closest_canonical(nib.load(SOURCE/f'{uid}.nii.gz'))
        a=img.get_fdata(dtype=np.float32)
        # Preserve every acquired in-plane sample and the original affine.
        assert a.shape[:2] == (NXY,NXY), a.shape
        native=a
        aff=img.affine.copy()
        # Network array is Z,Y,X. Physical mapping is retained separately.
        native=native.transpose(2,1,0).copy()
        np.save(WORK/f'{cid}-native.npy',native)
        corners=np.array([[x,y,z,1] for x in [0,NXY-1] for y in [0,NXY-1] for z in [0,a.shape[2]-1]])
        world=(aff@corners.T).T[:,:3]; lo=world.min(0); hi=world.max(0)
        spacing=1.25
        # Retain native X/Y sampling; interpolate only the coarse superior axis.
        spacing_xyz=np.array([*nib.affines.voxel_sizes(img.affine)[:2],spacing])
        shape=np.ceil((hi-lo)/spacing_xyz-1e-7).astype(int)+1
        target=np.diag([*spacing_xyz,1.]);target[:3,3]=lo
        mapper=np.linalg.inv(aff)@target
        xyz=np.indices(tuple(shape),dtype=np.float32).reshape(3,-1)
        coords=(mapper[:3,:3]@xyz+mapper[:3,3,None]).reshape(3,*shape)
        valid=np.all((coords>=-1e-4)&(coords<=np.array([NXY-1,NXY-1,a.shape[2]-1])[:,None,None,None]+1e-4),axis=0)
        np.savez_compressed(WORK/f'{cid}-geometry.npz',coords=coords,valid=valid,affine=aff,target=target)
        for name,order in [('nearest',0),('linear',1),('cubic',3)]:
            out=map_coordinates(native.transpose(2,1,0),coords,order=order,mode='nearest')
            out[~valid]=-1024
            # Binary is Z,Y,X so every coronal slice is row-major Z,X after gathering Y.
            write_array(ASSETS/f'{cid}-{name}.bin.gz',out.transpose(2,1,0))
            if name=='linear':
                thumb=Image.fromarray(np.uint8(np.clip(out[:,shape[1]//2,:].T[::-1,::-1]/80*255,0,255)))
                thumb=thumb.resize((thumb.width,max(1,round(thumb.height*spacing_xyz[2]/spacing_xyz[0]))))
                thumb.thumbnail((160,160))
                thumb.save(ASSETS/f'{cid}-thumb.png')
        write_array(ASSETS/f'{cid}-source.bin.gz',native)
        match=labels[(labels.patient_ID==uid[:11])&(labels.study_ID==uid[12:])]
        tags=[c for c in label_cols if len(match) and match[c].max()>0]
        dz=float(np.linalg.norm(aff[:3,2])); normal=np.cross(aff[:3,0],aff[:3,1]);normal/=np.linalg.norm(normal)
        tilt=float(np.degrees(np.arccos(np.clip(abs(normal[2]),0,1))))
        meta=dict(id=cid,source_id=uid,name=f'Study {len(cases)+1:02}',labels=tags,shape=shape[::-1].tolist(),source_shape=list(native.shape),spacing=spacing,spacing_xyz=spacing_xyz.tolist(),source_spacing=nib.affines.voxel_sizes(img.affine).tolist(),source_affine=aff.tolist(),target_affine=target.tolist(),inverse_source_affine=np.linalg.inv(aff).tolist(),tilt_deg=round(tilt,2),coverage_mm=round((a.shape[2]-1)*dz,1),native_slices=a.shape[2],methods=['nearest','linear','cubic'],ai_status='fitting',source_inplane_shape=list(a.shape[:2]),comparison_inplane_size=NXY)
        (WORK/f'{cid}.json').write_text(json.dumps(meta,indent=2)); cases.append(meta)
        print(cid,meta['native_slices'],round(dz,3),'mm',meta['tilt_deg'],'deg',shape,flush=True)
    manifest=dict(title='Coronal reconstruction lab',dataset='BHSD / RSNA ICH',cases=cases,inventory=dict(series=21744,slices=752802,train=19102,train_positive=7627,bhsd=192),provenance=dict(source='BHSD label_192/images NIfTI, overlapping local RSNA series',selection='Four deterministic positive examples near 2.5, 5.0, 5.3, and 6.5 mm; not a representative sample.',geometry='NIfTI affine, canonical RAS; axis-aligned patient coronal resampling at native X/Y spacing and 1.25 mm superior spacing. Header geometry has not been independently validated against DICOM.',comparison='Full native 512 × 512 in-plane samples retained for every method; every source axial plane retained. Display spacing is not acquired resolution.',ai='3D convolutional generative prior, fitted separately per volume and seed. No external weights or high-resolution targets. Not a pretrained diffusion model or GAN.'))
    (ASSETS/'manifest.json').write_text(json.dumps(manifest,indent=2))

def train(seed,device,steps,only):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    class Decoder(nn.Module):
        def __init__(self,shape):
            super().__init__();self.shape=shape
            self.blocks=nn.ModuleList([nn.Sequential(nn.Conv3d(16,32,3,padding=1),nn.LeakyReLU(.1)),nn.Sequential(nn.Conv3d(32,24,3,padding=1),nn.LeakyReLU(.1)),nn.Sequential(nn.Conv3d(24,12,3,padding=1),nn.LeakyReLU(.1))]);self.out=nn.Conv3d(12,1,1)
        def forward(self,z):
            for i,b in enumerate(self.blocks):
                z=b(z)
                size=[max(2,int(np.ceil(n/(4/2**i)))) for n in self.shape]
                z=F.interpolate(z,size=size,mode='trilinear',align_corners=True)
            return torch.sigmoid(self.out(z))
    for cid,uid in CASES:
        if only and cid!=only: continue
        torch.manual_seed(seed); np.random.seed(seed)
        native=np.load(WORK/f'{cid}-native.npy'); norm=(np.clip(native,-100,200)+100)/300
        target=torch.tensor(norm,device=device)[None,None]
        shape=((native.shape[0]-1)*FACTOR+1,NXY,NXY)
        net=Decoder(shape).to(device);noise=torch.randn(1,16,*[max(2,int(np.ceil(n/8))) for n in shape],device=device)*.1
        opt=torch.optim.Adam(net.parameters(),lr=.003)
        weight=1+3*((target>1/3)&(target<.7)).float()
        start=time.time(); losses=[]
        for it in range(steps):
            opt.zero_grad(set_to_none=True)
            pred=net(noise); observed=pred[:,:,::FACTOR]
            loss=((observed-target).square()*weight).mean()
            tv=(pred[:,:,1:]-pred[:,:,:-1]).abs().mean()
            total=loss+.0002*tv;total.backward();opt.step()
            if it%50==0 or it==steps-1:
                losses.append([it,float(loss.detach())]);print(cid,'seed',seed,'step',it,'loss',round(float(loss.detach()),6),'sec',round(time.time()-start),flush=True)
        with torch.no_grad():
            pred=net(noise)[0,0].cpu().numpy()*300-100
        raw_mae=float(np.abs(pred[::FACTOR]-np.clip(native,-100,200)).mean())
        # Exact correction at source samples in this simplified point-sampling model.
        # This does NOT model the acquisition slice-sensitivity / slab averaging kernel.
        residual=native-pred[::FACTOR]
        from scipy.interpolate import interp1d
        correction=interp1d(np.arange(native.shape[0])*FACTOR,residual,axis=0,kind='linear')(np.arange(shape[0]))
        projected=(pred+correction).astype(np.float32)
        np.save(WORK/f'{cid}-seed{seed}.npy',projected)
        stats=dict(seed=seed,steps=steps,seconds=round(time.time()-start,1),raw_observed_mae_hu=raw_mae,projected_observed_max_error_hu=float(np.abs(projected[::FACTOR]-native).max()),loss_trace=losses,parameters=sum(p.numel() for p in net.parameters()))
        (WORK/f'{cid}-seed{seed}.json').write_text(json.dumps(stats,indent=2))
        del net,opt,target,pred;torch.cuda.empty_cache()

def finish():
    m=json.loads((ASSETS/'manifest.json').read_text())
    for c in m['cases']:
        cid=c['id'];arrays=[];stats=[]
        for seed in [88,123]:
            p=WORK/f'{cid}-seed{seed}.npy'
            if not p.exists():break
            arrays.append(np.load(p)); stats.append(json.loads((WORK/f'{cid}-seed{seed}.json').read_text()))
        if len(arrays)!=2:continue
        geo=np.load(WORK/f'{cid}-geometry.npz');coords=geo['coords'].copy();coords[2]*=FACTOR
        samples=[]
        for a in arrays:
            out=map_coordinates(a.transpose(2,1,0),coords,order=1,mode='nearest');out[~geo['valid']]=-1024;samples.append(out)
        for name,out in [('dip',samples[0]),('ensemble',(samples[0]+samples[1])/2),('disagreement',np.abs(samples[0]-samples[1]))]:
            write_array(ASSETS/f'{cid}-{name}.bin.gz',out.transpose(2,1,0))
        c.update(ai_status='ready',methods=['nearest','linear','cubic','dip','ensemble','disagreement'],ai=dict(seeds=stats,training_range_hu=[-100,200],upsample_factor=FACTOR,description='Two independently fitted 3D convolutional neural priors; piecewise-linear residual projection preserves full-range source-plane samples. Ensemble is the mean; disagreement is absolute difference, not calibrated uncertainty.',validation='No ground truth between acquired planes. Source consistency is enforced, not evidence of recovered anatomy.'))
        print(cid,'AI ready',flush=True)
    (ASSETS/'manifest.json').write_text(json.dumps(m,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','finish']);p.add_argument('--seed',type=int,default=88);p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=350);p.add_argument('--case');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='train':train(a.seed,a.device,a.steps,a.case)
    else:finish()
