"""Inference adapters for official pretrained QDM and Real-ESR(GAN/Net).

RRDB definitions follow XPixelGroup/BasicSR rrdbnet_arch.py (Apache-2.0; see BasicSR-LICENSE.txt).
No external checkpoint is deserialized with unrestricted pickle.
"""
from pathlib import Path
import sys, json, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'work/deps'))
WEIGHTS=ROOT/'work/weights'

class ResidualDenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(1,6):setattr(self,f'conv{i}',nn.Conv2d(64+(i-1)*32,32 if i<5 else 64,3,1,1))
        self.lrelu=nn.LeakyReLU(.2,inplace=True)
    def forward(self,x):
        out=[x]
        for i in range(1,5):out.append(self.lrelu(getattr(self,f'conv{i}')(torch.cat(out,1))))
        return x+.2*self.conv5(torch.cat(out,1))
class RRDB(nn.Module):
    def __init__(self):
        super().__init__();self.rdb1=ResidualDenseBlock();self.rdb2=ResidualDenseBlock();self.rdb3=ResidualDenseBlock()
    def forward(self,x):return x+.2*self.rdb3(self.rdb2(self.rdb1(x)))
class RRDBNet(nn.Module):
    def __init__(self):
        super().__init__();self.conv_first=nn.Conv2d(3,64,3,1,1);self.body=nn.Sequential(*[RRDB() for _ in range(23)])
        for name in ['conv_body','conv_up1','conv_up2','conv_hr']:setattr(self,name,nn.Conv2d(64,64,3,1,1))
        self.conv_last=nn.Conv2d(64,3,3,1,1);self.lrelu=nn.LeakyReLU(.2,inplace=True)
    def forward(self,x):
        x=self.conv_first(x);x=x+self.conv_body(self.body(x))
        x=self.lrelu(self.conv_up1(F.interpolate(x,scale_factor=2,mode='nearest')))
        x=self.lrelu(self.conv_up2(F.interpolate(x,scale_factor=2,mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(x)))

class RealESR:
    def __init__(self,method,device):
        name={'esrgan':'RealESRGAN_x4plus.pth','esrnet':'RealESRNet_x4plus.pth'}[method]
        state=torch.load(WEIGHTS/name,map_location='cpu',weights_only=True)
        state=state.get('params_ema',state.get('params',state))
        self.net=RRDBNet();self.net.load_state_dict(state,strict=True);self.net=self.net.eval().to(device)
        self.parameters=sum(p.numel() for p in self.net.parameters())
        print(method,'loaded',self.parameters,'parameters',flush=True)
    @torch.inference_mode()
    def __call__(self,x):
        with torch.autocast('cuda',dtype=torch.float16):return self.net(x).clamp(0,1).float()

def pad_to(x,size):
    """Repeated reflection handles acquired stacks shorter than half a tile."""
    ph=max(0,size-x.shape[-2]);pw=max(0,size-x.shape[-1])
    while ph or pw:
        h=min(ph,x.shape[-2]-1);w=min(pw,x.shape[-1]-1)
        x=F.pad(x,(0,w,0,h),mode='reflect');ph-=h;pw-=w
    return x

class QDMModel:
    def __init__(self,device):
        sys.path.insert(0,str(ROOT/'work/vendor/QDM'))
        from omegaconf import OmegaConf
        from models.dit import QDM
        from models.script_util import create_masked_gaussian_diffusion
        from ldm.models.autoencoder import AutoencoderKLTorch
        from utils.util_image import generate_quadtree_masks
        self.generate_masks=generate_quadtree_masks
        self.device=device
        cfg=OmegaConf.load(ROOT/'work/vendor/QDM/configs/inference/medx4_qdm_l.yaml')
        self.net=QDM(**OmegaConf.to_container(cfg.model.params));self.ae=AutoencoderKLTorch(**OmegaConf.to_container(cfg.autoencoder.params))
        for model,path in [(self.net,WEIGHTS/'qdm_medx4_l.pth'),(self.ae,WEIGHTS/'medical_autoencoder.ckpt')]:
            state=torch.load(path,map_location='cpu',weights_only=True)
            if 'state_dict' in state:state=state['state_dict']
            state={k.removeprefix('module.'):v for k,v in state.items()}
            # Official AE training checkpoint includes loss/discriminator modules;
            # inference architecture has only encode/decode weights.
            required=model.state_dict();missing=set(required)-set(state)
            if missing:raise ValueError(f'Missing checkpoint weights: {sorted(missing)[:10]}')
            excluded=[k for k in state if k not in required]
            model.load_state_dict({k:state[k] for k in required},strict=True)
            print(path.name,'loaded; excluded training-only keys:',len(excluded),flush=True)
            model.to(device).eval();model.requires_grad_(False)
        self.diffusion=create_masked_gaussian_diffusion(**OmegaConf.to_container(cfg.diffusion.params))
        self.parameters=sum(p.numel() for p in self.net.parameters())
        print('QDM parameters',self.parameters,flush=True)
    @torch.inference_mode()
    def tile(self,x):
        # Exact published mask threshold, 15-step schedule, and 64 chunks.
        y=x*2-1;h,w=y.shape[-2:]
        mask=self.generate_masks(y,.15,h,w).unsqueeze(1)
        kwargs={'lq':y,'mask':self.generate_masks(y,.15,h//2,w//2),'up_pred':False,'chunk_size':64}
        with torch.autocast('cuda',dtype=torch.float16):
            out=self.diffusion.p_sample_loop(y=y,mask=mask,model=self.net.forward_w_mask,first_stage_model=self.ae,noise=None,noise_repeat=False,clip_denoised=False,denoised_fn=None,device=self.device,model_kwargs=kwargs,progress=False)
        return (out.float().clamp(-1,1)+1)/2
    @torch.inference_mode()
    def __call__(self,x):
        h,w=x.shape[-2:];p=128
        # The native coronal stack is <=60 rows and 512 columns: overlapping
        # 128-column tiles. Reflect only for network context, crop before output.
        starts=list(range(0,max(1,w-p+1),p-16))
        if starts[-1]!=max(0,w-p):starts.append(max(0,w-p))
        total=torch.zeros(x.shape[0],3,h*4,w*4,device=x.device);weights=torch.zeros_like(total)
        for start in starts:
            tile=pad_to(x[:,:,:,start:start+p],p);out=self.tile(tile)[:,:,:h*4,:min(p,w)*4]
            weight=torch.exp(-.5*(torch.linspace(-1,1,out.shape[-1],device=x.device)/.5)**2)[None,None,None,:]
            total[:,:,:,start*4:start*4+out.shape[-1]]+=out*weight;weights[:,:,:,start*4:start*4+out.shape[-1]]+=weight
        return total/weights.clamp_min(1e-8)

def load_model(method,device):return QDMModel(device) if method=='qdm' else RealESR(method,device)

@torch.inference_mode()
def infer_coronal(model,source,device,plane_indices,batch_size=2):
    """Return Z-high,Y-selected,X; RGB is repeated grayscale in [-100,200] HU."""
    nz,ny,nx=source.shape;out=[]
    for start in range(0,len(plane_indices),batch_size):
        indices=plane_indices[start:start+batch_size]
        # Radiological view: superior at top and patient right at image left.
        low=source[::-1,indices,::-1].transpose(1,0,2).copy()
        x=torch.tensor((np.clip(low,-100,200)+100)/300,device=device)[:,None].repeat(1,3,1,1)
        high=model(x).mean(1,keepdim=True)
        high=F.interpolate(high,size=(nz*4,nx),mode='area')
        # Standard 4x pixel-center mapping places the native samples at 1.5+4*k.
        gz=(torch.arange((nz-1)*4+1,device=device)+1.5)/(nz*4-1)*2-1
        gx=torch.linspace(-1,1,nx,device=device);zz,xx=torch.meshgrid(gz,gx,indexing='ij')
        grid=torch.stack([xx,zz],dim=-1)[None].expand(len(indices),-1,-1,-1)
        aligned=F.grid_sample(high,grid,mode='bilinear',align_corners=True)[:,0]
        out.append((aligned.cpu().numpy()*300-100)[:,::-1,::-1])
    return np.concatenate(out,axis=0).transpose(1,0,2).copy()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('method',choices=['qdm','esrgan','esrnet']);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(88)
    model=load_model(args.method,args.device);source=np.load(ROOT/'work/case-02-native.npy')
    t=time.time();pred=infer_coronal(model,source,args.device,[90,94],2);torch.cuda.synchronize()
    np.save(ROOT/f'work/{args.method}-smoke.npy',pred)
    print('SMOKE',pred.shape,'finite',np.isfinite(pred).all(),'range',float(pred.min()),float(pred.max()),'seconds',time.time()-t,flush=True)
