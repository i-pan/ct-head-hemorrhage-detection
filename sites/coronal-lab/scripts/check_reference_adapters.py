"""Compare adapters numerically against upstream inference code (no evaluation tuning)."""
import ast,json,math,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pretrained_models import ROOT,RealESR,QDMModel,pad_to

# Compare the in-tree RRDB inference implementation with the upstream class
# definitions. Only class nodes are compiled; imports / executable module code
# and the registry decorator are excluded. Initialization is overwritten by
# the same strictly loaded public weights.
def check_rrdb():
    source=ROOT/'work/vendor/rrdbnet_arch.py';tree=ast.parse(source.read_text())
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
    for c in classes:c.decorator_list=[]
    namespace={'torch':torch,'nn':torch.nn,'F':F,'make_layer':lambda block,n,**kwargs:torch.nn.Sequential(*[block(**kwargs) for _ in range(n)]),'default_init_weights':lambda *a,**kw:None,'pixel_unshuffle':F.pixel_unshuffle}
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),str(source),'exec'),namespace)
    ref=namespace['RRDBNet'](3,3,scale=4,num_feat=64,num_block=23,num_grow_ch=32).eval()
    ours=RealESR('esrgan','cpu').net
    ref.load_state_dict(ours.state_dict(),strict=True)
    torch.manual_seed(77);x=torch.rand(1,3,12,15)
    with torch.inference_mode():difference=float((ref(x)-ours(x)).abs().max())
    assert difference<1e-6
    return difference

def check_qdm():
    from omegaconf import OmegaConf
    torch.cuda.set_device(0);model=QDMModel('cuda:0')
    from utils import util_image
    source=ROOT/'work/vendor/QDM/sampler.py';tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QDMSampler')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='sample_func')
    namespace={'torch':torch,'F':F,'math':math,'np':np,'util_image':util_image}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),namespace)
    class Context:pass
    ref=Context();ref.configs=OmegaConf.load(ROOT/'work/vendor/QDM/configs/inference/medx4_qdm_l.yaml');ref.chop_size=128;ref.rank=0;ref.sf=4;ref.base_diffusion=model.diffusion;ref.model=model.net;ref.autoencoder=model.ae;ref.setup_seed=lambda:torch.manual_seed(77)
    native=np.load(ROOT/'work/case-01-native.npy');low=native[::-1,90,::-1].copy();x=pad_to(torch.tensor((np.clip(low,-100,200)+100)/300,device='cuda:0')[None,None].repeat(1,3,1,1)[:,:,:,:128],128)
    with torch.inference_mode():
        torch.manual_seed(77);ours=model.tile(x)
        torch.manual_seed(77)
        with torch.autocast('cuda',dtype=torch.float16):expected=namespace['sample_func'](ref,x*2-1)
        expected=(expected.float()+1)/2
        difference=float((expected-ours).abs().max())
    assert difference<1e-5,difference
    return difference

if __name__=='__main__':
    torch.set_num_threads(4)
    result={'rrdb_max_abs_error_vs_upstream':check_rrdb(),'qdm_tile_max_abs_error_vs_upstream':check_qdm()}
    (ROOT/'work/reference-adapter-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
