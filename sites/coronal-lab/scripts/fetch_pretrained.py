#!/usr/bin/env python3
"""Fetch pinned public sources/checkpoints; verify hashes; never upload study data."""
import sys,hashlib,subprocess,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'work';sys.path.insert(0,str(WORK/'deps'))
import requests,gdown
SOURCES={
 'QDM':('https://github.com/linYDTHU/QDM.git','19461ceb2a73e39bb0c5945e176a4d3022e82db7'),
 'Real-ESRGAN':('https://github.com/xinntao/Real-ESRGAN.git','a4abfb2979a7bbff3f69f58f58ae324608821e27')}
FILES={
 'medical_autoencoder.ckpt':('gdrive:177wL116e495OTxxaBeXZh9c0Wz7_slHU','7ff3fd770138132825e8aabc49ef411ae70264cf22b4a6fd7a13b457a462281c'),
 'qdm_medx4_l.pth':('gdrive:165iWcLFRPmIZPBDygiaeK6iKmMDCTlN6','1951be1327f090488da1f08f31923e747051c45543df5be5a82a322c4867bcec'),
 'RealESRGAN_x4plus.pth':('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth','4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1'),
 'RealESRNet_x4plus.pth':('https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth','a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df')}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 (WORK/'vendor').mkdir(parents=True,exist_ok=True);(WORK/'weights').mkdir(exist_ok=True)
 for name,(url,rev) in SOURCES.items():
  destination=WORK/'vendor'/name
  if not destination.exists():
   subprocess.run(['git','init',str(destination)],check=True)
   subprocess.run(['git','-C',str(destination),'fetch','--depth','1',url,rev],check=True)
   subprocess.run(['git','-C',str(destination),'checkout','--detach','FETCH_HEAD'],check=True)
  actual=subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()
  if actual!=rev:raise RuntimeError(f'{name} source differs from pinned revision; preserve and inspect it before proceeding.')
 reference=WORK/'vendor/rrdbnet_arch.py'
 reference_hash='6aa8eb3f77e918b90eeff4ebfaf8745abe0c40eb8fcc03abbc1928693d70ab58'
 if not reference.exists():
  url='https://raw.githubusercontent.com/XPixelGroup/BasicSR/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/archs/rrdbnet_arch.py'
  response=requests.get(url,timeout=30);response.raise_for_status();reference.write_bytes(response.content)
 if digest(reference)!=reference_hash:raise RuntimeError('Upstream RRDB reference source hash mismatch')
 hashes={}
 for name,(url,expected) in FILES.items():
  dest=WORK/'weights'/name
  if not dest.exists():
   tmp=dest.with_suffix(dest.suffix+'.download')
   if url.startswith('gdrive:'):gdown.download(id=url.split(':',1)[1],output=str(tmp),quiet=False)
   else:
    with requests.get(url,stream=True,timeout=120) as r:
     r.raise_for_status()
     with tmp.open('wb') as out:
      for chunk in r.iter_content(1024*1024):out.write(chunk)
   if digest(tmp)!=expected:raise RuntimeError(f'{name}: unexpected checkpoint hash; download retained for inspection')
   tmp.replace(dest)
  actual=digest(dest)
  if actual!=expected:raise RuntimeError(f'{name}: existing checkpoint does not match expected hash')
  hashes[name]={'bytes':dest.stat().st_size,'sha256':actual};print(name,'verified',flush=True)
 (WORK/'checkpoint-hashes.json').write_text(json.dumps(hashes,indent=2))
if __name__=='__main__':main()
