const $ = (id) => document.getElementById(id);
const windows = {brain:[40,80],subdural:[80,200],wide:[50,300]};
const pretrainedKeys=new Set(['qdm','esrgan','esrnet']);
function activeMethod(){const key=$('advanced-method').value;return pretrainedKeys.has(key)&&!$('preserve-source').checked?key+'_raw':key;}
const descriptions = {
  qdm:['PRETRAINED CT','QDM-L · 15-step diffusion','CT-specific weights; through-plane transfer.','MEDICAL CT PRIOR · 15 STEPS'],
  esrgan:['OUT-OF-DOMAIN GAN','Real-ESRGAN ×4plus','Natural-image GAN; no medical fine-tuning.','NATURAL-IMAGE PRIOR · GAN'],
  esrnet:['OUT-OF-DOMAIN CONTROL','Real-ESRNet ×4plus','Non-adversarial counterpart to Real-ESRGAN.','NATURAL-IMAGE PRIOR · PIXEL LOSS'],
  ensemble:['EXPERIMENTAL AI','Two fitted 3D generators','Mean of two source-constrained estimates.','SOURCE-CONSTRAINED · 2 SEEDS'],
  dip:['EXPERIMENTAL AI','Deep generative prior · seed 88','A fitted generator, not a pretrained CT model.','SOURCE-CONSTRAINED · SEED 88'],
  cubic:['CONVENTIONAL','Cubic B-spline interpolation','Smooth interpolation; overshoot is possible.','CORONAL · CUBIC B-SPLINE'],
  disagreement:['EXPLORATORY MAP','Absolute difference between seeds','Not calibrated uncertainty or a truth error map.','|SEED 88 − SEED 123| · 0–30'],
  delta:['EXPLORATORY MAP','Absolute AI − linear difference','Shows change, not improvement or error.','|ENSEMBLE − LINEAR| · 0–30'],
};
let manifest, current, arrays={}, coronal=0, axial=0, generation=0, playTimer=null, busy=false;
const cache=new Map(),pending=new Map();
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const titleCase=s=>s.replaceAll('_',' ');
function toast(message){$('toast').textContent=message;$('toast').hidden=false;setTimeout(()=>$('toast').hidden=true,3200);}
function stopPlay(){clearInterval(playTimer);playTimer=null;$('play-button').textContent='▷ Play';$('play-button').setAttribute('aria-label','Play coronal slices');}
function saveFile(blob,name){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
async function readArray(cid,method){
  const key=`${cid}-${method}`;
  if(cache.has(key))return cache.get(key);
  if(pending.has(key))return pending.get(key);
  const promise=(async()=>{
    const response=await fetch(`assets512/${key}.bin.gz`);
    if(!response.ok)throw new Error(`${method}: HTTP ${response.status}`);
    if(typeof DecompressionStream==='undefined')throw new Error('This browser needs gzip DecompressionStream support. Please use a current browser.');
    const buffer=await new Response(response.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    const result=new Int16Array(buffer);
    const c=manifest.cases.find(c=>c.id===cid),shape=method==='source'?c.source_shape:method.startsWith('eval-')?c.evaluation.shape:c.shape;
    if(result.length!==shape.reduce((a,b)=>a*b,1))throw new Error(`Invalid volume dimensions for ${key}`);
    cache.set(key,result);
    let bytes=[...cache.values()].reduce((n,a)=>n+a.byteLength,0);
    for(const [old,a] of cache){if(bytes<=256*1024*1024)break;cache.delete(old);bytes-=a.byteLength;}
    return result;
  })();
  pending.set(key,promise);
  try{return await promise;}finally{pending.delete(key);}
}
function setBusy(value,message=''){
  busy=value;$('load-status').textContent=message;
  for(const id of ['export-button','play-button','prev-slice','next-slice','coronal-slider','axial-slider'])$(id).disabled=value;
}
function updateMetadata(){
  const c=current,dz=c.source_spacing[2];
  $('metric-spacing').textContent=`${dz.toFixed(2)} mm`;
  $('metric-anisotropy').textContent=`${(dz/c.source_spacing[0]).toFixed(1)}× native in-plane spacing`;
  $('metric-planes').textContent=c.native_slices;
  $('metric-coverage').textContent=`${c.coverage_mm} mm between first / last centers`;
  $('study-kicker').textContent=`${c.name.toUpperCase()} / BHSD → RSNA`;
  $('study-title').textContent=`${dz.toFixed(2)} mm spacing${c.tilt_deg>.1?` · ${c.tilt_deg.toFixed(1)}° tilt`:' · untilted acquisition'}`;
  $('study-tags').replaceChildren();
  for(const tag of ['Hemorrhage-positive',...c.labels.map(titleCase)]){
    const span=document.createElement('span');span.className='tag'+(tag==='Hemorrhage-positive'?' positive':'');span.textContent=tag;$('study-tags').append(span);
  }
  $('geometry-note').textContent=`${c.spacing_xyz[0].toFixed(3)} × ${c.spacing_xyz[2].toFixed(2)} mm coronal pixels; ${c.shape[2]} × ${c.shape[0]} image. Full native 512 × 512 axial samples retained. ${c.tilt_deg>.1?'Tilt corrected using the full NIfTI affine.':'Patient-coordinate coronal reconstruction.'}`;
  $('coronal-slider').max=c.shape[1]-1;$('axial-slider').max=c.native_slices-1;
  $('provenance-details').textContent=JSON.stringify({study:c.source_id,labels:c.labels,native_spacing_mm:c.source_spacing,tilt_degrees:c.tilt_deg,native_slices:c.native_slices,source_affine:c.source_affine,display_affine:c.target_affine,ai:c.ai??'Not available'},null,2);
  if(c.ai){
    $('fit-stat').textContent=`${c.ai.seeds[0].steps} steps × 2 seeds · ${c.ai.seeds[0].parameters.toLocaleString()} parameters`;
    const err=Math.max(...c.ai.seeds.map(s=>s.projected_observed_max_error_hu));
    $('consistency-stat').textContent=`Source-grid max residual < ${Math.max(.001,err).toFixed(3)} HU after projection`;
  }
  document.querySelectorAll('.study-button').forEach(b=>{const active=b.dataset.id===c.id;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
}
async function selectCase(index){
  stopPlay();const token=++generation;current=manifest.cases[index];arrays={};
  coronal=Math.floor(current.shape[1]/2);axial=Math.floor(current.native_slices/2);
  updateMetadata();setBusy(true,'Loading real CT volumes and reconstruction comparisons…');
  for(const canvas of document.querySelectorAll('canvas'))canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
  const methods=['nearest','linear','source'];const selected=activeMethod();
  methods.push(selected==='delta'?'ensemble':selected);
  const id=current.id;
  try{
    const values=await Promise.all(methods.map(m=>readArray(id,m)));
    if(token!==generation)return;
    methods.forEach((m,i)=>arrays[m]=values[i]);
    setBusy(false);renderAll();updateResultsTable();loadEvaluation();
    const url=new URL(location.href);url.searchParams.set('study',id);history.replaceState(null,'',url);
    // Keep only the current and most recently used study to limit memory.
    const ids=[...new Set([...cache.keys()].map(k=>k.split('-').slice(0,2).join('-')))];
    if(ids.length>2)for(const key of cache.keys())if(ids.slice(0,-2).includes(key.split('-').slice(0,2).join('-')))cache.delete(key);
  }catch(error){if(token!==generation)return;setBusy(true,`Unable to load study. ${error.message}. Select another study or reload to retry.`);}
}
function heat(v){
  const stops=[[7,23,39],[34,88,136],[36,187,175],[251,214,122],[255,115,90]];
  const t=clamp(v/30,0,1)*4,i=Math.min(3,Math.floor(t)),f=t-i;
  return stops[i].map((a,j)=>Math.round(a+(stops[i+1][j]-a)*f));
}
function gray(value){const [level,width]=windows[$('window').value];return clamp(Math.round((value-(level-width/2))/width*255),0,255);}
function fitCoronalCanvas(canvas){
  if(!current)return;
  const view=canvas.parentElement;
  const w=canvas.width*current.spacing_xyz[0],h=canvas.height*current.spacing_xyz[2];
  const scale=Math.min((view.clientWidth-28)/w,(view.clientHeight-30)/h);
  canvas.style.width=`${w*scale}px`;canvas.style.height=`${h*scale}px`;
}
const coronalResize=new ResizeObserver(entries=>{for(const {target} of entries){const c=target.querySelector('canvas');if(c)fitCoronalCanvas(c);}});
for(const view of document.querySelectorAll('.image-view'))coronalResize.observe(view);
function drawCoronal(canvas,method){
  const [nz,ny,nx]=current.shape;canvas.width=nx;canvas.height=nz;fitCoronalCanvas(canvas);
  const ctx=canvas.getContext('2d'),image=ctx.createImageData(nx,nz);
  const data=arrays[method==='delta'?'ensemble':method];if(!data)return;
  const isHeat=['disagreement','delta'].includes(method);
  for(let py=0;py<nz;py++)for(let px=0;px<nx;px++){
    const k=((nz-1-py)*ny+coronal)*nx+(nx-1-px),i=(py*nx+px)*4;
    let value=data[k];if(method==='delta')value=Math.abs(value-arrays.linear[k]);
    if(isHeat){const color=heat(value);image.data[i]=color[0];image.data[i+1]=color[1];image.data[i+2]=color[2];}
    else image.data[i]=image.data[i+1]=image.data[i+2]=gray(value);
    image.data[i+3]=255;
  }
  ctx.putImageData(image,0,0);
  if($('plane-overlay').checked){
    const inv=current.inverse_source_affine,t=current.target_affine,[dx,dy,dz]=current.spacing_xyz;
    const worldY=t[1][3]+coronal*dy,worldZtop=t[2][3]+(nz-1)*dz;
    ctx.save();ctx.strokeStyle='#54d0c49c';ctx.lineWidth=.5;ctx.setLineDash([2,3]);
    for(let k=0;k<current.native_slices;k++){
      const at=(x)=>{const wx=t[0][3]+(nx-1-x)*dx;return (worldZtop-(k-inv[2][0]*wx-inv[2][1]*worldY-inv[2][3])/inv[2][2])/dz;};
      ctx.beginPath();ctx.moveTo(0,at(0));ctx.lineTo(nx-1,at(nx-1));ctx.stroke();
    }
    ctx.restore();
  }
}
function drawAxial(){
  if(!arrays.source)return;
  const [nz,ny,nx]=current.source_shape,canvas=$('axial-canvas');canvas.width=nx;canvas.height=ny;
  const ctx=canvas.getContext('2d'),im=ctx.createImageData(nx,ny);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const k=(axial*ny+(ny-1-y))*nx+(nx-1-x),p=(y*nx+x)*4;
    im.data[p]=im.data[p+1]=im.data[p+2]=gray(arrays.source[k]);im.data[p+3]=255;
  }
  ctx.putImageData(im,0,0);
  const a=current.source_affine,worldY=current.target_affine[1][3]+coronal*current.spacing_xyz[1];
  // General intersection line for constant patient Y, including possible in-plane obliquity.
  ctx.strokeStyle='#6bdacf';ctx.lineWidth=.7;ctx.setLineDash([4,2]);ctx.beginPath();
  for(const [i,x] of [0,nx-1].entries()){
    const sx=nx-1-x,sy=(worldY-a[1][3]-a[1][0]*sx-a[1][2]*axial)/a[1][1],y=ny-1-sy;
    if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);
  }
  ctx.stroke();ctx.setLineDash([]);
  $('axial-slider').value=axial;$('axial-position').textContent=`${axial+1} / ${nz}`;
}
function renderAll(){
  if(busy||!current)return;
  drawCoronal($('nearest-canvas'),'nearest');drawCoronal($('linear-canvas'),'linear');drawCoronal($('advanced-canvas'),activeMethod());drawAxial();
  $('coronal-slider').value=coronal;
  const y=current.target_affine[1][3]+coronal*current.spacing_xyz[1];
  $('slice-value').textContent=`${coronal+1} / ${current.shape[1]} · Y ${y.toFixed(1)} mm`;
}
function changeCoronal(next){if(busy)return;coronal=clamp(next,0,current.shape[1]-1);renderAll();}
function changeAxial(next){if(busy)return;axial=clamp(next,0,current.native_slices-1);drawAxial();}
async function changeMethod(event){
  const method=$('advanced-method').value,info=descriptions[method],token=generation;
  $('advanced-pill').textContent=info[0];$('advanced-pill').className='pill '+(method==='cubic'?'baseline':'ai');
  $('advanced-footer-title').textContent=info[1];$('advanced-footer-text').textContent=info[2];$('advanced-label').textContent=info[3];
  $('heat-legend').hidden=!['delta','disagreement'].includes(method);
  if(!current)return;
  const key=method==='delta'?'ensemble':activeMethod();
  $('projection-control').hidden=!pretrainedKeys.has(method);$('transfer-note').hidden=!pretrainedKeys.has(method);
  if(pretrainedKeys.has(method)){ $('advanced-label').textContent=info[3]+($('preserve-source').checked?' · CORRECTED':' · RAW');if(!event||event.target?.id==='advanced-method')$('evaluation-method').value=method; }
  $('eval-preserve-source').checked=$('preserve-source').checked;
  if(!arrays[key]){
    $('load-status').textContent='Loading comparison method…';
    try{const a=await readArray(current.id,key);if(token!==generation)return;arrays[key]=a;$('load-status').textContent='';}
    catch(e){if(token===generation)$('load-status').textContent=`Could not load this method: ${e.message}`;return;}
  }
  for(const name of Object.keys(arrays))if(!['nearest','linear','source',key].includes(name))delete arrays[name];
  renderAll();updateResultsTable();loadEvaluation();
}
function wheelControl(element,callback){
  let last=0;
  element.addEventListener('wheel',event=>{event.preventDefault();if(event.deltaY===0)return;const now=performance.now();if(now-last<35)return;last=now;callback(event.deltaY>0?1:-1);},{passive:false});
  let previousY=null;
  element.addEventListener('pointerdown',e=>{if(e.pointerType==='touch'){previousY=e.clientY;element.setPointerCapture(e.pointerId);}});
  element.addEventListener('pointermove',e=>{if(previousY===null)return;const dy=e.clientY-previousY;if(Math.abs(dy)>7){callback(dy>0?1:-1);previousY=e.clientY;}});
  element.addEventListener('pointerup',()=>previousY=null);element.addEventListener('pointercancel',()=>previousY=null);
  element.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();callback(['ArrowRight','ArrowDown'].includes(e.key)?1:-1);}});
}
for(const pane of document.querySelectorAll('.image-view')){
  wheelControl(pane,step=>changeCoronal(coronal+step));
  pane.querySelector('.expand').addEventListener('click',async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await pane.requestFullscreen();}catch{toast('Fullscreen is unavailable in this browser.');}});
}
wheelControl($('axial-view'),step=>changeAxial(axial+step));
$('axial-view').addEventListener('click',event=>{
  if(busy||event.pointerType==='touch')return;
  const rect=$('axial-canvas').getBoundingClientRect(),[nz,ny,nx]=current.source_shape;
  const sy=(1-(event.clientY-rect.top)/rect.height)*(ny-1),sx=(1-(event.clientX-rect.left)/rect.width)*(nx-1),a=current.source_affine;
  const worldY=a[1][0]*sx+a[1][1]*sy+a[1][2]*axial+a[1][3];
  changeCoronal(Math.round((worldY-current.target_affine[1][3])/current.spacing_xyz[1]));
});
$('coronal-slider').addEventListener('input',e=>changeCoronal(Number(e.target.value)));
$('axial-slider').addEventListener('input',e=>changeAxial(Number(e.target.value)));
$('prev-slice').addEventListener('click',()=>changeCoronal(coronal-1));$('next-slice').addEventListener('click',()=>changeCoronal(coronal+1));
$('window').addEventListener('change',()=>{renderAll();renderEvaluation();});$('preserve-source').addEventListener('change',changeMethod);$('plane-overlay').addEventListener('change',renderAll);$('advanced-method').addEventListener('change',changeMethod);
$('play-button').addEventListener('click',()=>{if(playTimer){stopPlay();return;}if(busy)return;$('play-button').textContent='Ⅱ Pause';$('play-button').setAttribute('aria-label','Pause coronal slices');playTimer=setInterval(()=>changeCoronal((coronal+1)%current.shape[1]),110);});
$('toggle-source').addEventListener('click',()=>{const card=document.querySelector('.source-card');const shown=card.classList.toggle('mobile-visible');$('toggle-source').setAttribute('aria-expanded',String(shown));$('toggle-source').textContent=shown?'Hide acquired-slice navigator':'Show acquired-slice navigator';});
$('about-button').addEventListener('click',()=>$('methods').scrollIntoView({behavior:'smooth'}));
$('download-manifest').addEventListener('click',()=>{if(manifest)saveFile(new Blob([JSON.stringify(manifest,null,2)],{type:'application/json'}),'coronal-lab-metadata.json');});
$('export-button').addEventListener('click',()=>{
  if(busy)return;const canvas=document.createElement('canvas');canvas.width=1800;canvas.height=900;const ctx=canvas.getContext('2d');ctx.fillStyle='#0d1820';ctx.fillRect(0,0,1800,900);ctx.fillStyle='#d9eeeb';ctx.font='24px sans-serif';ctx.fillText(`CORONAL LAB / ${current.name} / ${current.source_spacing[2].toFixed(2)} mm acquired spacing`,30,44);ctx.font='14px sans-serif';ctx.fillStyle='#8da8ac';ctx.fillText(`${$('slice-value').textContent} · ${$('window').selectedOptions[0].textContent} · NIfTI affine / ${current.spacing_xyz[0].toFixed(3)} × ${current.spacing_xyz[2].toFixed(2)} mm coronal pixels`,30,76);
  const titles=['Nearest neighbor','Linear interpolation',$('advanced-method').selectedOptions[0].textContent+ (pretrainedKeys.has($('advanced-method').value)?($('preserve-source').checked?' · corrected':' · raw'):'')];
  for(const [i,id] of ['nearest-canvas','linear-canvas','advanced-canvas'].entries()){
    const source=$(id),pw=source.width*current.spacing_xyz[0],ph=source.height*current.spacing_xyz[2],scale=Math.min(560/pw,600/ph),w=pw*scale,h=ph*scale;
    ctx.fillStyle='#d9eeeb';ctx.font='18px sans-serif';ctx.fillText(titles[i],i*600+25,125);ctx.drawImage(source,i*600+250-w/2,155+(480-h)/2,w,h);
  }
  ctx.fillStyle='#a7c2c3';ctx.font='13px sans-serif';ctx.fillText('Experimental neural estimates. No missing-plane ground truth. Differences and seed disagreement are not error maps.',30,820);ctx.fillText(`Source ${current.source_id} · Native 512 × 512 axial sampling · Research only`,30,850);canvas.toBlob(blob=>{if(blob)saveFile(blob,`${current.id}-coronal-${coronal+1}-${activeMethod()}.png`);});
});

let evalArrays={},evalIndex=0,evalGeneration=0,evalBusy=true,evalStudy=null;
const methodNames={nearest:'Nearest neighbor',linear:'Linear interpolation',cubic:'Cubic interpolation',qdm:'QDM-L · CT diffusion',esrgan:'Real-ESRGAN · GAN',esrnet:'Real-ESRNet · control'};
function evalMethod(){const key=$('evaluation-method').value;return pretrainedKeys.has(key)&&!$('preserve-source').checked?key+'_raw':key;}
function updateResultsTable(){
  if(!current?.evaluation)return;
  $('benchmark-body').hidden=false;$('benchmark-pending').hidden=true;
  const total=manifest.cases.reduce((n,c)=>n+c.evaluation.heldout_count,0);
  $('benchmark-count').textContent=`${total} WITHHELD PLANES · 4 SELECTED STUDIES`;
  const all=$('benchmark-scope').value==='all',cases=all?manifest.cases:[current],corrected=$('preserve-source').checked;
  $('benchmark-projection-note').textContent=corrected?'Acquired-sample correction enabled':'Raw neural output · source correction disabled';
  $('score-rows').replaceChildren();
  for(const method of ['nearest','linear','cubic','qdm','esrgan','esrnet']){
    if(!cases.every(c=>c.evaluation.methods[method]))continue;
    const values=cases.map(c=>!corrected&&pretrainedKeys.has(method)?c.evaluation.raw_methods[method]:c.evaluation.methods[method]);
    const tr=document.createElement('tr');if(method===$('evaluation-method').value)tr.className='selected-row';
    const name=document.createElement('td');name.textContent=methodNames[method];
    if(pretrainedKeys.has(method)){const sub=document.createElement('small');sub.textContent=method==='qdm'?'Head-and-neck CT pretraining':'Natural-image pretraining';name.append(sub);}tr.append(name);
    for(const key of ['soft_tissue_mae_hu','hemorrhage_mae_hu','brain_window_ssim']){const td=document.createElement('td');const valid=values.map(v=>v[key]).filter(v=>v!==null);td.textContent=valid.length?(valid.reduce((a,b)=>a+b,0)/valid.length).toFixed(key==='brain_window_ssim'?3:2):'N/A';tr.append(td);}
    $('score-rows').append(tr);
  }
  const e=current.evaluation,vox=e.methods.linear.hemorrhage_voxels;
  $('benchmark-caveat').textContent=(all?'Scores give each study equal weight. ':'Current-study scores. ')+`${current.name}: ${e.heldout_count} withheld slices; retained spacing ${e.input_spacing_mm.toFixed(2)} mm. ${vox.toLocaleString()} evaluated hemorrhage-mask voxels at the full 512 × 512 in-plane grid.`+(vox<100?' Very small hemorrhage ROI: interpret its MAE cautiously.':'');
}
async function loadEvaluation(){
  if(!current?.evaluation)return;
  const token=++evalGeneration,cid=current.id;evalBusy=true;$('eval-status').textContent='Loading withheld slices…';
  if(evalStudy!==cid){evalIndex=Math.floor(current.evaluation.heldout_count/2);evalStudy=cid;}
  const method=evalMethod(),keys=['reference','linear','lesion',method];
  try{
    const data=await Promise.all(keys.map(k=>readArray(cid,'eval-'+k)));
    if(token!==evalGeneration||cid!==current.id)return;
    evalArrays={};keys.forEach((k,i)=>evalArrays[k]=data[i]);evalBusy=false;$('eval-status').textContent='';$('eval-slider').max=current.evaluation.heldout_count-1;renderEvaluation();
  }catch(e){if(token===evalGeneration)$('eval-status').textContent=`Evaluation unavailable: ${e.message}`;}
}
function drawEval(canvas,array){
  if(!array)return;const [n,ny,nx]=current.evaluation.shape;canvas.width=nx;canvas.height=ny;
  const ctx=canvas.getContext('2d'),im=ctx.createImageData(nx,ny),mask=evalArrays.lesion;
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const k=(evalIndex*ny+(ny-1-y))*nx+(nx-1-x),p=(y*nx+x)*4,v=gray(array[k]);im.data[p]=im.data[p+1]=im.data[p+2]=v;im.data[p+3]=255;
    if($('lesion-overlay').checked&&mask?.[k]){im.data[p]=Math.round(v*.5+127.5);im.data[p+1]=Math.round(v*.5+35);im.data[p+2]=Math.round(v*.5+35);}
  }ctx.putImageData(im,0,0);
}
function renderEvaluation(){
  if(evalBusy||!current?.evaluation||evalStudy!==current.id)return;
  drawEval($('eval-reference'),evalArrays.reference);drawEval($('eval-linear'),evalArrays.linear);drawEval($('eval-model'),evalArrays[evalMethod()]);
  const method=$('evaluation-method').value;$('eval-method-heading').textContent=methodNames[method]+(pretrainedKeys.has(method)?($('preserve-source').checked?' · corrected':' · raw'):'');
  const slice=current.evaluation.heldout_indices[evalIndex];$('eval-slice-label').textContent=`Withheld axial ${slice+1} / ${current.native_slices} · test ${evalIndex+1} / ${current.evaluation.heldout_count}`;$('eval-slider').value=evalIndex;
}
function changeEval(value){if(evalBusy||!current?.evaluation)return;evalIndex=clamp(value,0,current.evaluation.heldout_count-1);renderEvaluation();}
$('eval-slider').addEventListener('input',e=>changeEval(Number(e.target.value)));
$('evaluation-method').addEventListener('change',()=>{updateResultsTable();loadEvaluation();});
$('eval-preserve-source').addEventListener('change',()=>{$('preserve-source').checked=$('eval-preserve-source').checked;changeMethod({target:{id:'eval-preserve-source'}});});
$('benchmark-scope').addEventListener('change',updateResultsTable);$('lesion-overlay').addEventListener('change',renderEvaluation);
for(const view of document.querySelectorAll('.eval-view'))wheelControl(view,step=>changeEval(evalIndex+step));
async function init(){
  try{
    const response=await fetch('assets512/manifest.json',{cache:'no-store'});if(!response.ok)throw new Error(`Manifest HTTP ${response.status}`);manifest=await response.json();
    if(manifest.pretrained_models){
      const pending=manifest.pretrained_pending||[];
      for(const select of [$('advanced-method'),$('evaluation-method')])for(const option of select.options)if(pretrainedKeys.has(option.value))option.disabled=pending.includes(option.value);
      const initial=pending.includes('qdm')?'esrnet':'qdm';$('advanced-method').value=initial;$('evaluation-method').value=initial;
      $('rebuild-status').hidden=!pending.length;
      $('rebuild-status').textContent=pending.length?'Full 512 × 512 source data are live. The full-resolution QDM run was stopped. Its results are unavailable. All selectable methods and benchmark scores use the full-resolution inputs.':'';
      $('qdm-status').textContent=pending.includes('qdm')?'FULL-RESOLUTION RUN STOPPED':'PRETRAINED · MEDICAL CT';
    }
    manifest.cases.forEach((c,index)=>{
      const b=document.createElement('button');b.type='button';b.className='study-button';b.dataset.id=c.id;b.setAttribute('aria-pressed','false');
      const img=document.createElement('img');img.src=`assets512/${c.id}-thumb.png`;img.alt='';
      const info=document.createElement('span'),strong=document.createElement('strong'),small=document.createElement('small'),spacing=document.createElement('span');strong.textContent=c.name;small.textContent=`${c.native_slices} slices · ${c.tilt_deg>.1?'Tilted':'Untilted'}`;spacing.className='study-spacing';spacing.textContent=`${c.source_spacing[2].toFixed(2)} mm`;
      info.append(strong,small,spacing);b.append(img,info);b.addEventListener('click',()=>selectCase(index));$('study-list').append(b);
    });
    const requested=new URL(location.href).searchParams.get('study'),index=manifest.cases.findIndex(c=>c.id===requested);await selectCase(index>=0?index:1);await changeMethod();
    window.coronalLab={getState:()=>({study:current.id,coronal,axial,method:$('advanced-method').value,activeMethod:activeMethod(),evalIndex,evalBusy,busy}),getManifest:()=>manifest};
  }catch(error){$('load-status').textContent=`Dashboard could not load: ${error.message}`;console.error(error);}
}
init();
