const {chromium}=require('/home/ian/projects/eden-report-review/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1512,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`)});
 await page.goto((process.env.CORONAL_URL||'http://100.68.163.27:8773'));
 await page.waitForFunction(()=>window.coronalLab&&!window.coronalLab.getState().busy&&!window.coronalLab.getState().evalBusy);
 const state=()=>page.evaluate(()=>window.coronalLab.getState());
 const dimensions=await page.evaluate(()=>{
   const m=window.coronalLab.getManifest(),c=m.cases.find(c=>c.id===window.coronalLab.getState().study),canvas=document.getElementById('linear-canvas'),rect=canvas.getBoundingClientRect();
   return {source:document.getElementById('axial-canvas').width,width:canvas.width,height:canvas.height,ratio:rect.width/rect.height,expected:canvas.width*c.spacing_xyz[0]/(canvas.height*c.spacing_xyz[2]),cases:m.cases.map(c=>c.source_shape.slice(1))};
 });
 assert.equal(dimensions.source,512);assert.equal(dimensions.width,512);assert.ok(Math.abs(dimensions.ratio-dimensions.expected)<.01);
 for(const shape of dimensions.cases)assert.deepEqual(shape,[512,512]);
 const pending=await page.evaluate(()=>window.coronalLab.getManifest().pretrained_pending||[]),models=['qdm','esrgan','esrnet'].filter(m=>!pending.includes(m));
 assert.equal((await state()).method,pending.includes('qdm')?'esrnet':'qdm');assert.equal(await page.locator('#score-rows tr').count(),3+models.length);
 for(const key of pending)assert.ok(await page.locator(`#advanced-method option[value="${key}"]`).isDisabled());
 assert.match(await page.locator('#benchmark-count').textContent(),/59 WITHHELD/);
 for(const method of models){
   await page.selectOption('#advanced-method',method);await page.waitForFunction(()=>!document.getElementById('load-status').textContent&&!window.coronalLab.getState().evalBusy);
   assert.equal((await state()).activeMethod,method);
   const before=await page.locator('#advanced-canvas').evaluate(c=>c.toDataURL());
   await page.locator('#preserve-source').uncheck();await page.waitForFunction(()=>!document.getElementById('load-status').textContent&&!window.coronalLab.getState().evalBusy);
   assert.equal((await state()).activeMethod,method+'_raw');const after=await page.locator('#advanced-canvas').evaluate(c=>c.toDataURL());assert.notEqual(before,after);
   assert.match(await page.locator('#benchmark-projection-note').textContent(),/Raw/);
   await page.locator('#preserve-source').check();await page.waitForFunction(()=>!document.getElementById('load-status').textContent&&!window.coronalLab.getState().evalBusy);
 }
 await page.selectOption('#benchmark-scope','all');
 const manifest=await page.evaluate(()=>window.coronalLab.getManifest());
 const expected=(manifest.cases.reduce((v,c)=>v+c.evaluation.methods.linear.soft_tissue_mae_hu,0)/4).toFixed(2);
 assert.equal(await page.locator('#score-rows tr').nth(1).locator('td').nth(1).textContent(),expected);
 let before=await state();await page.locator('.eval-view').first().hover();await page.mouse.wheel(0,100);await page.waitForTimeout(150);assert.equal((await state()).evalIndex,before.evalIndex+1);
 await page.locator('#eval-slider').fill('0');assert.equal((await state()).evalIndex,0);
 await page.locator('#lesion-overlay').check();await page.selectOption('#evaluation-method','cubic');await page.waitForFunction(()=>!window.coronalLab.getState().evalBusy);
 for(const id of ['case-01','case-03','case-04','case-02']){
  await page.locator(`.study-button[data-id="${id}"]`).click();await page.waitForFunction(id=>window.coronalLab.getState().study===id&&!window.coronalLab.getState().busy&&!window.coronalLab.getState().evalBusy,id);
 }
 await page.selectOption('#advanced-method',models[0]);await page.waitForFunction(()=>!document.getElementById('load-status').textContent&&!window.coronalLab.getState().evalBusy);
 await page.selectOption('#benchmark-scope','current');await page.locator('#lesion-overlay').uncheck();
 await page.screenshot({path:'/home/ian/projects/ich/sites/coronal-lab/work/pretrained-desktop.png',fullPage:true});
 await page.locator('#benchmark').screenshot({path:'/home/ian/projects/ich/sites/coronal-lab/work/benchmark-panel.png'});
 const mobile=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});await mobile.goto((process.env.CORONAL_URL||'http://100.68.163.27:8773'));await mobile.waitForFunction(()=>window.coronalLab&&!window.coronalLab.getState().busy&&!window.coronalLab.getState().evalBusy);
 assert.ok(await mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await mobile.locator('#toggle-source').click();assert.ok(await mobile.locator('.source-card').isVisible());await mobile.locator('#toggle-source').click();
 await mobile.screenshot({path:'/home/ian/projects/ich/sites/coronal-lab/work/pretrained-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);console.log(JSON.stringify({result:'passed',checks:[`${models.length} completed pretrained models`,`${pending.length} pending models disabled`,'raw/corrected volumes differ','raw/corrected score state','available-model table','aggregate score calculation','withheld slice scrolling','hemorrhage overlay','4 studies','mobile layout and source navigator'],browserErrors:errors}));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
