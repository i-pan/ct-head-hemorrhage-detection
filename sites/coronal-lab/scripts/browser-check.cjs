const {chromium}=require('/home/ian/projects/eden-report-review/node_modules/playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1512,height:1100}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`)});
 await page.goto((process.env.CORONAL_URL||'http://100.68.163.27:8773'));
 await page.waitForFunction(()=>window.coronalLab&&!window.coronalLab.getState().busy);
 await page.screenshot({path:'/home/ian/projects/ich/sites/coronal-lab/work/dashboard-desktop.png',fullPage:true});
 const state=()=>page.evaluate(()=>window.coronalLab.getState());
 const dimensions=await page.evaluate(()=>{
   const m=window.coronalLab.getManifest(),c=m.cases.find(c=>c.id===window.coronalLab.getState().study),canvas=document.getElementById('linear-canvas'),rect=canvas.getBoundingClientRect();
   return {source:document.getElementById('axial-canvas').width,width:canvas.width,height:canvas.height,ratio:rect.width/rect.height,expected:canvas.width*c.spacing_xyz[0]/(canvas.height*c.spacing_xyz[2]),cases:m.cases.map(c=>c.source_shape.slice(1))};
 });
 assert.equal(dimensions.source,512);assert.equal(dimensions.width,512);assert.ok(Math.abs(dimensions.ratio-dimensions.expected)<.01);
 for(const shape of dimensions.cases)assert.deepEqual(shape,[512,512]);
 let before=await state();await page.locator('[data-method="linear"]').hover();await page.mouse.wheel(0,100);await page.waitForTimeout(150);assert.equal((await state()).coronal,before.coronal+1);
 await page.locator('#coronal-slider').fill('0');await page.locator('#prev-slice').click();assert.equal((await state()).coronal,0);
 await page.locator('#next-slice').click();assert.equal((await state()).coronal,1);
 await page.locator('#axial-view').hover();before=await state();await page.mouse.wheel(0,100);await page.waitForTimeout(100);assert.equal((await state()).axial,before.axial+1);
 for(const method of ['cubic','dip','ensemble','disagreement','delta']){await page.selectOption('#advanced-method',method);await page.waitForFunction(()=>!document.getElementById('load-status').textContent);assert.equal((await state()).method,method);}
 await page.locator('#plane-overlay').check();await page.selectOption('#window','subdural');await page.locator('#play-button').click();before=await state();await page.waitForTimeout(400);await page.locator('#play-button').click();assert.ok((await state()).coronal>before.coronal);
 const downloadPromise=page.waitForEvent('download');await page.locator('#export-button').click();const download=await downloadPromise;await download.saveAs('/home/ian/projects/ich/sites/coronal-lab/work/qa-export.png');
 for(const id of ['case-01','case-03','case-04','case-02']){await page.locator(`.study-button[data-id="${id}"]`).click();await page.waitForFunction(id=>window.coronalLab.getState().study===id&&!window.coronalLab.getState().busy,id);}
 await page.selectOption('#advanced-method','ensemble');await page.selectOption('#window','brain');await page.locator('#plane-overlay').uncheck();
 await page.screenshot({path:'/home/ian/projects/ich/sites/coronal-lab/work/dashboard-final.png',fullPage:true});
 const mobile=await browser.newPage({viewport:{width:390,height:844},isMobile:true,hasTouch:true});await mobile.goto((process.env.CORONAL_URL||'http://100.68.163.27:8773'));await mobile.waitForFunction(()=>window.coronalLab&&!window.coronalLab.getState().busy);
 assert.ok(await mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await mobile.screenshot({path:'/home/ian/projects/ich/sites/coronal-lab/work/dashboard-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);console.log(JSON.stringify({result:'passed',checks:['tailnet HTTP','4 studies','synchronized wheel navigation','axial scrolling','slice boundaries','5 comparison methods','window preset','plane overlay','cine playback','PNG export','mobile layout'],browserErrors:errors}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
