"""
build_walk_galleries.py — one visualization gallery per walk, from the per-step geojsons
your pipeline already writes into walk_maps/.

For each <stem> it looks for (any subset works):
    <stem>_tagged.geojson     property 'tag'      -> ground-truth surface
    <stem>_surface.geojson    property 'surface'  -> GIS (3-layer) surface
    <stem>_pred_3class.geojson property 'surface' -> model predicted surface
    <stem>_evenness.geojson   property 'value'    -> evenness 0-100
    <stem>_hardness.geojson   property 'value'    -> hardness 0-100
All are per-segment over the same GPX points, so segment i lines up across layers.

Writes <stem>_gallery.html per walk (horizon, ridgeline-by-surface, beeswarm,
alluvial, slope-by-surface) plus an index.html linking them.

Usage:
    python build_walk_galleries.py GPS/walk_maps
    python build_walk_galleries.py GPS/walk_maps --out galleries
"""
import argparse, os, glob, json, html

LAYERS = {  # stem suffix -> (property key, output field)
    'tagged':     ('tag',     'truth'),
    'surface':    ('surface', 'gis'),
    'pred_3class':('surface', 'pred'),
    'evenness':   ('value',   'ev'),
    'hardness':   ('value',   'hd'),
}

def load_layer(path, key):
    try: fc = json.load(open(path, encoding='utf-8'))
    except Exception: return None
    out = []
    for f in fc.get('features', []):
        out.append(f.get('properties', {}).get(key))
    return out

def load_coords(path):
    try: fc = json.load(open(path, encoding='utf-8'))
    except Exception: return None
    pts = []
    for f in fc.get('features', []):
        c = f.get('geometry', {}).get('coordinates')
        if c: pts.append([c[0][0], c[0][1]])   # [lon,lat] of segment start
    return pts

def assemble_walk(out_dir, stem):
    cols = {}
    for suffix, (key, field) in LAYERS.items():
        p = os.path.join(out_dir, f"{stem}_{suffix}.geojson")
        if os.path.exists(p):
            v = load_layer(p, key)
            if v: cols[field] = v
    if not cols: return []
    # route coordinates from whichever layer is present
    coords = None
    for suffix in ['evenness','surface','tagged','hardness','pred_3class']:
        p = os.path.join(out_dir, f"{stem}_{suffix}.geojson")
        if os.path.exists(p):
            coords = load_coords(p)
            if coords: break
    n = min(len(v) for v in cols.values())
    if coords: n = min(n, len(coords))
    steps = []
    for i in range(n):
        row = {'i': i}
        for field, v in cols.items():
            val = v[i]
            row[field] = (None if val is None else float(val)) if field in ('ev','hd') else (val or 'unknown')
        if coords: row['lon'], row['lat'] = coords[i]
        steps.append(row)
    return steps

CHARTS_JS = r"""
const COL={paved:'#607D8B',grass:'#4CAF50',loose:'#8D6E63',hard:'#607D8B',soft:'#4CAF50',dirt:'#8D6E63',unknown:'#BDBDBD'};
const catcol=s=>COL[(s||'unknown').toLowerCase()]||COL.unknown;
const RAMP_EV=[[0,[215,48,39]],[0.5,[254,224,139]],[1,[26,152,80]]];
function ramp(v,R){let t=Math.max(0,Math.min(1,v/100));for(let i=1;i<R.length;i++){if(t<=R[i][0]){const[a,c0]=R[i-1],[b,c1]=R[i],f=(t-a)/((b-a)||1);return`rgb(${c0.map((x,k)=>Math.round(x+(c1[k]-x)*f)).join(',')})`;}}return`rgb(${R[R.length-1][1].join(',')})`;}
const SVGNS='http://www.w3.org/2000/svg';
const el=(t,a)=>{const n=document.createElementNS(SVGNS,t);for(const k in a)n.setAttribute(k,a[k]);return n;};
const tip=document.getElementById('tip');
const showTip=(e,h)=>{tip.innerHTML=h;tip.style.opacity=1;tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';};
const hideTip=()=>{tip.style.opacity=0;};
const mean=a=>a.reduce((s,v)=>s+v,0)/(a.length||1);
const MIN_STEPS=5;
const groupKey = STEPS.some(s=>s.truth&&s.truth!=='unknown')?'truth':STEPS.some(s=>s.gis&&s.gis!=='unknown')?'gis':'pred';
const ORDER=['paved','loose','grass','soft','hard'];
function counts(f){const c={};STEPS.forEach(s=>{const v=s[f];if(v&&v!=='unknown')c[v]=(c[v]||0)+1;});return c;}
function present(f,m){const c=counts(f);const set=Object.keys(c).filter(k=>c[k]>=(m||1));return ORDER.filter(o=>set.includes(o)).concat(set.filter(o=>!ORDER.includes(o)));}
function omitted(f,m){const c=counts(f);return Object.keys(c).filter(k=>c[k]<(m||1)).map(k=>`${k}: ${c[k]}`);}
function has(f){return STEPS.some(s=>s[f]!==undefined&&s[f]!==null&&s[f]!=='unknown');}
function note(id,msg){const d=document.getElementById(id);if(d)d.innerHTML='<p class="na">'+msg+'</p>';}

// ---- 1. route ribbon: Leaflet map if available, SVG route fallback if not; profile always ----
function routeRibbon(){
  const rp=STEPS.filter(s=>s.lat!=null&&s.lon!=null&&isFinite(s.lat)&&isFinite(s.lon)
      &&!(s.lat===0&&s.lon===0)&&Math.abs(s.lat)<=90&&Math.abs(s.lon)<=180);
  const prof=document.getElementById('profile'), box=document.getElementById('routemap');
  if(rp.length<2){note('profile','no route coordinates for this walk');if(box)box.style.display='none';return;}

  // profile (independent of the map so it always renders)
  const svg=el('svg',{viewBox:'0 0 1000 120'});prof.appendChild(svg);
  const W=1000,H=120,padL=28,padR=10,padT=12,padB=20,n=rp.length;
  const x=i=>padL+i/(n-1)*(W-padL-padR),y=v=>padT+(1-v/100)*(H-padT-padB);
  [0,50,100].forEach(v=>{svg.appendChild(el('line',{x1:padL,y1:y(v),x2:W-padR,y2:y(v),stroke:v===50?'#c7d0cb':'#eef2f0','stroke-dasharray':v===50?'3 3':''}));
    const t=el('text',{x:padL-4,y:y(v)+3,'text-anchor':'end',class:'ax'});t.textContent=v;svg.appendChild(t);});
  for(let i=0;i<n-1;i++){if(rp[i].ev==null||rp[i+1].ev==null)continue;const v=(rp[i].ev+rp[i+1].ev)/2;
    svg.appendChild(el('path',{d:`M ${x(i)} ${y(0)} L ${x(i).toFixed(1)} ${y(rp[i].ev).toFixed(1)} L ${x(i+1).toFixed(1)} ${y(rp[i+1].ev).toFixed(1)} L ${x(i+1)} ${y(0)} Z`,fill:ramp(v,RAMP_EV),'fill-opacity':0.9}));}
  svg.appendChild(el('text',{x:padL,y:10,class:'ax'})).textContent='evenness along the route (hover to locate)';
  let locate=()=>{};
  const hit=el('rect',{x:0,y:0,width:W,height:H,fill:'transparent'});svg.appendChild(hit);
  hit.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect();let i=Math.round((e.clientX-r.left)/r.width*W/((W-padL-padR)/(n-1)));i=Math.max(0,Math.min(n-1,i));
    locate(i);showTip(e,`step ${rp[i].i} · ${rp[i].ev==null?'evenness n/a':'evenness '+rp[i].ev.toFixed(0)} · ${rp[i][groupKey]||''}`);});
  hit.addEventListener('mouseleave',()=>{locate(-1);hideTip();});

  // map: real Leaflet basemap — mirrors build_all_walks exactly (immediate fitBounds, no deferral)
  let ok=false, status='';
  if(typeof L!=='undefined'){ try{
    const map=L.map('routemap',{renderer:L.svg({padding:0.6}),preferCanvas:false});
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenStreetMap contributors',maxZoom:19}).addTo(map);
    let nseg=0; const latlngs=rp.map(p=>[p.lat,p.lon]);
    L.polyline(latlngs,{color:'#111',weight:8,opacity:.9,lineCap:'round',lineJoin:'round'}).addTo(map);   // dark casing under the route
    for(let i=0;i<n-1;i++){let v=(rp[i].ev!=null&&rp[i+1].ev!=null)?(rp[i].ev+rp[i+1].ev)/2:(rp[i].ev!=null?rp[i].ev:rp[i+1].ev);
      L.polyline([[rp[i].lat,rp[i].lon],[rp[i+1].lat,rp[i+1].lon]],{color:v==null?'#BDBDBD':ramp(v,RAMP_EV),weight:5,opacity:1,lineCap:'round',lineJoin:'round'}).addTo(map);nseg++;}
    let bd=null;rp.forEach(p=>{const ll=L.latLng(p.lat,p.lon);bd=bd?bd.extend(ll):L.latLngBounds(ll,ll);});
    map.fitBounds(bd,{padding:[20,20]});
    setTimeout(()=>map.invalidateSize(),200);
    const hlm=L.circleMarker(bd.getCenter(),{radius:7,color:'#111',weight:2,fillColor:'#ffeb3b',fillOpacity:.95});
    locate=i=>{if(i<0){if(map.hasLayer(hlm))map.removeLayer(hlm);return;}hlm.setLatLng([rp[i].lat,rp[i].lon]);if(!map.hasLayer(hlm))hlm.addTo(map);};
    ok=true; status=`map: Leaflet, ${nseg} segments, zoom ${map.getZoom()}`;
  }catch(e){ ok=false; status='Leaflet error: '+e.message; } }
  else { status='Leaflet not loaded (typeof L undefined)'; }
  // tiny status line so the map path is visible/reportable
  const st=document.createElement('div'); st.className='na'; st.style.cssText='padding:2px 0;font-size:11px'; st.textContent=status; prof.appendChild(st);

  if(!ok){  // SVG fallback (Leaflet/basemap unavailable) — draw the route on a blank panel
    box.style.background='#eef2f0';const bw=box.clientWidth||900,bh=box.clientHeight||338,pad=18;
    const xs=rp.map(p=>p.lon),ys=rp.map(p=>p.lat);const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
    const sx=v=>pad+(v-x0)/((x1-x0)||1)*(bw-2*pad),sy=v=>bh-pad-(v-y0)/((y1-y0)||1)*(bh-2*pad);
    const m=el('svg',{viewBox:`0 0 ${bw} ${bh}`,style:'width:100%;height:100%'});box.appendChild(m);
    let cpts=rp.map(p=>`${sx(p.lon).toFixed(1)},${sy(p.lat).toFixed(1)}`).join(' ');
    m.appendChild(el('polyline',{points:cpts,fill:'none',stroke:'#111','stroke-width':8,'stroke-linecap':'round','stroke-linejoin':'round','stroke-opacity':0.9}));  // dark casing
    for(let i=0;i<n-1;i++){let v=(rp[i].ev!=null&&rp[i+1].ev!=null)?(rp[i].ev+rp[i+1].ev)/2:(rp[i].ev!=null?rp[i].ev:rp[i+1].ev);
      m.appendChild(el('line',{x1:sx(rp[i].lon),y1:sy(rp[i].lat),x2:sx(rp[i+1].lon),y2:sy(rp[i+1].lat),stroke:v==null?'#BDBDBD':ramp(v,RAMP_EV),'stroke-width':5,'stroke-linecap':'round'}));}
    m.appendChild(el('circle',{cx:sx(rp[0].lon),cy:sy(rp[0].lat),r:5,fill:'#2e7d32'}));
    m.appendChild(el('circle',{cx:sx(rp[n-1].lon),cy:sy(rp[n-1].lat),r:5,fill:'#c62828'}));
    const dot=el('circle',{r:6,fill:'none',stroke:'#111','stroke-width':2,opacity:0});m.appendChild(dot);
    locate=i=>{if(i<0){dot.setAttribute('opacity',0);return;}dot.setAttribute('cx',sx(rp[i].lon));dot.setAttribute('cy',sy(rp[i].lat));dot.setAttribute('opacity',1);};
    const cap=document.createElement('div');cap.className='na';cap.style.padding='4px 0';cap.textContent='basemap offline — route shown without map tiles';prof.parentNode.insertBefore(cap,prof);
  }
}

// ---- 2. raincloud (evenness by surface) ----
function raincloud(id){
  if(!has('ev')||!has(groupKey)){note(id,'needs evenness + surface labels');return;}
  const groups=present(groupKey,MIN_STEPS);const host=document.getElementById(id);
  const om=omitted(groupKey,MIN_STEPS);const noteH=om.length?18:0;
  const W=460,H=Math.max(180,60+groups.length*80)+noteH,padL=8,padR=8,padT=10+noteH,padB=26;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`});host.appendChild(svg);
  if(om.length){const t=el('text',{x:padL,y:12,class:'ax'});t.textContent='omitted (<'+MIN_STEPS+' steps): '+om.join(', ');svg.appendChild(t);}
  const rowH=(H-padT-padB)/groups.length,x=v=>padL+v/100*(W-padL-padR);
  groups.forEach((g,gi)=>{const vals=STEPS.filter(s=>s[groupKey]===g&&s.ev!=null).map(s=>s.ev).sort((a,b)=>a-b);if(!vals.length)return;
    const yB=padT+gi*rowH+rowH*0.6;const bins=40,h=new Array(bins).fill(0);vals.forEach(v=>h[Math.min(bins-1,Math.floor(v/100*bins))]++);
    const sm=h.map((_,i)=>{let a=0,c=0;for(let j=-2;j<=2;j++){const k=i+j;if(k>=0&&k<bins){a+=h[k];c++;}}return a/c;});
    const mx=Math.max(...sm)||1,amp=rowH*0.5;let d=`M ${x(0)} ${yB}`;sm.forEach((v,i)=>d+=` L ${x((i+0.5)/bins*100).toFixed(1)} ${(yB-v/mx*amp).toFixed(1)}`);d+=` L ${x(100)} ${yB} Z`;
    svg.appendChild(el('path',{d,fill:catcol(g),'fill-opacity':0.5,stroke:catcol(g)}));
    vals.filter(()=>Math.random()<0.6).forEach(v=>svg.appendChild(el('circle',{cx:x(v),cy:yB+8+Math.random()*(rowH*0.22),r:1.6,fill:catcol(g),'fill-opacity':0.5})));
    const q=pp=>vals[Math.floor(pp*(vals.length-1))];svg.appendChild(el('rect',{x:x(q(.25)),y:yB+2,width:Math.max(1,x(q(.75))-x(q(.25))),height:5,fill:catcol(g)}));
    svg.appendChild(el('line',{x1:x(q(.5)),y1:yB,x2:x(q(.5)),y2:yB+7,stroke:'#fff','stroke-width':1.5}));
    const t=el('text',{x:padL,y:yB-amp-2,class:'ax'});t.textContent=`${g} (${vals.length})`;svg.appendChild(t);});
  [0,25,50,75,100].forEach(v=>{const t=el('text',{x:x(v),y:H-8,'text-anchor':'middle',class:'ax'});t.textContent=v;svg.appendChild(t);});
}

// ---- 3. beeswarm (kept) ----
function beeswarm(id){
  if(!has('ev')||!has(groupKey)){note(id,'needs evenness + surface labels');return;}
  const host=document.getElementById(id);const groups=present(groupKey,MIN_STEPS);
  const W=460,H=300,padT=14,padB=26;const colW=(W-16)/groups.length;const maxHalf=Math.min(colW*0.42,60);
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`});host.appendChild(svg);
  const cx=k=>8+(groups.indexOf(k)+0.5)*colW;const y=v=>padT+(1-v/100)*(H-padT-padB);
  [0,25,50,75,100].forEach(v=>{svg.appendChild(el('line',{x1:0,y1:y(v),x2:W,y2:y(v),stroke:'#eef2f0'}));const t=el('text',{x:2,y:y(v)-2,class:'ax'});t.textContent=v;svg.appendChild(t);});
  groups.forEach(g=>{const arr=STEPS.filter(s=>s[groupKey]===g&&s.ev!=null).sort((a,b)=>a.ev-b.ev);const bk={};
    arr.forEach(p=>{const yy=Math.round(y(p.ev)/3);const off=bk[yy]||0;const dir=(off%2===0?1:-1)*Math.ceil(off/2);
      let px=cx(g)+dir*3.4;px=Math.max(cx(g)-maxHalf,Math.min(cx(g)+maxHalf,px));bk[yy]=off+1;const c=el('circle',{cx:px.toFixed(1),cy:y(p.ev).toFixed(1),r:2.2,fill:catcol(g),'fill-opacity':0.5});
      c.addEventListener('mousemove',e=>showTip(e,`${g} · step ${p.i} · evenness ${p.ev.toFixed(0)}`));c.addEventListener('mouseleave',hideTip);svg.appendChild(c);});
    if(arr.length){const m=arr.reduce((a,p)=>a+p.ev,0)/arr.length;svg.appendChild(el('line',{x1:cx(g)-26,y1:y(m),x2:cx(g)+26,y2:y(m),stroke:catcol(g),'stroke-width':2.5}));}
    const t=el('text',{x:cx(g),y:H-8,'text-anchor':'middle',class:'ax'});t.textContent=g;svg.appendChild(t);});
}

// ---- 4. alluvial (kept) ----
function alluvial(id){
  const cols=['truth','gis','pred'].filter(has);
  if(cols.length<2){note(id,'needs at least two of truth / GIS / model');return;}
  const host=document.getElementById(id);const W=1000,H=300,padT=18,padB=22;const svg=el('svg',{viewBox:`0 0 ${W} ${H}`});host.appendChild(svg);
  const labMap={truth:'ground truth',gis:'GIS map',pred:'model'};const colX=cols.map((_,i)=>120+i*(W-240)/(cols.length-1));
  const nodeW=15,gapY=14;const rows=STEPS.filter(s=>cols.every(c=>s[c]&&s[c]!=='unknown'));const N=rows.length||1;const totalH=H-padT-padB;
  const nodes={};cols.forEach(c=>{let y=padT;present(c).forEach(o=>{const h=rows.filter(s=>s[c]===o).length/N*(totalH-gapY*2);nodes[c+'|'+o]={y0:y,y1:y+h};y+=h+gapY;});});
  function ribbons(ca,cb,xa,xb){const src={},dst={};present(ca).forEach(o=>src[o]=nodes[ca+'|'+o].y0);present(cb).forEach(o=>dst[o]=nodes[cb+'|'+o].y0);
    present(ca).forEach(o1=>present(cb).forEach(o2=>{const cnt=rows.filter(s=>s[ca]===o1&&s[cb]===o2).length;if(!cnt)return;
      const h=cnt/N*(totalH-gapY*2);const ya0=src[o1],ya1=ya0+h;src[o1]=ya1;const yb0=dst[o2],yb1=yb0+h;dst[o2]=yb1;const xm=(xa+xb)/2;
      const d=`M ${xa+nodeW} ${ya0} C ${xm} ${ya0}, ${xm} ${yb0}, ${xb} ${yb0} L ${xb} ${yb1} C ${xm} ${yb1}, ${xm} ${ya1}, ${xa+nodeW} ${ya1} Z`;
      const p=el('path',{d,fill:catcol(o1),'fill-opacity':o1===o2?0.4:0.62});
      p.addEventListener('mousemove',e=>showTip(e,`${labMap[ca]}: ${o1} → ${o2} · ${cnt} steps${o1!==o2?' (disagree)':''}`));p.addEventListener('mouseleave',hideTip);svg.appendChild(p);}));}
  for(let i=0;i<cols.length-1;i++)ribbons(cols[i],cols[i+1],colX[i],colX[i+1]);
  cols.forEach((c,ci)=>{present(c).forEach(o=>{const nd=nodes[c+'|'+o];svg.appendChild(el('rect',{x:colX[ci],y:nd.y0,width:nodeW,height:Math.max(1,nd.y1-nd.y0),fill:catcol(o),rx:2}));});
    const t=el('text',{x:colX[ci]+nodeW/2,y:padT-5,'text-anchor':'middle',class:'ax'});t.textContent=labMap[c];svg.appendChild(t);});
}

// ---- 5. confusion matrices (truth x GIS, truth x model) ----
function confusion(id){
  if(!has('truth')||(!has('gis')&&!has('pred'))){note(id,'needs ground truth + GIS or model');return;}
  const host=document.getElementById(id);const labels=present('truth').length?present('truth'):['paved','grass','loose'];
  const which=[['gis','truth × GIS'],['pred','truth × model']].filter(w=>has(w[0]));
  const cell=38,gap=2,stride=180;const W=460,H=30+which.length*stride;const svg=el('svg',{viewBox:`0 0 ${W} ${H}`});host.appendChild(svg);
  which.forEach(([key,title],mi)=>{const oy=40+mi*stride,ox=70;
    const M={};labels.forEach(a=>labels.forEach(b=>M[a+'|'+b]=0));STEPS.forEach(p=>{if(p.truth&&p.truth!=='unknown'&&p[key]&&p[key]!=='unknown'&&M[p.truth+'|'+p[key]]!==undefined)M[p.truth+'|'+p[key]]++;});
    const mx=Math.max(1,...Object.values(M));
    const t=el('text',{x:ox+cell*labels.length/2,y:oy-12,'text-anchor':'middle',class:'ax'});t.textContent=title;svg.appendChild(t);
    labels.forEach((a,r)=>labels.forEach((b,c)=>{const v=M[a+'|'+b];const x=ox+c*(cell+gap),y=oy+r*(cell+gap);
      const rc=el('rect',{x,y,width:cell,height:cell,rx:3,fill:a===b?'#2f855a':'#c05621','fill-opacity':0.12+0.82*v/mx});
      rc.addEventListener('mousemove',e=>showTip(e,`truth ${a} · ${title.includes('GIS')?'GIS':'model'} ${b} · ${v}`));rc.addEventListener('mouseleave',hideTip);svg.appendChild(rc);
      const tx=el('text',{x:x+cell/2,y:y+cell/2+4,'text-anchor':'middle','font-size':11.5,fill:'#1a201e'});tx.textContent=v;svg.appendChild(tx);}));
    labels.forEach((a,r)=>{const tt=el('text',{x:ox-6,y:oy+r*(cell+gap)+cell/2+3,'text-anchor':'end',class:'ax'});tt.textContent=a;svg.appendChild(tt);});
    labels.forEach((b,c)=>{const tt=el('text',{x:ox+c*(cell+gap)+cell/2,y:oy+labels.length*(cell+gap)+12,'text-anchor':'middle',class:'ax'});tt.textContent=b;svg.appendChild(tt);});});
}

routeRibbon();   // handles Leaflet-or-not internally (SVG fallback if unavailable)
raincloud('rain'); beeswarm('swarm'); alluvial('alluvial'); confusion('confusion');
"""

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{--paper:#f5f7f6;--panel:#fff;--ink:#1a201e;--muted:#78827d;--line:#e3e8e5}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;font-feature-settings:"tnum" 1;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:40px 26px 90px}
a.back{color:var(--muted);text-decoration:none;font-size:13.5px}
h1{font-size:26px;font-weight:650;letter-spacing:-0.01em;margin:8px 0 4px}
.sub{color:var(--muted);margin:0 0 8px}
section{margin-top:40px}.sec-h{display:flex;align-items:baseline;gap:12px;margin-bottom:2px}
.sec-h h2{font-size:15px;font-weight:650;margin:0}.sec-h .q{color:var(--muted);font-size:14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin-top:16px}
.panel h3{font-size:16px;font-weight:620;margin:0 0 3px}.cap{color:var(--muted);font-size:13.5px;margin:0 0 16px;max-width:72ch}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
#profile svg,#rain svg,#swarm svg,#alluvial svg,#confusion svg,#routemap>svg{display:block;width:100%;height:auto;overflow:visible}text{fill:var(--ink)}.ax{fill:var(--muted);font-size:11px}/* never let the chart-svg sizing touch Leaflet's own overlay svg */.leaflet-container svg{width:auto;height:auto}
.na{color:var(--muted);font-size:13px;font-style:italic;padding:20px 0}
#routemap{height:340px;border-radius:8px;overflow:hidden;border:1px solid var(--line)}
.lgd{display:flex;gap:16px;flex-wrap:wrap;color:var(--muted);font-size:12.5px;margin-top:12px}
.sw{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:-1px;margin-right:5px}
.grad{height:10px;border-radius:3px;background:linear-gradient(to right,#d73027,#fee08b,#1a9850);margin-top:10px}
.grad-l{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}
.tip{position:fixed;pointer-events:none;background:#11201b;color:#eef5f1;font-size:12px;padding:5px 8px;border-radius:6px;opacity:0;transition:opacity .08s;z-index:1000;white-space:nowrap}
@media(max-width:720px){.grid2{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<a class="back" href="index.html">&larr; all walks</a>
<h1>__TITLE__</h1>
<p class="sub">__NSTEPS__ steps. Evenness is the unsupervised score. Hover any mark for detail.</p>

<section><div class="sec-h"><h2>On the map</h2><span class="q">where is the route rough or smooth?</span></div>
<div class="panel"><h3>Route ribbon with linked profile</h3>
<p class="cap">The track on the map, colored by evenness (red rough to green smooth). Hover the profile below and the matching spot lights up on the map.</p>
<div id="routemap"></div><div id="profile"></div>
<div class="grad"></div><div class="grad-l"><span>0 rough</span><span>100 smooth</span></div></div></section>

<section><div class="sec-h"><h2>Distribution by surface</h2><span class="q">how does evenness spread within each surface?</span></div>
<div class="grid2">
<div class="panel"><h3>Raincloud</h3><p class="cap">Per surface: a density cloud above, the raw steps as rain below, and a box for the middle 50%. Shape and every point at once.</p><div id="rain"></div></div>
<div class="panel"><h3>Beeswarm</h3><p class="cap">Every step as a dot, grouped by surface, nudged apart so density shows. The bar marks each surface's mean evenness.</p><div id="swarm"></div></div></div></section>

<section><div class="sec-h"><h2>Where the layers agree</h2><span class="q">truth vs map vs model</span></div>
<div class="panel"><h3>Alluvial flow</h3><p class="cap">Each step flows through ground truth, the GIS map, then the model. Straight ribbons are agreement; crossing ribbons are disagreement.</p><div id="alluvial"></div>
<div class="lgd"><span><span class="sw" style="background:#607D8B"></span>paved</span><span><span class="sw" style="background:#4CAF50"></span>grass</span><span><span class="sw" style="background:#8D6E63"></span>loose</span></div></div>
<div class="panel"><h3>Confusion matrices</h3><p class="cap">Ground truth against the GIS map and the model, as counts. The diagonal is agreement; off-diagonal cells show which surfaces get confused for which.</p><div id="confusion"></div></div></section>

</div><div class="tip" id="tip"></div>
<script>const STEPS=__STEPS__;</script>
<script>__CHARTS__</script>
</body></html>"""

INDEX = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Walk galleries</title>
<style>:root{--paper:#f5f7f6;--ink:#1a201e;--muted:#78827d;--line:#e3e8e5;--panel:#fff}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;font-feature-settings:"tnum" 1}
.wrap{max-width:1000px;margin:0 auto;padding:48px 26px}h1{font-size:28px;font-weight:650;margin:0 0 4px}
.sub{color:var(--muted);margin:0 0 24px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin-bottom:26px}
.panel h3{font-size:16px;font-weight:620;margin:0 0 3px}.cap{color:var(--muted);font-size:13.5px;margin:0 0 14px}
svg{display:block;width:100%;height:auto;overflow:visible}.ax{fill:var(--muted);font-size:11px}
ul{list-style:none;padding:0;margin:0}li{border:1px solid var(--line);border-radius:9px;margin-bottom:10px;background:#fff}
a{display:flex;justify-content:space-between;padding:14px 18px;text-decoration:none;color:var(--ink);font-size:16px}a:hover{background:#eef2f0}.n{color:var(--muted);font-size:13.5px}
.tip{position:fixed;pointer-events:none;background:#11201b;color:#eef5f1;font-size:12px;padding:5px 8px;border-radius:6px;opacity:0;transition:opacity .08s;z-index:10}
</style></head><body><div class="wrap">
<h1>Walk galleries</h1><p class="sub">__COUNT__ walks. Each page shows the route ribbon, raincloud, beeswarm, alluvial, and confusion matrices.</p>
<div class="panel"><h3>Evenness heatmap — steps by walk</h3>
<p class="cap">One row per walk, distance left to right, colored by the unsupervised evenness score. Scan every walk's rough spots at once.</p>
<svg id="heat" viewBox="0 0 1000 __HEATH__"></svg></div>
<ul>__ITEMS__</ul></div><div class="tip" id="tip"></div>
<script>const HEAT=__HEAT__;</script>
<script>
const SVGNS='http://www.w3.org/2000/svg',el=(t,a)=>{const n=document.createElementNS(SVGNS,t);for(const k in a)n.setAttribute(k,a[k]);return n;};
const tip=document.getElementById('tip'),showTip=(e,h)=>{tip.innerHTML=h;tip.style.opacity=1;tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';},hideTip=()=>tip.style.opacity=0;
const RAMP=[[0,[215,48,39]],[0.5,[254,224,139]],[1,[26,152,80]]];
function ramp(v){if(v==null||isNaN(v))return '#ddd';let t=Math.max(0,Math.min(1,v/100));for(let i=1;i<RAMP.length;i++){if(t<=RAMP[i][0]){const[a,c0]=RAMP[i-1],[b,c1]=RAMP[i],f=(t-a)/((b-a)||1);return`rgb(${c0.map((x,k)=>Math.round(x+(c1[k]-x)*f)).join(',')})`;}}return`rgb(${RAMP[RAMP.length-1][1].join(',')})`;}
(function(){const svg=document.getElementById('heat');const W=1000,padL=150,padR=10,padT=10,padB=24,nb=60;
 const rowH=26,H=padT+HEAT.length*rowH+padB;const cw=(W-padL-padR)/nb;
 HEAT.forEach((w,r)=>{const y=padT+r*rowH;const s=w.ev;
   for(let b=0;b<nb;b++){const a=Math.floor(b/nb*s.length),c=Math.max(a+1,Math.floor((b+1)/nb*s.length));const seg=s.slice(a,c).filter(v=>v!=null);
     const m=seg.length?seg.reduce((x,y)=>x+y,0)/seg.length:null;const rc=el('rect',{x:padL+b*cw,y:y,width:cw+0.5,height:rowH-3,fill:ramp(m)});
     rc.addEventListener('mousemove',e=>showTip(e,`${w.name} · ~${Math.round(b/nb*100)}% · evenness ${m==null?'n/a':m.toFixed(0)}`));rc.addEventListener('mouseleave',hideTip);svg.appendChild(rc);}
   const t=el('text',{x:padL-8,y:y+rowH/2,'text-anchor':'end',class:'ax'});t.textContent=w.name;svg.appendChild(t);});
 const t=el('text',{x:padL,y:H-8,class:'ax'});t.textContent='distance along walk →';svg.appendChild(t);
})();
</script></body></html>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('walk_maps', nargs='?', default='GPS/walk_maps',
                    help='folder with the per-step geojsons (default GPS/walk_maps)')
    ap.add_argument('--out', default=None, help='output folder for galleries (default: <walk_maps>/galleries)')
    a = ap.parse_args()
    src = a.walk_maps
    out = a.out or os.path.join(src, 'galleries')
    os.makedirs(out, exist_ok=True)

    stems = sorted({os.path.basename(p).rsplit('_', 1)[0] if p.endswith(('_evenness.geojson','_hardness.geojson'))
                    else os.path.basename(p).replace('_pred_3class.geojson','').replace('_tagged.geojson','').replace('_surface.geojson','')
                    for suff in ['_evenness','_hardness','_tagged','_surface','_pred_3class']
                    for p in glob.glob(os.path.join(src, f'*{suff}.geojson'))})
    # robust stem detection
    stems = set()
    for suff, key in [('_evenness.geojson',-1),('_hardness.geojson',-1),('_tagged.geojson',0),
                      ('_surface.geojson',0),('_pred_3class.geojson',0)]:
        for p in glob.glob(os.path.join(src, f'*{suff}')):
            stems.add(os.path.basename(p)[:-len(suff)])
    stems = sorted(stems)
    if not stems:
        raise SystemExit(f"No *_evenness/_surface/_tagged/... geojsons found in {os.path.abspath(src)!r}. "
                         "Run predict_walks.py and build_all_walks.py first.")

    made = []; heat = []
    for stem in stems:
        steps = assemble_walk(src, stem)
        if not steps:
            print(f"[skip] {stem}: no usable layers"); continue
        page = (PAGE.replace('__TITLE__', html.escape(stem))
                    .replace('__NSTEPS__', str(len(steps)))
                    .replace('__STEPS__', json.dumps(steps))
                    .replace('__CHARTS__', CHARTS_JS))
        fp = os.path.join(out, f"{stem}_gallery.html")
        open(fp, 'w', encoding='utf-8').write(page)
        made.append((stem, len(steps)))
        heat.append({'name': stem, 'ev': [s.get('ev') for s in steps]})
        print(f"[ok] {stem}: {len(steps)} steps -> {os.path.basename(fp)}")

    items = ''.join(f'<li><a href="{html.escape(s)}_gallery.html"><span>{html.escape(s)}</span>'
                    f'<span class="n">{n} steps</span></a></li>' for s, n in made)
    heat_h = 10 + max(1, len(heat)) * 26 + 24
    open(os.path.join(out, 'index.html'), 'w', encoding='utf-8').write(
        INDEX.replace('__COUNT__', str(len(made)))
             .replace('__ITEMS__', items)
             .replace('__HEAT__', json.dumps(heat))
             .replace('__HEATH__', str(heat_h)))
    print(f"\nDone -> {len(made)} galleries in '{out}'. Open index.html.")

if __name__ == '__main__':
    main()
