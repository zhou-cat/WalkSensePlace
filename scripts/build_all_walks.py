"""
build_all_walks.py - batch-build an individual surface-comparison map + outputs for EVERY
walk, with NO manual pasting/copying. Uses the THREE-layer GIS stack (sidewalk + land cover
+ open space, the v7 precedence) for the GIS line, your insole tags for the ground-truth line,
and - if predict_walks.py has been run - model-prediction lines too.

Put in one folder: all walk GPX + their insole REC_*.csv, plus the GIS files
(sidewalk_inventory.geojson, open_space.geojson, and the land-cover tile LCLU_*.shp).

    pip install geopandas pyogrio gpxpy pandas numpy      # NOT fiona
    python build_all_walks.py "C:/Users/Sam/WalkSensePlace/GPS"

For each walk it writes into <out> (default <walks_dir>/walk_maps):
    <gpx_stem>.html            self-contained map, all layers embedded (nothing to paste)
    <gpx_stem>_tagged.geojson  ground truth (insole tags on GPS coords)
    <gpx_stem>_surface.geojson combined 3-layer GIS
plus summary.csv (one row per walk: profiles, source mix, agreement %).
"""
import sys, os, json, glob, argparse, collections
import numpy as np, pandas as pd, gpxpy
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree

METRIC = "EPSG:26986"
SW_TOL, LC_TOL, OS_TOL = 15.0, 1.5, 8.0
SURFACE_MAP = {'grass':'grass','dirt':'loose','compact ground':'loose','compacted ground':'loose',
               'concrete':'paved','asphalt':'paved','exposed aggregate':'paved','brick':'paved'}
EXCLUDE_TAGS = {'transition','wait'}
SENSOR_COLS = set(['sole_id','timestamp','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z',
                   'magn_x','magn_y','magn_z','corrupt'] + [f'pressure_{i:02d}' for i in range(1,13)])
SW_CODES = {'cc':'paved','bc':'paved','gb':'paved','br':'paved','cb':'paved','bl':'paved','ot':'unknown'}
SW_FIELDS = ['MATERIAL','material','Material','surface','SURFACE']
LC_FIELDS = ['COVERNAME','COVERTYPE','COVER_DESC','LC_DESC','LANDCOVER','LCLU','LU_DESC','USEGENNAME']
COLORS = {'paved':'#607D8B','grass':'#4CAF50','loose':'#8D6E63','unknown':'#BDBDBD'}

def sw_class(v):
    if v is None: return 'unknown'
    s=str(v).strip().lower()
    if s in SW_CODES: return SW_CODES[s]
    if any(k in s for k in ['concret','asphalt','brick','granit','paver','stone','slate','cement','bitum']): return 'paved'
    return 'unknown'
def lc_class(v):
    s=str(v).strip().lower()
    if not s or s in ('none','nan','null'): return 'unknown'
    if any(k in s for k in ['open space','grass','pasture','cultivat','agric','recreation','golf','lawn','cemeter']): return 'grass'
    if any(k in s for k in ['impervious','pavement','paved','road','building','commercial','industrial','transportation','parking','residential']): return 'paved'
    if any(k in s for k in ['bare','soil','sand','barren','quarr','excavat']): return 'loose'
    return 'unknown'

# ---------- GIS layers ----------
def read_layer(path):
    if not path or not os.path.exists(path): print(f"  (layer '{path}' not found - skipping)"); return None
    gdf = gpd.read_file(path)
    if 'shp_link' in gdf.columns and not any(c in gdf.columns for c in LC_FIELDS):
        here=os.path.dirname(os.path.abspath(path)) or '.'
        for cand in glob.glob(os.path.join(here,'**','*.shp'),recursive=True):
            try: cols=list(gpd.read_file(cand,rows=1).columns)
            except Exception: continue
            if any(c in cols for c in LC_FIELDS): print(f"  (land cover from {cand})"); gdf=gpd.read_file(cand); break
    if gdf.crs is None: gdf=gdf.set_crs("EPSG:4326")
    return gdf.to_crs(METRIC)

def read_landcover(path, walks_dir):
    # accept a .shp, a folder, or fall back to any LCLU_*.shp in walks_dir
    cands=[]
    if path and os.path.isfile(path): cands=[path]
    elif path and os.path.isdir(path): cands=glob.glob(os.path.join(path,'**','LCLU_*.shp'),recursive=True) or glob.glob(os.path.join(path,'**','*.shp'),recursive=True)
    if not cands: cands=glob.glob(os.path.join(walks_dir,'**','LCLU_*.shp'),recursive=True)
    frames=[]
    for c in cands:
        try:
            g=gpd.read_file(c)
            if any(f in g.columns for f in LC_FIELDS): frames.append(g if g.crs is None else g.to_crs(METRIC) if g.crs else g)
        except Exception: pass
    if not frames: return None
    out=pd.concat(frames,ignore_index=True); out=gpd.GeoDataFrame(out,geometry='geometry')
    if out.crs is None: out=out.set_crs(METRIC)
    return out.to_crs(METRIC)

def build_tree(gdf, cls):
    geoms,c=[],[]
    if gdf is not None:
        for g,k in zip(gdf.geometry.values, cls):
            if g is not None and not g.is_empty: geoms.append(g); c.append(k)
    return (STRtree(geoms) if geoms else None), geoms, c

def classify_points(pts_lonlat, layers):
    (swt,swg,swc),(lct,lcg,lcc),(ost,osg,osc)=layers
    P=gpd.GeoSeries([Point(xy) for xy in pts_lonlat], crs="EPSG:4326").to_crs(METRIC)
    out=[]; src=[]
    for pt in P:
        cls,s='unknown','none'
        if swt is not None:
            i=swt.nearest(pt)
            if swg[i].distance(pt)<=SW_TOL and swc[i]!='unknown': cls,s=swc[i],'sidewalk'
        if cls=='unknown' and lct is not None:
            i=lct.nearest(pt)
            if lcg[i].distance(pt)<=LC_TOL and lcc[i] in ('paved','grass','loose'): cls,s=lcc[i],'landcover'
        if cls=='unknown' and ost is not None:
            i=ost.nearest(pt)
            if osg[i].distance(pt)<=OS_TOL: cls,s='grass','open_space'
        out.append(cls); src.append(s)
    return out, src

# ---------- GPX / CSV ----------
def parse_gpx(path):
    g=gpxpy.parse(open(path)); pts=[]
    for tr in g.tracks:
        for seg in tr.segments:
            for p in seg.points:
                if p.time is None: continue
                pts.append({'lon':p.longitude,'lat':p.latitude,'tms':int(p.time.timestamp()*1000)})
    name=(g.tracks[0].name if g.tracks and g.tracks[0].name else os.path.splitext(os.path.basename(path))[0])
    return pts,name
def csv_time_range(path):
    try:
        ts=pd.to_numeric(pd.read_csv(path,usecols=['timestamp'],low_memory=False)['timestamp'],errors='coerce').dropna()
        return (int(ts.min()),int(ts.max())) if len(ts) else None
    except Exception: return None
def is_insole_csv(path):
    try:
        cols=pd.read_csv(path,nrows=0).columns
        return 'timestamp' in cols and any(k in cols for k in SURFACE_MAP)
    except Exception: return False
def tagged_from_csv(path, pts):
    d=pd.read_csv(path,low_memory=False).sort_values('timestamp').reset_index(drop=True); ev=[]
    for col in d.columns:
        if col in SENSOR_COLS: continue
        key=col.strip().lower()
        if key in EXCLUDE_TAGS: tag='__EXC__'
        elif key in SURFACE_MAP: tag=SURFACE_MAP[key]
        else: continue
        for idx in d.index[d[col].astype(str).str.strip()=='x']: ev.append((int(d.loc[idx,'timestamp']),tag))
    ev=sorted(set(ev)); ets=np.array([e[0] for e in ev]); elab=[e[1] for e in ev]
    def at(t):
        if not len(ets): return None
        i=np.searchsorted(ets,t,side='right')-1
        return None if i<0 or elab[i]=='__EXC__' else elab[i]
    return [at(p['tms']) for p in pts]
def seg_fc(pts, vals, key):
    return {'type':'FeatureCollection','features':[
        {'type':'Feature','properties':{key:(vals[i] or 'unknown')},
         'geometry':{'type':'LineString','coordinates':[[pts[i]['lon'],pts[i]['lat']],[pts[i+1]['lon'],pts[i+1]['lat']]]}}
        for i in range(len(pts)-1)]}

# ---------- HTML (N parallel lines) ----------
HTML=r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>__NAME__</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body{margin:0;height:100%;font-family:-apple-system,Segoe UI,Roboto,sans-serif}#map{height:74vh}#timeline{height:26vh;overflow-y:auto;overflow-x:hidden;background:#fafafa;border-top:1px solid #ddd}
.panel{position:absolute;top:12px;right:12px;z-index:1000;background:#fff;padding:10px 12px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.15);font-size:12.5px;max-width:250px}.panel h3{margin:0 0 6px;font-size:14px}
.row{display:flex;align-items:center;gap:7px;margin:3px 0}.sw{width:15px;height:6px;border-radius:3px;display:inline-block}.hd{font-weight:600;margin-top:6px}
#agree{position:absolute;top:12px;left:12px;z-index:1000;background:#111;color:#fff;padding:7px 13px;border-radius:16px;font-size:12.5px;box-shadow:0 2px 8px rgba(0,0,0,.3)}</style></head><body>
<div id="agree">__AGREE__</div><div id="map"></div>
<div id="timeline"></div>
<div class="panel"><h3>__NAME__</h3><div id="layers" style="font-size:11.5px;color:#555;margin-bottom:6px"></div>
<div class="hd">surface</div>
<div class="row"><span class="sw" style="background:#607D8B"></span>paved / hard</div>
<div class="row"><span class="sw" style="background:#4CAF50"></span>grass / soft</div>
<div class="row"><span class="sw" style="background:#8D6E63"></span>loose</div>
<div class="row"><span class="sw" style="background:#BDBDBD"></span>unknown</div>
<div id="gradients" style="margin-top:8px"></div></div>
<script>
const LAYERS=__LAYERS__;
const COL={paved:'#607D8B',grass:'#4CAF50',loose:'#8D6E63',unknown:'#BDBDBD',hard:'#607D8B',soft:'#4CAF50',dirt:'#8D6E63'};
const catcol=s=>COL[(s||'unknown').toLowerCase()]||COL.unknown;
function _hx(c){return [parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)];}
function _mix(a,b,t){const A=_hx(a),B=_hx(b);return 'rgb('+A.map((x,i)=>Math.round(x+(B[i]-x)*t)).join(',')+')';}
function contcol(v,ramp){if(v==null||isNaN(v))return '#DDD';let t=Math.max(0,Math.min(1,v/100));
  for(let i=1;i<ramp.length;i++){if(t<=ramp[i][0]){const[t0,c0]=ramp[i-1],[t1,c1]=ramp[i];return _mix(c0,c1,(t-t0)/((t1-t0)||1));}}return ramp[ramp.length-1][1];}
function colorOf(Ly,v){return Ly.mode==='cont'?contcol(v,Ly.ramp):catcol(v);}
const map=L.map('map',{renderer:L.svg({padding:0.6}),preferCanvas:false});
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'&copy; OpenStreetMap contributors',maxZoom:19}).addTo(map);
document.getElementById('layers').innerHTML=LAYERS.map((L,i)=>`line ${i+1}: <b>${L.name}</b>`+(L.mode==='cont'?' <span style="color:#888">(gradient 0-100)</span>':'')).join('<br>');
const _ends={evenness:['rough','smooth'],hardness:['soft','hard']};
document.getElementById('gradients').innerHTML=LAYERS.filter(L=>L.mode==='cont').map(L=>{const stops=L.ramp.map(s=>s[1]).join(',');const e=_ends[L.name]||['low','high'];return `<div class="hd">${L.name}</div>`+`<div style="height:10px;border-radius:3px;background:linear-gradient(to right,${stops})"></div>`+`<div style="display:flex;justify-content:space-between;font-size:10px;color:#777"><span>0 ${e[0]}</span><span>100 ${e[1]}</span></div>`;}).join('');
const N=LAYERS.length, WT=2.5, GAP_M=2.5;   // centred bundle: small offsets stay on the path
const base=LAYERS[0].fc.features;
const SMOOTH_W=6;   // moving-average half-window on the centreline (0 = raw GPS)
let RPTS0=base.map(f=>f.geometry.coordinates[0]); RPTS0.push(base[base.length-1].geometry.coordinates[1]);
function smoothPath(pts,w){const n=pts.length,out=[];for(let i=0;i<n;i++){let sx=0,sy=0,k=0;for(let j=Math.max(0,i-w);j<=Math.min(n-1,i+w);j++){sx+=pts[j][0];sy+=pts[j][1];k++;}out.push([sx/k,sy/k]);}return out;}
const RPTS=SMOOTH_W>0?smoothPath(RPTS0,SMOOTH_W):RPTS0;
const CLAT=RPTS.reduce((a,c)=>a+c[1],0)/RPTS.length;
const MLAT=111320.0, MLON=111320.0*Math.cos(CLAT*Math.PI/180);
// Parallel offset computed ONCE in geographic space (metres->deg). Native lat/lng polylines
// then scale correctly on zoom: constant pixel thickness, shared vertices (no gaps/overlap), no redraw.
function offsetLatLng(slot){
  const d=(slot-(N-1)/2)*GAP_M; const X=RPTS.map(c=>[c[0]*MLON,c[1]*MLAT]);   // centred on the path
  const sn=[];                                   // per-segment unit normal (null if zero-length)
  for(let i=0;i<X.length-1;i++){let dx=X[i+1][0]-X[i][0],dy=X[i+1][1]-X[i][1],ln=Math.hypot(dx,dy);sn.push(ln<1e-6?null:[-dy/ln,dx/ln]);}
  const out=[]; let last=null;
  for(let i=0;i<X.length;i++){
    const a=(i>0)?sn[i-1]:null, b=(i<sn.length)?sn[i]:null;
    let nx=0,ny=0,k=0;
    if(a){nx+=a[0];ny+=a[1];k++;} if(b){nx+=b[0];ny+=b[1];k++;}
    if(k===0){ if(last){nx=last[0];ny=last[1];} }          // stationary point: keep previous offset (no collapse)
    else { let ln=Math.hypot(nx,ny);
      if(ln<0.35){ const t=b||a; nx=t[0]; ny=t[1]; }        // near-hairpin: use a segment normal (no spike)
      else { nx/=ln; ny/=ln; const t=b||a; let cs=nx*t[0]+ny*t[1]; if(cs<0.5)cs=0.5; const sc=Math.min(1/cs,1.7); nx*=sc; ny*=sc; } }
    last=[nx,ny];
    out.push([ RPTS[i][1]+(ny*d)/MLAT, RPTS[i][0]+(nx*d)/MLON ]);
  }
  return out;
}
const layer=L.layerGroup().addTo(map); const OFFPATHS=[];
LAYERS.forEach((Ly,slot)=>{const op=offsetLatLng(slot); OFFPATHS[slot]=op;
  for(let i=0;i<base.length;i++){
    // skip folded slivers: where the offset segment reverses vs the path (inside of a U-turn)
    const cdx=RPTS[i+1][0]-RPTS[i][0], cdy=RPTS[i+1][1]-RPTS[i][1];      // path dir (lon,lat)
    const odx=op[i+1][1]-op[i][1], ody=op[i+1][0]-op[i][0];              // offset dir (lon,lat); op is [lat,lon]
    if(cdx*odx+cdy*ody < 0) continue;                                    // reversed -> don't draw (avoids self-overlap)
    const f=Ly.fc.features[i];const v=f?f.properties[Ly.key]:null;const c=colorOf(Ly,v);
    const disp=Ly.mode==='cont'?(v==null?'n/a':Math.round(v)+'/100'):(v||'unknown');
    L.polyline([op[i],op[i+1]],{color:c,weight:WT,opacity:.95,lineCap:'round',lineJoin:'round'}).addTo(layer).bindPopup(Ly.name+': '+disp);
  }});
let bd=null;RPTS.forEach(c=>{const p=L.latLng(c[1],c[0]);bd=bd?bd.extend(p):L.latLngBounds(p,p);});
map.fitBounds(bd,{padding:[30,30]});
const f0=[RPTS[0][0],RPTS[0][1]],fl=[RPTS[RPTS.length-1][0],RPTS[RPTS.length-1][1]];
L.circleMarker([f0[1],f0[0]],{radius:6,color:'#2E7D32',fillColor:'#2E7D32',fillOpacity:1}).addTo(map).bindPopup('start');
L.circleMarker([fl[1],fl[0]],{radius:6,color:'#C62828',fillColor:'#C62828',fillOpacity:1}).addTo(map).bindPopup('end');

// ---- step timeline: one band per line, one block per step; hover links to the map ----
let hlm=L.circleMarker([RPTS[0][1],RPTS[0][0]],{radius:8,color:'#111',weight:2.5,fillColor:'#ffeb3b',fillOpacity:.95,pane:'markerPane'}).bindTooltip('',{direction:'top',offset:[0,-8]});
function highlight(r,i){
  if(r<0||i<0||i>=base.length){ if(map.hasLayer(hlm)) map.removeLayer(hlm); return; }
  const p=OFFPATHS[r]; const a=p[i], b=p[i+1]||p[i];
  hlm.setLatLng([(a[0]+b[0])/2,(a[1]+b[1])/2]);
  const f=LAYERS[r].fc.features[i]; const v=f?f.properties[LAYERS[r].key]:null;
  hlm.setStyle({fillColor:colorOf(LAYERS[r],v)});
  if(!map.hasLayer(hlm)) hlm.addTo(map);
  const disp=LAYERS[r].mode==='cont'?(v==null?'n/a':Math.round(v)+'/100'):(v||'unknown');
  hlm.setTooltipContent(LAYERS[r].name+' step '+i+': '+disp);
}
function buildTimeline(){
  const div=document.getElementById('timeline'); const n=base.length;
  const padL=22, labelW=124, padR=30, rowH=17, gap=7, top=30, axisH=20;
  const plotW=Math.max(280, div.clientWidth - padL - labelW - padR);
  const px=plotW/n, x0=padL+labelW, yb=top+LAYERS.length*(rowH+gap), W=div.clientWidth, H=yb+axisH;
  const P=[`<text x="${padL}" y="19" font-size="13" font-weight="700" fill="#222">Step timeline</text>`,
           `<text x="${padL+150}" y="19" font-size="11" fill="#999">(hover a line to locate it on the map)</text>`];
  for(let i=0;i<=n;i+=50){const gx=x0+i*px;
    P.push(`<line x1="${gx}" y1="${top-3}" x2="${gx}" y2="${yb}" stroke="#e6e6e6" stroke-width="1"/>`);
    P.push(`<text x="${gx}" y="${yb+14}" font-size="10" fill="#aaa" text-anchor="middle">${i}</text>`);}
  P.push(`<text x="${x0+plotW}" y="${yb+14}" font-size="10" fill="#aaa" text-anchor="end">step</text>`);
  LAYERS.forEach((Ly,r)=>{const y=top+r*(rowH+gap);
    P.push(`<rect x="${x0}" y="${y}" width="${plotW}" height="${rowH}" fill="#eee" rx="3"/>`);
    P.push(`<text x="${padL+labelW-8}" y="${y+rowH/2+4}" font-size="11" font-weight="600" fill="#333" text-anchor="end">${Ly.name}</text>`);
    for(let i=0;i<n;i++){const f=Ly.fc.features[i];const v=f?f.properties[Ly.key]:null;
      P.push(`<rect x="${x0+i*px}" y="${y}" width="${px+0.6}" height="${rowH}" fill="${colorOf(Ly,v)}"></rect>`);}
    P.push(`<rect x="${x0}" y="${y}" width="${plotW}" height="${rowH}" fill="none" stroke="#dcdcdc" rx="3"/>`);
  });
  P.push(`<line id="tlc" x1="0" x2="0" y1="${top-3}" y2="${yb}" stroke="#111" stroke-width="1.3" style="display:none;pointer-events:none"/>`);
  P.push(`<rect id="tlhit" x="${x0}" y="${top-3}" width="${plotW}" height="${yb-top+5}" fill="transparent" style="cursor:crosshair"/>`);
  div.innerHTML=`<svg width="${W}" height="${H}" style="font-family:inherit;display:block">${P.join('')}</svg>`;
  const svg=div.querySelector('svg'), cur=svg.querySelector('#tlc'), hit=svg.querySelector('#tlhit');
  hit.addEventListener('mousemove',e=>{const rr=svg.getBoundingClientRect();
    const mx=e.clientX-rr.left, my=e.clientY-rr.top;
    let i=Math.floor((mx-x0)/px), r=Math.floor((my-top)/(rowH+gap));
    const inRow=(r>=0 && r<LAYERS.length && (my-top-r*(rowH+gap))<=rowH);
    if(i<0||i>=n||!inRow){highlight(-1,-1);cur.style.display='none';return;}
    const gx=x0+(i+0.5)*px; cur.setAttribute('x1',gx); cur.setAttribute('x2',gx); cur.style.display=''; highlight(r,i);});
  hit.addEventListener('mouseleave',()=>{highlight(-1,-1);cur.style.display='none';});
}
buildTimeline(); window.addEventListener('resize',buildTimeline);
</script></body></html>"""
def build_html(name, layers, agree):
    payload=[{'name':nm,'key':key,'fc':fc,'mode':mode,'ramp':ramp} for (nm,fc,key,mode,ramp) in layers]
    return HTML.replace('__NAME__',name).replace('__AGREE__',agree).replace('__LAYERS__',json.dumps(payload))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('walks_dir',nargs='?',default='.')
    ap.add_argument('--sidewalk',default='sidewalk_inventory.geojson')
    ap.add_argument('--openspace',default='open_space.geojson')
    ap.add_argument('--landcover',default=None,help='land-cover tile .shp (or folder of LCLU_*.shp)')
    ap.add_argument('--out',default='walk_maps')
    a=ap.parse_args()
    def _res(p):
        if p and os.path.exists(p): return p
        c=os.path.join(a.walks_dir,os.path.basename(p)) if p else None
        return c if c and os.path.exists(c) else p
    a.sidewalk=_res(a.sidewalk); a.openspace=_res(a.openspace)
    if a.out=='walk_maps': a.out=os.path.join(a.walks_dir,'walk_maps')
    os.makedirs(a.out,exist_ok=True)
    print(f"walks dir: {a.walks_dir}\nout: {a.out}\n")

    print("Loading GIS layers (once)...")
    sw=read_layer(a.sidewalk); osp=read_layer(a.openspace); lc=read_landcover(a.landcover,a.walks_dir)
    sw_field=next((c for c in SW_FIELDS if sw is not None and c in sw.columns),None)
    lc_field=next((c for c in LC_FIELDS if lc is not None and c in lc.columns),None)
    layers=(build_tree(sw,[sw_class(v) for v in sw[sw_field]] if sw_field else []),
            build_tree(lc,[lc_class(v) for v in lc[lc_field]] if lc_field else []),
            build_tree(osp,['grass']*len(osp) if osp is not None else []))
    print(f"  sidewalk={len(layers[0][1])}, land cover={len(layers[1][1])}, open space={len(layers[2][1])}\n")

    gpx_files=sorted(glob.glob(os.path.join(a.walks_dir,'*.gpx')))
    _all_csv=[p for p in glob.glob(os.path.join(a.walks_dir,'*.csv')) if is_insole_csv(p)]
    _gps=[p for p in _all_csv if '_GPS' in os.path.basename(p)]
    csvs={p:csv_time_range(p) for p in (_gps or _all_csv)}   # prefer REC_*_GPS.csv for ground-truth tags
    print(f"{len(gpx_files)} GPX, {len(csvs)} insole CSV(s)\n")
    rows=[]
    for gpx in gpx_files:
        pts,name=parse_gpx(gpx); stem=os.path.splitext(os.path.basename(gpx))[0]
        if len(pts)<2: print(f"[skip] {stem}: <2 pts"); continue
        g0,g1=pts[0]['tms'],pts[-1]['tms']; best,ov=None,0
        for p,r in csvs.items():
            if not r: continue
            o=max(0,min(g1,r[1])-max(g0,r[0]))
            if o>ov: ov,best=o,p
        tags=tagged_from_csv(best,pts) if best else [None]*len(pts)
        gis,gsrc=classify_points([(p['lon'],p['lat']) for p in pts],layers)
        tagged_fc=seg_fc(pts,tags,'tag'); gis_fc=seg_fc(pts,gis,'surface')
        json.dump(tagged_fc,open(os.path.join(a.out,f"{stem}_tagged.geojson"),'w'))
        json.dump(gis_fc,open(os.path.join(a.out,f"{stem}_surface.geojson"),'w'))
        RAMP_EVEN=[[0,'#d73027'],[0.5,'#fee08b'],[1,'#1a9850']]   # rough(red) -> smooth(green)
        RAMP_HARD=[[0,'#3b4cc0'],[0.5,'#e0e0e0'],[1,'#b40426']]   # soft(blue) -> hard(red)  [coolwarm]
        maplayers=[('truth',tagged_fc,'tag','cat',None),('GIS (3-layer)',gis_fc,'surface','cat',None)]
        def _load(tag):
            pth=os.path.join(a.out,f"{stem}_{tag}.geojson")
            if os.path.exists(pth):
                try: return json.load(open(pth))
                except Exception: return None
            return None
        for tag,label in [('pred_3class','model 3-class'),('pred_2class','model 2-class')]:
            fc=_load(tag)
            if fc: maplayers.append((label,fc,'surface','cat',None))
        ev=_load('evenness'); hd=_load('hardness')
        if ev: maplayers.append(('evenness',ev,'value','cont',RAMP_EVEN))
        if hd: maplayers.append(('hardness',hd,'value','cont',RAMP_HARD))
        CL={'paved','grass','loose'}; n=agn=0
        for i in range(len(pts)-1):
            t,gg=(tags[i] or 'unknown'),gis[i]
            if t in CL and gg in CL: n+=1; agn+=(t==gg)
        agree=f"agreement: {100*agn/n:.0f}%  ({agn}/{n})" if n else "agreement: n/a"
        open(os.path.join(a.out,f"{stem}.html"),'w',encoding='utf-8').write(build_html(f"{name} ({stem})",maplayers,agree))
        gc=collections.Counter(gis); sc=collections.Counter(gsrc); tc=collections.Counter(t or 'unknown' for t in tags)
        rows.append({'walk':name,'file':os.path.basename(gpx),'points':len(pts),
                     'paired_csv':(os.path.basename(best) if best else 'NONE'),
                     'truth_paved':tc.get('paved',0),'truth_grass':tc.get('grass',0),'truth_loose':tc.get('loose',0),
                     'gis_paved':gc.get('paved',0),'gis_grass':gc.get('grass',0),'gis_loose':gc.get('loose',0),'gis_unknown':gc.get('unknown',0),
                     'src_sidewalk':sc.get('sidewalk',0),'src_landcover':sc.get('landcover',0),'src_openspace':sc.get('open_space',0),'src_none':sc.get('none',0),
                     'agreement_pct':(round(100*agn/n,1) if n else None),'agree_n':n})
        print(f"[ok] {stem:26s} pts={len(pts):4d} csv={rows[-1]['paired_csv']:26s} {agree}  src={dict(sc)}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out,'summary.csv'),index=False)
    print(f"\nDone -> '{a.out}/' (one .html + geojsons per walk, plus summary.csv)")

if __name__=='__main__':
    main()
