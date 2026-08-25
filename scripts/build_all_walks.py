"""
build_all_walks.py — batch-build an individual surface-comparison map + outputs for EVERY
walk, with NO manual pasting/copying. Uses the THREE-layer GIS stack (sidewalk + land cover
+ open space, the v7 precedence) for the GIS line, your insole tags for the ground-truth line,
and — if predict_walks.py has been run — model-prediction lines too.

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
<style>html,body{margin:0;height:100%;font-family:-apple-system,Segoe UI,Roboto,sans-serif}#map{height:100%}
.panel{position:absolute;top:12px;right:12px;z-index:1000;background:#fff;padding:10px 12px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,.15);font-size:12.5px;max-width:250px}.panel h3{margin:0 0 6px;font-size:14px}
.row{display:flex;align-items:center;gap:7px;margin:3px 0}.sw{width:15px;height:6px;border-radius:3px;display:inline-block}.hd{font-weight:600;margin-top:6px}
#agree{position:absolute;top:12px;left:12px;z-index:1000;background:#111;color:#fff;padding:7px 13px;border-radius:16px;font-size:12.5px;box-shadow:0 2px 8px rgba(0,0,0,.3)}</style></head><body>
<div id="agree">__AGREE__</div><div id="map"></div>
<div class="panel"><h3>__NAME__</h3><div id="layers" style="font-size:11.5px;color:#555;margin-bottom:6px"></div>
<div class="hd">surface</div>
<div class="row"><span class="sw" style="background:#607D8B"></span>paved / hard</div>
<div class="row"><span class="sw" style="background:#4CAF50"></span>grass / soft</div>
<div class="row"><span class="sw" style="background:#8D6E63"></span>loose</div>
<div class="row"><span class="sw" style="background:#BDBDBD"></span>unknown</div></div>
<script>
const LAYERS=__LAYERS__;
const COL={paved:'#607D8B',grass:'#4CAF50',loose:'#8D6E63',unknown:'#BDBDBD',hard:'#607D8B',soft:'#4CAF50',dirt:'#8D6E63'};
const col=s=>COL[(s||'unknown').toLowerCase()]||COL.unknown;
const map=L.map('map');
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',{attribution:'&copy; OSM, &copy; CARTO',subdomains:'abcd',maxZoom:20}).addTo(map);
document.getElementById('layers').innerHTML=LAYERS.map((L,i)=>`line ${i+1}: <b>${L.name}</b>`).join('<br>');
const N=LAYERS.length,GAP=7.0;let layer=L.layerGroup().addTo(map);const base=LAYERS[0].fc.features;
function off(a,b,slot){const pa=map.latLngToLayerPoint([a[1],a[0]]),pb=map.latLngToLayerPoint([b[1],b[0]]);let dx=pb.x-pa.x,dy=pb.y-pa.y;const L2=Math.hypot(dx,dy)||1;dx/=L2;dy/=L2;const d=(slot-(N-1)/2)*GAP;return[map.layerPointToLatLng(L.point(pa.x-dy*d,pa.y+dx*d)),map.layerPointToLatLng(L.point(pb.x-dy*d,pb.y+dx*d))];}
function render(){layer.clearLayers();for(let i=0;i<base.length;i++){const a=base[i].geometry.coordinates[0],b=base[i].geometry.coordinates[1];LAYERS.forEach((Ly,slot)=>{const f=Ly.fc.features[i];const v=f?f.properties[Ly.key]:'unknown';L.polyline(off(a,b,slot),{color:col(v),weight:2.4,opacity:.95}).addTo(layer).bindPopup(Ly.name+': '+(v||'unknown'));});}}
let bd=null;base.forEach(f=>{const c=f.geometry.coordinates;const bb=L.latLngBounds([[c[0][1],c[0][0]],[c[1][1],c[1][0]]]);bd=bd?bd.extend(bb):bb;});
map.fitBounds(bd,{padding:[30,30]});render();map.on('zoomend moveend',render);
const f0=base[0].geometry.coordinates[0],fl=base[base.length-1].geometry.coordinates[1];
L.circleMarker([f0[1],f0[0]],{radius:6,color:'#2E7D32',fillColor:'#2E7D32',fillOpacity:1}).addTo(map).bindPopup('start');
L.circleMarker([fl[1],fl[0]],{radius:6,color:'#C62828',fillColor:'#C62828',fillOpacity:1}).addTo(map).bindPopup('end');
</script></body></html>"""
def build_html(name, layers, agree):
    payload=[{'name':nm,'key':key,'fc':fc} for (nm,fc,key) in layers]
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
    csvs={p:csv_time_range(p) for p in glob.glob(os.path.join(a.walks_dir,'*.csv')) if is_insole_csv(p)}
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
        maplayers=[('truth',tagged_fc,'tag'),('GIS (3-layer)',gis_fc,'surface')]
        for tag,label in [('pred_3class','model 3-class'),('pred_2class','model 2-class')]:
            pth=os.path.join(a.out,f"{stem}_{tag}.geojson")
            if os.path.exists(pth):
                try: maplayers.append((label,json.load(open(pth)),'surface'))
                except Exception: pass
        CL={'paved','grass','loose'}; n=agn=0
        for i in range(len(pts)-1):
            t,gg=(tags[i] or 'unknown'),gis[i]
            if t in CL and gg in CL: n+=1; agn+=(t==gg)
        agree=f"agreement: {100*agn/n:.0f}%  ({agn}/{n})" if n else "agreement: n/a"
        open(os.path.join(a.out,f"{stem}.html"),'w').write(build_html(f"{name} ({stem})",maplayers,agree))
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
