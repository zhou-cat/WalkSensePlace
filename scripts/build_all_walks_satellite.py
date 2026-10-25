"""
build_all_walks_satellite.py - same per-walk maps as build_all_walks.py, but built for
reading lines against aerial imagery.

What is different from build_all_walks.py:

  * every line is individually toggleable: ground truth, GIS, 3-class model, evenness,
    hardness, and a plain line (the bare GPS track with no classification).
  * the default basemap is SATELLITE: Stadia Alidade Satellite, Esri World Imagery,
    USGS Imagery, USGS Imagery Topo.
  * a BAREBONES group is also offered for reading the GIS surface classification without
    imagery underneath it: Stadia Stamen Toner Lite, Esri World Gray Canvas, CARTO
    Positron. Switching to one of these swaps the palette back to the muted one.
  * the GIS SURFACE CLASSIFICATION AS A MAP: the sidewalk, land-cover, open-space and
    OpenStreetMap path polygons around each walk, coloured by surface class, stacked in
    the same precedence the GIS line uses (open space under land cover under sidewalks,
    with OSM paths on top by default).
  * OPENSTREETMAP PATH SURFACES as a 4th GIS layer (--osm export.geojson, an Overpass
    export of highway ways). This is the only layer that sees loose ground on paths -
    the stone-dust and gravel paths through parks that overhead land cover misses under
    the trees. It feeds the GIS line too, so the line and the map always agree. It is offered two
    ways: as basemaps in their own right - "GIS surface + CARTO Positron" and the other
    barebones combinations - and as an overlay you can draw over imagery.
  * on imagery the lines use a bolder palette, a dark casing underneath each segment, and
    an adjustable white scrim over the tiles. All three are needed: saturated colour alone
    still disappears over bright concrete, and a scrim alone washes the imagery out.
  * every line is also listed in the layer control (bottom left), kept in step with the
    panel checkboxes.

The GIS/GPX/tagging logic is imported from build_all_walks.py rather than copied, so the
two scripts cannot drift apart. Keep them in the same folder.

    pip install geopandas pyogrio gpxpy pandas numpy      # NOT fiona
    python build_all_walks_satellite.py "C:/Users/Sam/WalkSensePlace/GPS"
    python build_all_walks_satellite.py GPS --gis-map           # open on the GIS surface map
    python build_all_walks_satellite.py GPS --lines truth,plain --gis-overlay
    python build_all_walks_satellite.py GPS --osm export.geojson --osm-compacted paved
"""
import argparse
import collections
import glob
import json
import os
import sys

import pandas as pd

# Import the data pipeline from build_all_walks.py so the GIS/GPX/tagging logic has one
# home. Look next to THIS file, not just the working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import build_all_walks as base
except ModuleNotFoundError as e:
    if e.name == 'build_all_walks':
        sys.exit('build_all_walks.py not found - keep it in the same folder as this '
                 'script (the GIS and GPX functions are imported from it)')
    sys.exit(f"build_all_walks.py needs '{e.name}', which is not installed:\n"
             '    pip install geopandas pyogrio gpxpy pandas numpy      # NOT fiona')
except ImportError as e:
    sys.exit(f'could not import build_all_walks.py: {e}')

# Which lines exist, in the order they are drawn. 'plain' is new: the bare GPS track with
# no classification, useful as a reference against the imagery and when a walk has no
# model output at all.
# Stadia Maps key for the Stadia basemaps (Alidade Satellite, Stamen Toner Lite).
# The STADIA_KEY environment variable overrides it.
STADIA_KEY = os.environ.get('STADIA_KEY', '377f7e02-33d0-4f45-bd6e-c273fe50c30e')

LINE_ROLES = ['truth', 'gis', 'model3', 'evenness', 'hardness', 'plain']
DEFAULT_ON = ['truth', 'gis', 'model3']

# Colour-vision-safe palette, matching scripts/figstyle.py so the maps and the paper's
# charts agree. These are Okabe-Ito colours, chosen by exhaustive search for the largest
# worst-case separation under normal vision and all three common deficiencies; the
# previous green/brown/blue-grey set separated grass from loose almost entirely along a
# hue axis that red-green colour-blind readers cannot see.
#   paved  sky blue   #56B4E9      grass  black   #000000
#   loose  vermillion #D55E00      unknown purple #CC79A7
# The continuous ramps are blue-to-orange, the standard CVD-safe diverging pair, and both
# ends differ in lightness so they survive greyscale printing.
RAMP_EVEN = [[0, '#D55E00'], [0.5, '#F2F2F2'], [1, '#0072B2']]   # rough -> smooth
RAMP_HARD = [[0, '#0072B2'], [0.5, '#F2F2F2'], [1, '#D55E00']]   # soft  -> hard

HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>__NAME__</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body{margin:0;height:100%;font-family:-apple-system,Segoe UI,Roboto,sans-serif}
#map{height:74vh}#timeline{height:26vh;overflow-y:auto;overflow-x:hidden;background:#fafafa;border-top:1px solid #ddd}
.panel{position:absolute;top:10px;right:10px;z-index:1000;background:rgba(255,255,255,.96);padding:7px 9px;border-radius:9px;box-shadow:0 2px 10px rgba(0,0,0,.15);font-size:11px;max-width:215px;max-height:calc(74vh - 40px);overflow-y:auto}
.panel h3{margin:0 0 5px;font-size:12.5px}.row{display:flex;align-items:center;gap:7px;margin:3px 0}
.sw{width:15px;height:6px;border-radius:3px;display:inline-block}.hd{font-weight:600;margin-top:6px}
#agree{margin-top:10px;background:#111;color:#fff;padding:7px 10px;border-radius:9px;font-size:12px}
#agree b{color:#ffd54f}
.ctl{margin-top:10px;font-size:12.5px;border-top:1px solid #eee;padding-top:8px}
.ctl label{display:block;margin:4px 0;cursor:pointer}
.ctl input[type=range]{width:100%;margin:2px 0}
.note{color:#888;font-size:11px}
#cm{margin-top:10px;border-top:1px solid #eee;padding-top:8px;font-size:11px}
#cm table{border-collapse:collapse;margin:3px 0 8px}#cm td,#cm th{border:1px solid #e0e0e0;padding:1px 3px;text-align:center;min-width:20px;font-size:10px}
#cm caption{font-weight:600;font-size:11.5px;text-align:left;margin-bottom:2px}
#nolines{display:none;margin-top:8px;padding:6px 8px;border-radius:7px;background:#fff3cd;color:#664d03;font-size:11px}</style></head><body>
<div id="map"></div><div id="timeline"></div>
<div class="panel"><h3>__NAME__</h3>
<div id="layers" style="font-size:11.5px;color:#555;margin-bottom:6px"></div>
<div class="hd">surface</div>
<div class="row"><span class="sw" id="k_paved"></span>paved / hard</div>
<div class="row"><span class="sw" id="k_grass"></span>grass / soft</div>
<div class="row"><span class="sw" id="k_loose"></span>loose</div>
<div class="row"><span class="sw" id="k_unknown"></span>unknown</div>
<div id="gradients" style="margin-top:8px"></div>
<div id="agree">__AGREE__</div>
<div class="ctl"><b>lines</b>
  <label><input type="checkbox" class="lt" data-r="truth"> ground truth</label>
  <label><input type="checkbox" class="lt" data-r="gis"> GIS (along the track)</label>
  <label><input type="checkbox" class="lt" data-r="model3"> 3-class model</label>
  <label><input type="checkbox" class="lt" data-r="evenness"> evenness</label>
  <label><input type="checkbox" class="lt" data-r="hardness"> hardness</label>
  <label><input type="checkbox" class="lt" data-r="plain"> plain track</label>
  <div id="nolines">No line data was embedded for this walk.</div>
</div>
<div class="ctl"><b>GIS source priority</b> <span class="note">(top wins)</span>
  <div id="prio"></div>
  <div class="note">Used for the GIS line and to resolve overlaps on the surface map. Set
  it when building, with --gis-priority.</div>
</div>
<div class="ctl"><b>GIS surface map</b>
  <label><input type="checkbox" id="gistoggle"> draw over current basemap</label>
  <label>fill <span id="gisval">__GIS_OP__%</span>
    <input type="range" id="gisop" min="10" max="100" step="5" value="__GIS_OP__"></label>
  <div class="note" style="font-weight:600">__GIS_VIEW__</div>
  <div class="note" id="gisnote">Or pick a "GIS surface + ..." entry in the basemap list
  (bottom left) to use the classification itself as the map.</div>
</div>
<div class="ctl"><b>imagery</b>
  <label>white scrim <span id="scrimval">__SCRIM__%</span>
    <input type="range" id="scrim" min="0" max="90" step="5" value="__SCRIM__"></label>
  <label><input type="checkbox" id="bold" checked> bold lines + dark casing</label>
  <label><input type="checkbox" id="truthfx"> ground truth: black outline + shadow</label>
  <label><input type="checkbox" id="rawgps"> raw GPS fixes (unsmoothed)</label>
  <div class="note">Scrim and casing only apply over imagery; barebones and GIS-surface
  basemaps use the muted palette.</div>
</div>
<div id="cm"></div></div>
<script>
const ALL=__LAYERS__, LINES_ON=__LINES_ON__, ROLES=__ROLES__;
const GISMAP=__GISMAP__;
const ORDER=__GIS_ORDER__, GIS_AVAIL=new Set(__GIS_AVAIL__);
const SRC_LABEL={sidewalk:'sidewalk inventory',osm:'OpenStreetMap paths',landcover:'land cover',open_space:'open space'};
const HAS_GIS=!!(GISMAP&&GISMAP.features&&GISMAP.features.length);

// colour-vision-safe, matching scripts/figstyle.py (see the note by RAMP_EVEN)
const COL_MUTED={paved:'#56B4E9',grass:'#009E73',loose:'#D55E00',unknown:'#CC79A7',hard:'#0072B2',soft:'#E69F00',dirt:'#D55E00',plain:'#111111'};
const COL_BOLD ={paved:'#56B4E9',grass:'#009E73',loose:'#D55E00',unknown:'#CC79A7',hard:'#0072B2',soft:'#E69F00',dirt:'#D55E00',plain:'#111111'};
const RAMP_BOLD={evenness:[[0,'#FF1744'],[0.5,'#FFEA00'],[1,'#00E676']],
                 hardness:[[0,'#2979FF'],[0.5,'#FFFFFF'],[1,'#FF1744']]};
let SAT=true, BOLD=true, SCRIM=__SCRIM__/100, GIS_OP=__GIS_OP__/100;
const palette=()=>(SAT&&BOLD)?COL_BOLD:COL_MUTED;
// The GIS surface polygons use the SAME palette as the lines, so a reader matches a
// polygon to a line by colour without a second key. That is possible because grass is
// bluish green rather than black: a black fill would render a whole park as a dark mass.
// The polygons are drawn at --gis-opacity under the lines, so they stay distinguishable
// by lightness rather than by hue.
const COL_GIS=COL_MUTED;   // polygons use the SAME palette as the lines
const catcol=s=>palette()[(s||'unknown').toLowerCase()]||palette().unknown;
function _hx(c){return [parseInt(c.slice(1,3),16),parseInt(c.slice(3,5),16),parseInt(c.slice(5,7),16)];}
function _mix(a,b,t){const A=_hx(a),B=_hx(b);return 'rgb('+A.map((x,i)=>Math.round(x+(B[i]-x)*t)).join(',')+')';}
function rampFor(Ly){return (SAT&&BOLD&&RAMP_BOLD[Ly.name])?RAMP_BOLD[Ly.name]:Ly.ramp;}
function contcol(v,ramp){if(v==null||isNaN(v))return '#DDD';let t=Math.max(0,Math.min(1,v/100));
  for(let i=1;i<ramp.length;i++){if(t<=ramp[i][0]){const[t0,c0]=ramp[i-1],[t1,c1]=ramp[i];return _mix(c0,c1,(t-t0)/((t1-t0)||1));}}return ramp[ramp.length-1][1];}
function colorOf(Ly,v){return Ly.mode==='cont'?contcol(v,rampFor(Ly)):(Ly.mode==='plain'?palette().plain:catcol(v));}

const SHOW={}; ROLES.forEach(r=>SHOW[r]=LINES_ON.indexOf(r)>=0);
const byRole=r=>ALL.find(Ly=>Ly.role===r);
function activeLayers(){return ROLES.filter(r=>SHOW[r]).map(r=>byRole(r)).filter(Boolean);}

// ---- map + panes. Stacking, bottom to top:
//   tilePane 200   basemap tiles, then the white scrim as a GridLayer above them
//   gisPane  350   GIS surface polygons
//   overlayPane 400  the track lines (Leaflet's default pane, default renderer)
const map=L.map('map',{renderer:L.svg({padding:0.6}),preferCanvas:false});   // as build_all_walks.py
map.createPane('gisPane'); map.getPane('gisPane').style.zIndex=350;
// Ground truth gets its own pane, above the other lines (overlayPane is 400). That does
// two things: the reference line is never hidden under a prediction, and a CSS
// drop-shadow can be applied to the pane DIV - a real offset shadow, filtered as one
// image so overlapping segments don't double it, and supported in every browser
// (unlike CSS filters on individual SVG paths).
map.createPane('truthPane'); map.getPane('truthPane').style.zIndex=450;
const TRUTH_R=L.svg({pane:'truthPane',padding:0.6});
let TRUTH_FX=__TRUTH_FX__;
const TRUTH_SHADOW='drop-shadow(1.5px 2.5px 2px rgba(0,0,0,0.55))';
function applyTruthFx(){map.getPane('truthPane').style.filter=TRUTH_FX?TRUTH_SHADOW:'none';}

const ESRI='Tiles &copy; Esri', USGS='Tiles courtesy of the U.S. Geological Survey';
const esri=(svc,nz)=>L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/'+svc+'/MapServer/tile/{z}/{y}/{x}',{attribution:ESRI,maxZoom:19,maxNativeZoom:nz||19});
const usgs=svc=>L.tileLayer('https://basemap.nationalmap.gov/arcgis/rest/services/'+svc+'/MapServer/tile/{z}/{y}/{x}',{attribution:USGS,maxZoom:19,maxNativeZoom:16});
const STADIA_KEY="__STADIA_KEY__";
const stadia=(style,ext)=>L.tileLayer(`https://tiles.stadiamaps.com/tiles/${style}/{z}/{x}/{y}.${ext||'png'}`+(STADIA_KEY?`?api_key=${STADIA_KEY}`:''),
  {attribution:'&copy; Stadia Maps &copy; Stamen &copy; OpenMapTiles &copy; OpenStreetMap',maxZoom:20});
const carto=()=>L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  {subdomains:'abcd',attribution:'&copy; OpenStreetMap contributors &copy; CARTO',maxZoom:20});

// factories, so every basemap entry - including the GIS combinations - owns its tiles
const SAT_F={}, BARE_F={};
if(STADIA_KEY) SAT_F['Stadia Alidade Satellite']=()=>stadia('alidade_satellite','jpg');
SAT_F['Esri World Imagery']=()=>esri('World_Imagery',19);
SAT_F['USGS Imagery']=()=>usgs('USGSImageryOnly');
SAT_F['USGS Imagery Topo']=()=>usgs('USGSImageryTopo');
if(STADIA_KEY) BARE_F['Stadia Stamen Toner Lite']=()=>stadia('stamen_toner_lite');
BARE_F['Esri World Gray Canvas']=()=>esri('Canvas/World_Light_Gray_Base',16);
BARE_F['CARTO Positron']=carto;

// ---- GIS surface polygons ----
const GIS_RENDER=L.canvas({pane:'gisPane',padding:0.5});
function gisStyle(f){const c=COL_GIS[(f.properties.surface||'unknown')]||COL_GIS.unknown;
  return {fillColor:c,fillOpacity:GIS_OP,color:c,weight:0.6,opacity:Math.min(1,GIS_OP+0.25)};}
function makeGis(){return L.geoJSON(GISMAP,{pane:'gisPane',renderer:GIS_RENDER,style:gisStyle,
  onEachFeature:(f,ly)=>ly.bindPopup(`GIS: <b>${f.properties.surface}</b><br><span style="color:#777">from ${f.properties.src==='osm'?'OpenStreetMap':f.properties.src}`+(f.properties.tag?` (surface=${f.properties.tag})`:'')+`</span>`)});}
const GIS_LAYERS=[];                     // every GIS instance, so the fill slider reaches all
function gisInstance(){const g=makeGis(); GIS_LAYERS.push(g); return g;}

const BASEMAPS={}; const SAT_NAMES=Object.keys(SAT_F);
Object.keys(SAT_F).forEach(n=>BASEMAPS[n]=SAT_F[n]());
Object.keys(BARE_F).forEach(n=>BASEMAPS[n]=BARE_F[n]());
// the GIS classification AS the map: polygons layered on each barebones basemap
if(HAS_GIS) Object.keys(BARE_F).forEach(n=>BASEMAPS['GIS surface + '+n]=L.layerGroup([BARE_F[n](),gisInstance()]));
const GIS_OV=HAS_GIS?gisInstance():null;     // the same polygons as an overlay, for imagery

let START='__BASEMAP__';
if(!(START in BASEMAPS)){console.warn('basemap "'+START+'" unavailable for this walk; using the first one');START=Object.keys(BASEMAPS)[0];}
BASEMAPS[START].addTo(map);
SAT=SAT_NAMES.indexOf(START)>=0;

// ---- white scrim: a GridLayer of plain white tiles, ABOVE the basemap tiles (zIndex) and
// below every vector pane. Being a tile layer it covers exactly what is loaded, pans and
// zooms with the map, and needs no vector renderer.
const Scrim=L.GridLayer.extend({createTile(){const d=document.createElement('div');d.style.background='#ffffff';return d;}});
const scrim=new Scrim({zIndex:50,opacity:0,updateWhenIdle:false}).addTo(map);
function applyScrim(){scrim.setOpacity(SAT?SCRIM:0);}

// ---- one LayerGroup per line, so each appears in the layer control too ----
const LABEL={truth:'ground truth',gis:'GIS (along the track)',model3:'3-class model',evenness:'evenness',hardness:'hardness',plain:'plain track'};
const GROUPS={}; ROLES.forEach(r=>GROUPS[r]=L.layerGroup());
const OVERLAYS={};
ROLES.forEach(r=>{if(byRole(r)) OVERLAYS['line: '+LABEL[r]]=GROUPS[r];});
if(GIS_OV) OVERLAYS['GIS surface polygons']=GIS_OV;
L.control.layers(BASEMAPS,OVERLAYS,{position:'bottomleft',collapsed:true}).addTo(map);

// ---- geometry (reference path = first layer that actually has features) ----
const REF=[byRole('truth'),byRole('plain'),...ALL].find(Ly=>Ly&&Ly.fc&&Ly.fc.features&&Ly.fc.features.length);
if(!REF){document.getElementById('nolines').style.display='block';throw new Error('no line features embedded');}
const base_=REF.fc.features;
// GAP_M: spacing between parallel lines (the outer lines sit this far off the track).
// SMOOTH_W: points averaged either side of each fix; it cuts the inside of bends.
const WT=__WT__, CASE_W=__WT__+3.5, GAP_M=__GAP_M__, SMOOTH_W=__SMOOTH_W__;
let RPTS0=base_.map(f=>f.geometry.coordinates[0]); RPTS0.push(base_[base_.length-1].geometry.coordinates[1]);
function smoothPath(pts,w){const n=pts.length,out=[];for(let i=0;i<n;i++){let sx=0,sy=0,k=0;for(let j=Math.max(0,i-w);j<=Math.min(n-1,i+w);j++){sx+=pts[j][0];sy+=pts[j][1];k++;}out.push([sx/k,sy/k]);}return out;}
const RPTS=SMOOTH_W>0?smoothPath(RPTS0,SMOOTH_W):RPTS0;
const CLAT=RPTS.reduce((a,c)=>a+c[1],0)/RPTS.length;
const MLAT=111320.0, MLON=111320.0*Math.cos(CLAT*Math.PI/180);
function offsetLatLng(slot,N){
  const d=(slot-(N-1)/2)*GAP_M; const X=RPTS.map(c=>[c[0]*MLON,c[1]*MLAT]);
  const sn=[]; for(let i=0;i<X.length-1;i++){let dx=X[i+1][0]-X[i][0],dy=X[i+1][1]-X[i][1],ln=Math.hypot(dx,dy);sn.push(ln<1e-6?null:[-dy/ln,dx/ln]);}
  const out=[]; let last=null;
  for(let i=0;i<X.length;i++){const a=(i>0)?sn[i-1]:null,b=(i<sn.length)?sn[i]:null;let nx=0,ny=0,k=0;
    if(a){nx+=a[0];ny+=a[1];k++;} if(b){nx+=b[0];ny+=b[1];k++;}
    if(k===0){if(last){nx=last[0];ny=last[1];}}
    else{let ln=Math.hypot(nx,ny); if(ln<0.35){const t=b||a;nx=t[0];ny=t[1];}
      else{nx/=ln;ny/=ln;const t=b||a;let cs=nx*t[0]+ny*t[1];if(cs<0.5)cs=0.5;const sc=Math.min(1/cs,1.7);nx*=sc;ny*=sc;}}
    last=[nx,ny]; out.push([RPTS[i][1]+(ny*d)/MLAT, RPTS[i][0]+(nx*d)/MLON]);}
  return out;
}
let OFFPATHS=[], DRAWN=[], SYNC=false;
function drawRoute(){
  OFFPATHS=[]; DRAWN=activeLayers(); const N=DRAWN.length; const casing=(SAT&&BOLD);
  // keep the map's set of line groups equal to SHOW; SYNC stops the layer control's
  // overlayadd/overlayremove events (which also fire for programmatic changes) recursing
  SYNC=true;
  ROLES.forEach(r=>{GROUPS[r].clearLayers(); const want=!!(SHOW[r]&&byRole(r));
    if(want&&!map.hasLayer(GROUPS[r])) GROUPS[r].addTo(map);
    if(!want&&map.hasLayer(GROUPS[r])) map.removeLayer(GROUPS[r]);});
  SYNC=false;
  DRAWN.forEach((Ly,slot)=>{const op=offsetLatLng(slot,N); OFFPATHS[slot]=op; const grp=GROUPS[Ly.role];
    const isT=(Ly.role==='truth'), outl=isT&&TRUTH_FX;
    const where=isT?{pane:'truthPane',renderer:TRUTH_R}:{};
    // segments that face forward along the route (backward-facing offsets are dropped)
    const segs=[];
    for(let i=0;i<base_.length;i++){
      const cdx=RPTS[i+1][0]-RPTS[i][0],cdy=RPTS[i+1][1]-RPTS[i][1];
      const odx=op[i+1][1]-op[i][1],ody=op[i+1][0]-op[i][0];
      if(cdx*odx+cdy*ody>=0) segs.push(i);
    }
    // PASS 1: every outline/casing for this line, THEN pass 2: every coloured core.
    // Drawing outline_i, core_i, outline_i+1, core_i+1 ... lets each new outline land on
    // top of the previous core, which leaves a black ring at every joint - the 'chain of
    // beads' look. Two passes give one continuous outline under one continuous line.
    if(outl||casing) segs.forEach(i=>grp.addLayer(L.polyline([op[i],op[i+1]],Object.assign(
      outl?{color:'#000',weight:WT+3,opacity:.95}:{color:'#000',weight:CASE_W,opacity:.55},
      {lineCap:'round',lineJoin:'round',interactive:false},where))));
    segs.forEach(i=>{
      const f=Ly.fc.features[i];const v=f?f.properties[Ly.key]:null;const c=colorOf(Ly,v);
      const disp=Ly.mode==='cont'?(v==null?'n/a':Math.round(v)+'/100'):(Ly.mode==='plain'?'track':(v||'unknown'));
      // fully opaque: translucent segments with round caps darken where they overlap,
      // which prints as a dot at every joint
      grp.addLayer(L.polyline([op[i],op[i+1]],Object.assign({color:c,weight:WT,opacity:1,
        lineCap:'round',lineJoin:'round'},where)).bindPopup(Ly.name+': '+disp));
    });
  });
  document.getElementById('layers').innerHTML=DRAWN.length?DRAWN.map((Ly,i)=>`line ${i+1}: <b>${Ly.name}</b>`+(Ly.mode==='cont'?' <span style="color:#888">(gradient 0-100)</span>':'')).join('<br>'):'<i>no lines selected</i>';
  ['paved','grass','loose','unknown'].forEach(k=>{const el=document.getElementById('k_'+k);if(el)el.style.background=palette()[k];});
}
let bd=null;RPTS.forEach(c=>{const p=L.latLng(c[1],c[0]);bd=bd?bd.extend(p):L.latLngBounds(p,p);});
map.fitBounds(bd,{padding:[30,30],maxZoom:19});
// raw GPS fixes, unsmoothed and un-offset: where the phone actually reported each point
const RAW=L.layerGroup();
const RAWPTS=__RAWPTS__||RPTS0;
RAWPTS.forEach((c,i)=>RAW.addLayer(L.circleMarker([c[1],c[0]],{radius:2.2,color:'#000',weight:1,
  fillColor:'#fff',fillOpacity:1,pane:'markerPane'}).bindTooltip('GPS fix '+i,{direction:'top'})));
if(__RAW_ON__) RAW.addTo(map);
L.circleMarker([RPTS[0][1],RPTS[0][0]],{radius:6,color:'#fff',weight:2,fillColor:'#2E7D32',fillOpacity:1}).addTo(map).bindPopup('start');
L.circleMarker([RPTS[RPTS.length-1][1],RPTS[RPTS.length-1][0]],{radius:6,color:'#fff',weight:2,fillColor:'#C62828',fillOpacity:1}).addTo(map).bindPopup('end');

const _ends={evenness:['rough','smooth'],hardness:['soft','hard']};
function drawGradients(){document.getElementById('gradients').innerHTML=activeLayers().filter(Ly=>Ly.mode==='cont').map(Ly=>{const stops=rampFor(Ly).map(s=>s[1]).join(',');const e=_ends[Ly.name]||['low','high'];return `<div class="hd">${Ly.name}</div><div style="height:10px;border-radius:3px;background:linear-gradient(to right,${stops})"></div><div style="display:flex;justify-content:space-between;font-size:10px;color:#777"><span>0 ${e[0]}</span><span>100 ${e[1]}</span></div>`;}).join('');}

const CL3=['grass','loose','paved'];
function confusion(predLayer){
  const t=byRole('truth'); if(!t||!predLayer) return null;
  const idx=Object.fromEntries(CL3.map((c,i)=>[c,i]));
  const M=CL3.map(()=>CL3.map(()=>0)); let n=0,agree=0;
  for(let i=0;i<base_.length;i++){
    const tv=t.fc.features[i]?t.fc.features[i].properties[t.key]:null;
    const pv=predLayer.fc.features[i]?predLayer.fc.features[i].properties[predLayer.key]:null;
    if(tv in idx && pv in idx){M[idx[tv]][idx[pv]]++; n++; if(tv===pv)agree++;}
  }
  return {classes:CL3,M,n,agree};
}
function renderMatrix(title,cm){
  if(!cm||!cm.n) return `<table><caption>${title}</caption><tr><td>n/a</td></tr></table>`;
  const cls=cm.classes; let h=`<table><caption>${title} &mdash; ${(100*cm.agree/cm.n).toFixed(0)}% (${cm.agree}/${cm.n})</caption>`;
  h+=`<tr><th></th><th colspan="${cls.length}">predicted</th></tr><tr><th>truth&darr;</th>`+cls.map(c=>`<th>${c}</th>`).join('')+`</tr>`;
  for(let r=0;r<cls.length;r++){h+=`<tr><th>${cls[r]}</th>`+cm.M[r].map((x,c)=>{const on=r===c;return `<td style="background:${on?'#d7ecd9':(x?'#fdecea':'#fff')};font-weight:${on?600:400}">${x}</td>`;}).join('')+`</tr>`;}
  return h+`</table>`;
}
function updateInfo(){
  const gisCM=confusion(byRole('gis')), modCM=confusion(byRole('model3'));
  const f=cm=>cm&&cm.n?`${(100*cm.agree/cm.n).toFixed(0)}% (${cm.agree}/${cm.n})`:'n/a';
  document.getElementById('agree').innerHTML=`GIS vs truth: <b>${f(gisCM)}</b><br>3-class model vs truth: <b>${f(modCM)}</b>`;
  document.getElementById('cm').innerHTML=renderMatrix('GIS vs truth',gisCM)+renderMatrix('3-class model vs truth',modCM);
}

let hlm=L.circleMarker([RPTS[0][1],RPTS[0][0]],{radius:8,color:'#111',weight:2.5,fillColor:'#ffeb3b',fillOpacity:.95,pane:'markerPane'}).bindTooltip('',{direction:'top',offset:[0,-8]});
function highlight(r,i){
  if(r<0||i<0||i>=base_.length){if(map.hasLayer(hlm))map.removeLayer(hlm);return;}
  const p=OFFPATHS[r];if(!p)return;const a=p[i],b=p[i+1]||p[i];
  hlm.setLatLng([(a[0]+b[0])/2,(a[1]+b[1])/2]);
  const f=DRAWN[r].fc.features[i];const v=f?f.properties[DRAWN[r].key]:null;
  hlm.setStyle({fillColor:colorOf(DRAWN[r],v)}); if(!map.hasLayer(hlm))hlm.addTo(map);
  const disp=DRAWN[r].mode==='cont'?(v==null?'n/a':Math.round(v)+'/100'):(v||'track');
  hlm.setTooltipContent(DRAWN[r].name+' step '+i+': '+disp);
}
function buildTimeline(){
  const div=document.getElementById('timeline');const n=base_.length;const L2=DRAWN;
  if(!L2.length){div.innerHTML='<div style="padding:12px;color:#888;font-size:12px">no lines selected</div>';return;}
  const padL=22,labelW=124,padR=30,rowH=17,gap=7,top=30,axisH=20;
  const plotW=Math.max(280,div.clientWidth-padL-labelW-padR);
  const px=plotW/n,x0=padL+labelW,yb=top+L2.length*(rowH+gap),W=div.clientWidth,H=yb+axisH;
  const P=[`<text x="${padL}" y="19" font-size="13" font-weight="700" fill="#222">Step timeline</text>`,
           `<text x="${padL+150}" y="19" font-size="11" fill="#999">(hover a line to locate it on the map)</text>`];
  for(let i=0;i<=n;i+=50){const gx=x0+i*px;P.push(`<line x1="${gx}" y1="${top-3}" x2="${gx}" y2="${yb}" stroke="#e6e6e6" stroke-width="1"/>`);P.push(`<text x="${gx}" y="${yb+14}" font-size="10" fill="#aaa" text-anchor="middle">${i}</text>`);}
  L2.forEach((Ly,r)=>{const y=top+r*(rowH+gap);
    P.push(`<rect x="${x0}" y="${y}" width="${plotW}" height="${rowH}" fill="#eee" rx="3"/>`);
    P.push(`<text x="${padL+labelW-8}" y="${y+rowH/2+4}" font-size="11" font-weight="600" fill="#333" text-anchor="end">${Ly.name}</text>`);
    for(let i=0;i<n;i++){const f=Ly.fc.features[i];const v=f?f.properties[Ly.key]:null;P.push(`<rect x="${x0+i*px}" y="${y}" width="${px+0.6}" height="${rowH}" fill="${colorOf(Ly,v)}"></rect>`);}
    P.push(`<rect x="${x0}" y="${y}" width="${plotW}" height="${rowH}" fill="none" stroke="#dcdcdc" rx="3"/>`);});
  P.push(`<line id="tlc" x1="0" x2="0" y1="${top-3}" y2="${yb}" stroke="#111" stroke-width="1.3" style="display:none;pointer-events:none"/>`);
  P.push(`<rect id="tlhit" x="${x0}" y="${top-3}" width="${plotW}" height="${yb-top+5}" fill="transparent" style="cursor:crosshair"/>`);
  div.innerHTML=`<svg width="${W}" height="${H}" style="font-family:inherit;display:block">${P.join('')}</svg>`;
  const svg=div.querySelector('svg'),cur=svg.querySelector('#tlc'),hit=svg.querySelector('#tlhit');
  hit.addEventListener('mousemove',e=>{const rr=svg.getBoundingClientRect();const mx=e.clientX-rr.left,my=e.clientY-rr.top;
    let i=Math.floor((mx-x0)/px),r=Math.floor((my-top)/(rowH+gap));
    const inRow=(r>=0&&r<L2.length&&(my-top-r*(rowH+gap))<=rowH);
    if(i<0||i>=n||!inRow){highlight(-1,-1);cur.style.display='none';return;}
    const gx=x0+(i+0.5)*px;cur.setAttribute('x1',gx);cur.setAttribute('x2',gx);cur.style.display='';highlight(r,i);});
  hit.addEventListener('mouseleave',()=>{highlight(-1,-1);cur.style.display='none';});
}
function refreshAll(){drawRoute();drawGradients();buildTimeline();updateInfo();}

// ---- controls ----
document.querySelectorAll('.lt').forEach(cb=>{
  const r=cb.dataset.r;
  if(!byRole(r)){cb.disabled=true;cb.parentElement.style.opacity=.4;cb.parentElement.title='layer not available for this walk';SHOW[r]=false;}
  cb.checked=!!SHOW[r];
  cb.addEventListener('change',e=>{SHOW[r]=e.target.checked;refreshAll();});
});
// the layer control can toggle lines too; keep it and the checkboxes in step
map.on('overlayadd overlayremove',e=>{
  if(SYNC) return;
  const on=(e.type==='overlayadd');
  if(GIS_OV&&e.layer===GIS_OV){document.getElementById('gistoggle').checked=on;return;}
  const r=Object.keys(GROUPS).find(k=>GROUPS[k]===e.layer); if(!r) return;
  SHOW[r]=on; const cb=document.querySelector('.lt[data-r="'+r+'"]'); if(cb) cb.checked=on;
  refreshAll();
});
map.on('baselayerchange',e=>{SAT=SAT_NAMES.indexOf(e.name)>=0; applyScrim(); refreshAll();});

const gt=document.getElementById('gistoggle');
if(!GIS_OV){gt.disabled=true;gt.parentElement.style.opacity=.4;
  document.getElementById('gisnote').textContent='No GIS polygons were embedded for this walk (built with --no-gis-map, or none near the track).';}
gt.checked=__GIS_OVERLAY_ON__&&!!GIS_OV;
if(gt.checked) GIS_OV.addTo(map);
gt.addEventListener('change',e=>{if(!GIS_OV)return; if(e.target.checked)GIS_OV.addTo(map); else map.removeLayer(GIS_OV);});
document.getElementById('gisop').addEventListener('input',e=>{GIS_OP=(+e.target.value)/100;
  document.getElementById('gisval').textContent=e.target.value+'%'; GIS_LAYERS.forEach(g=>g.setStyle(gisStyle));});
document.getElementById('bold').addEventListener('change',e=>{BOLD=e.target.checked;refreshAll();});
const rg=document.getElementById('rawgps'); rg.checked=__RAW_ON__;
rg.addEventListener('change',e=>{if(e.target.checked)RAW.addTo(map); else map.removeLayer(RAW);});
const tfx=document.getElementById('truthfx'); tfx.checked=TRUTH_FX;
tfx.addEventListener('change',e=>{TRUTH_FX=e.target.checked;applyTruthFx();refreshAll();});
applyTruthFx();
document.getElementById('scrim').addEventListener('input',e=>{
  SCRIM=(+e.target.value)/100;document.getElementById('scrimval').textContent=e.target.value+'%';applyScrim();});
document.getElementById('prio').innerHTML=ORDER.map((k,i)=>`<div style="font-size:11.5px;margin:1px 0;color:${GIS_AVAIL.has(k)?'#333':'#aaa'}">${i+1}. ${SRC_LABEL[k]||k}${GIS_AVAIL.has(k)?'':' (not loaded)'}</div>`).join('');
applyScrim(); refreshAll(); window.addEventListener('resize',buildTimeline);
</script></body></html>"""


# ---------- GIS surface map: the classification as polygons ----------
# ---------- OpenStreetMap path surfaces (4th GIS layer) ----------
# surface=* values -> the three classes. 'compacted' is deliberately NOT in the table: in
# Boston parks it is mostly stone dust, and whether that counts as loose depends on how
# the walkers tagged it, so it is a command-line choice (--osm-compacted).
OSM_SURFACE = {
    'loose': {'gravel', 'fine_gravel', 'dirt', 'earth', 'ground', 'sand', 'pebblestone',
              'unpaved', 'rock', 'mud', 'woodchips', 'dirt/sand', 'grit', 'shells'},
    'grass': {'grass', 'grass_paver'},
    'paved': {'asphalt', 'concrete', 'paved', 'paving_stones', 'brick', 'bricks', 'sett',
              'cobblestone', 'unhewn_cobblestone', 'wood', 'metal', 'metal_grid',
              'chipseal', 'tartan'},
}


def osm_class(v, compacted='loose'):
    """surface=* -> paved/grass/loose/unknown. Untagged and unrecognised -> unknown, never
    guessed: a footway with no surface tag must not claim a point."""
    t = str(v or '').strip().lower()
    if not t or t in ('none', 'nan', 'null'):
        return 'unknown'
    if t == 'compacted':
        return compacted
    for k, vals in OSM_SURFACE.items():
        if t in vals:
            return k
    # tolerate subtypes and common typos seen in the export: concrete:plates,
    # paving_stonesc, ...
    if t.startswith(('concrete', 'paving_stones', 'brick', 'asphalt')):
        return 'paved'
    if any(w in t for w in ('gravel', 'dirt', 'sand')):
        return 'loose'
    return 'unknown'


def _width_m(v):
    try:
        return float(str(v).lower().replace('m', '').replace(',', '.').split(';')[0].strip())
    except (TypeError, ValueError):
        return None


def read_osm(path, compacted='loose', buf_default=1.5, buf_min=0.75, buf_max=4.0):
    """OSM highway ways -> {tree, geoms, cls, disp, tag}, all in METRIC.

    Only ways whose surface maps to a class are kept. geoms are the raw lines/areas, used
    for point classification by distance; disp are the same features buffered to their
    tagged width (or buf_default) so they can be drawn as polygons on the surface map.
    """
    import geopandas as gpd
    from shapely.strtree import STRtree
    if not path or not os.path.exists(path):
        print(f"  (OSM layer '{path}' not found - skipping)")
        return None
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs('EPSG:4326')
    g = g.to_crs(base.METRIC)
    surf = g['surface'] if 'surface' in g.columns else [None] * len(g)
    width = g['width'] if 'width' in g.columns else [None] * len(g)
    geoms, cls, disp, tag = [], [], [], []
    for geom, sv, wv in zip(g.geometry.values, surf, width):
        if geom is None or geom.is_empty:
            continue
        k = osm_class(sv, compacted)
        if k == 'unknown':
            continue
        if geom.geom_type in ('LineString', 'MultiLineString'):
            w = _width_m(wv)
            r = min(buf_max, max(buf_min, w / 2)) if w else buf_default
            d = geom.buffer(r, cap_style=2)        # flat ends, so paths don't bulge
        else:
            d = geom
        geoms.append(geom); cls.append(k); disp.append(d); tag.append(str(sv))
    if not geoms:
        print(f'  (OSM layer {path}: no ways with a usable surface tag)')
        return None
    c = collections.Counter(cls)
    print(f'  OSM paths: {len(geoms)} tagged ways of {len(g)} '
          f'(paved {c.get("paved",0)}, grass {c.get("grass",0)}, loose {c.get("loose",0)}; '
          f'compacted -> {compacted})')
    return dict(tree=STRtree(geoms), geoms=geoms, cls=cls, disp=disp, tag=tag)


GIS_SOURCES = ['sidewalk', 'osm', 'landcover', 'open_space']
SOURCE_ALIASES = {
    'sidewalk': 'sidewalk', 'sidewalks': 'sidewalk', 'sw': 'sidewalk',
    'sidewalk_inventory': 'sidewalk', 'inventory': 'sidewalk',
    'osm': 'osm', 'openstreetmap': 'osm', 'paths': 'osm',
    'landcover': 'landcover', 'land_cover': 'landcover', 'land-cover': 'landcover',
    'lc': 'landcover', 'lclu': 'landcover',
    'open_space': 'open_space', 'openspace': 'open_space', 'open-space': 'open_space',
    'os': 'open_space', 'parks': 'open_space',
}
# the four --osm-priority presets, as full orders (highest priority first)
OSM_PRIORITY_ORDERS = {
    'first':           ['osm', 'sidewalk', 'landcover', 'open_space'],
    'after-sidewalk':  ['sidewalk', 'osm', 'landcover', 'open_space'],
    'after-landcover': ['sidewalk', 'landcover', 'osm', 'open_space'],
    'last':            ['sidewalk', 'landcover', 'open_space', 'osm'],
}


def parse_priority(text):
    """'sidewalk,osm,landcover,open_space' -> canonical list, highest priority first.

    Aliases are accepted (sw, lc, os, ...). Sources left out are appended in their
    default relative order, with a note, so a partial list still yields a full order.
    """
    out = []
    for tok in [t.strip().lower() for t in str(text).split(',') if t.strip()]:
        if tok not in SOURCE_ALIASES:
            sys.exit(f"--gis-priority: unknown source '{tok}'. Use any of "
                     f"{GIS_SOURCES} (aliases: sw, osm, lc, os)")
        c = SOURCE_ALIASES[tok]
        if c in out:
            sys.exit(f"--gis-priority: '{c}' is listed twice")
        out.append(c)
    missing = [c for c in OSM_PRIORITY_ORDERS['first'] if c not in out]
    if missing:
        print(f'  [note] --gis-priority did not list {missing}; appending them last')
        out += missing
    return out


# How far from a feature a GPS fix may be and still take its surface.
#
# The inherited values (sidewalk 15 m, open space 8 m, OSM 5 m, land cover 1.5 m) were far
# larger than the GPS error for most sources, and unequal. A 15 m sidewalk reach labels a
# fix 10 m into a lawn 'paved', and because each source reached a different distance, the
# furthest-reaching one won points before the priority order was even consulted.
#
# Default: ONE reach for the polygon sources, sized to the GPS error (median ~3 m from the
# nearest mapped path across our walks). OSM distances are to a path's CENTRELINE, so it
# gets half a path width on top. Land cover is wall-to-wall - a fix is almost always
# inside some polygon, at distance 0 - so its reach only matters at polygon edges.
LEGACY_REACH = {'sidewalk': 15.0, 'open_space': 8.0, 'osm': 5.0, 'landcover': 1.5}


def make_reach(reach=3.0, path_halfwidth=1.0, landcover=1.5, **over):
    r = {'sidewalk': reach, 'open_space': reach, 'osm': reach + path_halfwidth,
         'landcover': landcover}
    r.update({k: v for k, v in over.items() if v is not None})
    return r


def point_nearest(pts_lonlat, layers, osm=None):
    """For each point, each source's NEAREST feature as (class or None, distance in m).

    No reach is applied here, so the same result can be resolved at any reach - which is
    what the reach sweep does. None marks a feature with no usable class ('unknown'
    sidewalk material, a land-cover class that is not a walking surface).
    """
    import geopandas as gpd
    from shapely.geometry import Point
    (swt, swg, swc), (lct, lcg, lcc), (ost, osg, osc) = layers
    P = gpd.GeoSeries([Point(xy) for xy in pts_lonlat], crs='EPSG:4326').to_crs(base.METRIC)
    out = []
    for pt in P:
        c = {}
        if swt is not None:
            i = int(swt.nearest(pt))
            c['sidewalk'] = (swc[i] if swc[i] != 'unknown' else None, swg[i].distance(pt))
        if lct is not None:
            i = int(lct.nearest(pt))
            c['landcover'] = (lcc[i] if lcc[i] in ('paved', 'grass', 'loose') else None,
                              lcg[i].distance(pt))
        if ost is not None:
            i = int(ost.nearest(pt))
            c['open_space'] = ('grass', osg[i].distance(pt))
        if osm:
            i = int(osm['tree'].nearest(pt))
            c['osm'] = (osm['cls'][i], osm['geoms'][i].distance(pt))
        out.append(c)
    return out


def apply_reach(nearest, reach):
    """-> per point {source: class}, keeping only usable classes within each source's reach."""
    return [{k: cl for k, (cl, d) in c.items() if cl is not None and d <= reach[k]}
            for c in nearest]


def point_candidates(pts_lonlat, layers, osm=None, osm_tol=None, reach=None):
    """What EACH source would say about each point, on its own - {source: class}.

    With reach=LEGACY_REACH this reproduces base.classify_points exactly.
    """
    reach = dict(reach or make_reach())
    if osm_tol is not None:
        reach['osm'] = osm_tol
    return apply_reach(point_nearest(pts_lonlat, layers, osm), reach)


SWEEP_REACH = [0, 1, 2, 3, 4, 5, 8, 10, 15]


def reach_sweep(nearest, tags, order, path_halfwidth=1.0, landcover=1.5):
    """GIS agreement with ground truth, and GIS class shares, at each common reach."""
    CL = ('paved', 'grass', 'loose')
    rows = []
    for R in SWEEP_REACH:
        gis, _ = resolve_candidates(apply_reach(nearest, make_reach(R, path_halfwidth, landcover)),
                                    order)
        seg = gis[:-1]                                   # one label per segment, as mapped
        n = agn = 0
        for i, g in enumerate(seg):
            t = tags[i] or 'unknown'
            if t in CL and g in CL:
                n += 1; agn += (t == g)
        cnt = {k: sum(g == k for g in seg) for k in CL + ('unknown',)}
        tot = len(seg) or 1
        rows.append(dict(reach_m=R, n=n, agreement_pct=round(100 * agn / n, 1) if n else None,
                         **{f'gis_{k}_pct': round(100 * v / tot, 1) for k, v in cnt.items()}))
    return rows


def resolve_candidates(cands, order):
    """First source in `order` that has an answer wins. -> (classes, sources)."""
    cls, src = [], []
    for c in cands:
        for k in order:
            if k in c:
                cls.append(c[k]); src.append(k); break
        else:
            cls.append('unknown'); src.append('none')
    return cls, src


def classify_points_osm(pts_lonlat, layers, osm=None, osm_tol=5.0, priority='first',
                        order=None):
    """Kept for compatibility: candidates + resolution under a preset or an explicit
    order. Note for 'after-landcover': land cover is wall-to-wall, so it claims every
    point whose class is known, and OSM only reaches the rest (mostly tree canopy)."""
    order = order or OSM_PRIORITY_ORDERS[priority]
    return resolve_candidates(point_candidates(pts_lonlat, layers, osm, osm_tol=osm_tol,
                                               reach=LEGACY_REACH), order)


# ---------- display-only map matching ----------
MM_FOOT = {'footway', 'path', 'pedestrian', 'cycleway', 'track', 'steps', 'bridleway',
           'living_street', 'service', 'residential'}


def load_match_network(path):
    """Every walkable OSM way in the export - tagged with a surface or not - as lon/lat
    lines. Untagged ways matter here: matching is about WHERE you walked, not surface."""
    if not path or not os.path.exists(path):
        return None
    d = json.load(open(path, encoding='utf-8'))
    out = []
    for f in d.get('features', []):
        g = f.get('geometry') or {}
        if (f.get('properties') or {}).get('highway') not in MM_FOOT:
            continue
        if g.get('type') == 'LineString':
            out.append(g['coordinates'])
        elif g.get('type') == 'MultiLineString':
            out.extend(g['coordinates'])
    return out or None


def map_match(latlon, network, sigma=4.0, beta=1.5, radius=18.0, null_m=12.0, k=6, blend=4):
    """latlon: [(lat, lon), ...] -> (matched [(lat, lon), ...], snapped [bool, ...]).

    sigma  : GPS noise scale (m); emission cost d^2 / (2 sigma^2)
    beta   : step-consistency scale (m); transition cost is the length of the DIFFERENCE
             between the matched step vector and the GPS step vector, / beta. Comparing
             vectors, not lengths, means a matched track must move the way the GPS moved:
             a stretch displaced parallel to a path snaps, a sideways jump between paths is
             expensive, and a stretch heading where no path goes stays raw.
    blend  : where the matched track still jumps (e.g. entering or leaving an unmatched
             stretch), the correction is eased over this many fixes either side
    radius : how far from a fix a path may be and still be a candidate (m)
    null_m : a fix stays unmatched when no candidate is closer than about this (m)
    """
    import math
    if not latlon or not network:
        return list(latlon), [False] * len(latlon)
    lat0, lon0 = latlon[0]
    mx = 111320.0 * math.cos(math.radians(lat0)); my = 111320.0
    to = lambda la, lo: ((lo - lon0) * mx, (la - lat0) * my)
    back = lambda x, y: (y / my + lat0, x / mx + lon0)
    Z = [to(la, lo) for la, lo in latlon]
    xs = [p[0] for p in Z]; ys = [p[1] for p in Z]; pad = radius + 5
    segs = []
    for line in network:
        P_ = [to(la, lo) for lo, la in line]
        for a_, b_ in zip(P_[:-1], P_[1:]):
            if (max(a_[0], b_[0]) < min(xs) - pad or min(a_[0], b_[0]) > max(xs) + pad or
                    max(a_[1], b_[1]) < min(ys) - pad or min(a_[1], b_[1]) > max(ys) + pad):
                continue
            segs.append((a_, b_))

    def proj(p, a_, b_):
        dx, dy = b_[0] - a_[0], b_[1] - a_[1]; L = dx * dx + dy * dy
        t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a_[0]) * dx + (p[1] - a_[1]) * dy) / L))
        q = (a_[0] + t * dx, a_[1] + t * dy)
        return q, math.hypot(p[0] - q[0], p[1] - q[1])

    # candidates per fix: (position, emission cost); index 0 is always the unmatched state
    null_cost = null_m * null_m / (2 * sigma * sigma)
    C = []
    for z in Z:
        cs = sorted((proj(z, a_, b_) for a_, b_ in segs), key=lambda c: c[1])
        keep = []
        for q, d in cs:
            if d > radius or len(keep) >= k:
                break
            if all(math.hypot(q[0] - r[0], q[1] - r[1]) > 1.0 for r, _ in keep):
                keep.append((q, d * d / (2 * sigma * sigma)))
        C.append([(z, null_cost)] + keep)

    SWITCH = 1.0      # small cost for entering/leaving the unmatched state (no flicker)
    cost = [c[1] for c in C[0]]; back_ptr = []
    for t in range(1, len(Z)):
        gx, gy = Z[t][0] - Z[t - 1][0], Z[t][1] - Z[t - 1][1]
        new, bp = [], []
        for j, (qj, ej) in enumerate(C[t]):
            best, arg = float('inf'), 0
            for i, (qi, _) in enumerate(C[t - 1]):
                tr = (math.hypot((qj[0] - qi[0]) - gx, (qj[1] - qi[1]) - gy) / beta
                      + (SWITCH if (i == 0) != (j == 0) else 0.0))
                v = cost[i] + tr
                if v < best:
                    best, arg = v, i
            new.append(best + ej); bp.append(arg)
        cost = new; back_ptr.append(bp)
    j = min(range(len(cost)), key=lambda i: cost[i])
    path = [j]
    for bp in reversed(back_ptr):
        j = bp[j]; path.append(j)
    path.reverse()
    M = [C[t][j][0] for t, j in enumerate(path)]
    # ease any remaining jump: the correction (matched - raw) is averaged over +-blend
    # fixes, but only around jumps, so everywhere else the track stays exactly on its path
    if blend and len(M) > 2:
        corr = [(m[0] - z[0], m[1] - z[1]) for m, z in zip(M, Z)]
        jump = [False] * len(M)
        for t in range(len(M) - 1):
            ms = math.hypot(M[t + 1][0] - M[t][0], M[t + 1][1] - M[t][1])
            gs = math.hypot(Z[t + 1][0] - Z[t][0], Z[t + 1][1] - Z[t][1])
            if ms > max(3.0, 2.5 * gs):
                for u in range(max(0, t - blend), min(len(M), t + blend + 2)):
                    jump[u] = True
        out = list(M)
        for t in range(len(M)):
            if jump[t]:
                w = [(blend + 1 - abs(u - t), u) for u in range(max(0, t - blend), min(len(M), t + blend + 1))]
                W = sum(a for a, _ in w)
                cx = sum(a * corr[u][0] for a, u in w) / W; cy = sum(a * corr[u][1] for a, u in w) / W
                out[t] = (Z[t][0] + cx, Z[t][1] + cy)
        M = out
    matched = [back(*q) for q in M]
    return matched, [j != 0 for j in path]


def gis_precedence(order):
    """Drawing order, bottom -> top: the REVERSE of the priority order, so the
    highest-priority polygon is drawn last and, being opaque, is the one you see.
    The flag is skip_unknown: open space is always 'grass' so it has none to skip."""
    if isinstance(order, str):
        order = OSM_PRIORITY_ORDERS[order]
    return [(k, k != 'open_space') for k in reversed(order)]


def _round_coords(c, nd=6):
    if isinstance(c, (list, tuple)):
        if c and isinstance(c[0], (int, float)):
            return [round(float(v), nd) for v in c]
        return [_round_coords(x, nd) for x in c]
    return c


# Within ONE layer, two classes can overlap (an OSM gravel path crossing a concrete one
# at a junction). There is no between-layer precedence to settle that, so the rarer and
# more informative class wins: a loose patch should not be painted over by the paved
# path it crosses.
OVERLAP_CLASS_ORDER = ['loose', 'grass', 'paved']
MIN_PIECE_M2 = 0.25          # drop slivers smaller than this left over by differencing


def _union(geoms):
    from shapely.ops import unary_union
    return unary_union(list(geoms))


def _valid(g):
    """Repair invalid geometry (land-cover shapefiles often have self-intersections),
    which would otherwise raise a TopologyException in difference()."""
    if g is None or g.is_empty or g.is_valid:
        return g
    try:
        from shapely import make_valid          # shapely >= 2
        return make_valid(g)
    except ImportError:
        try:
            from shapely.validation import make_valid
            return make_valid(g)
        except ImportError:
            return g.buffer(0)


def _polys_only(g, min_area=None):
    """Keep the areal part of a difference result and drop slivers. difference() can
    return GeometryCollections with stray lines/points along shared edges."""
    if min_area is None:
        min_area = MIN_PIECE_M2       # read at call time, so the setting can be changed
    if g is None or g.is_empty:
        return None
    parts = []
    if g.geom_type == 'Polygon':
        parts = [g]
    elif g.geom_type in ('MultiPolygon', 'GeometryCollection'):
        for p in g.geoms:
            if p.geom_type == 'Polygon':
                parts.append(p)
            elif p.geom_type == 'MultiPolygon':
                parts.extend(p.geoms)
    parts = [p for p in parts if p.area >= min_area]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else _union(parts)


def resolve_overlaps(items_by_src, order_bottom_up):
    """items_by_src: {src: [(geom, cls, tag), ...]} in METRIC.
    order_bottom_up: [(src, skip_unknown), ...] as gis_precedence returns it.

    Returns non-overlapping [(geom, cls, src, tag)] in bottom-up draw order. Each layer,
    taken from the TOP of the precedence down, keeps only what no higher layer covers.
    """
    covered = None
    pieces = []
    rank = {c: i for i, c in enumerate(OVERLAP_CLASS_ORDER)}
    for src, _skip in reversed(order_bottom_up):
        items = items_by_src.get(src) or []
        groups = {}
        for g, cls, tag in items:
            groups.setdefault((cls, tag), []).append(g)
        for key in sorted(groups, key=lambda k: (rank.get(k[0], 99), k[1])):
            u = _valid(_union(groups[key]))
            if u is None or u.is_empty:
                continue
            vis = u if covered is None else _valid(u.difference(covered))
            vis = _polys_only(vis)
            if vis is not None:
                pieces.append((vis, key[0], src, key[1]))
            covered = u if covered is None else _valid(_union([covered, u]))
    pieces.reverse()             # bottom-up, so 'stack' and 'resolve' draw the same way
    return pieces


def gis_surface_fc(pts, layers, pad_m=80.0, simplify_m=0.6, max_features=8000,
                   osm=None, osm_priority='first', overlap='resolve', order=None,
                   reach=None):
    # reach=None: FOOTPRINT view - each polygon as published (OSM paths at their width).
    # reach={...}: LABEL view - sidewalk, open-space and OSM features grown by their reach
    #   before overlaps are resolved, so each spot shows the label a GPS fix there gets.
    #   Land cover is not grown: it is wall-to-wall, a fix is almost always inside a
    #   polygon at distance 0, and growing it would let a neighbour of a different class
    #   claim the edge where the label would not. Remaining mismatches are rare edge
    #   cases: the label uses each source's NEAREST feature, the map its class order.
    """GIS surface classification as polygons, clipped to the walk.

    Mirrors classify_points' precedence VISUALLY: open space at the bottom, land cover
    above it (unknown land-cover polygons skipped, exactly as the point classifier skips
    them), sidewalks on top. So the colour a polygon map shows at any spot is the colour
    the GIS line would get there.

    Returns a WGS84 FeatureCollection with properties {surface, src}, or None.
    """
    import geopandas as gpd
    from shapely.geometry import Point, box

    P = gpd.GeoSeries([Point(p['lon'], p['lat']) for p in pts],
                      crs='EPSG:4326').to_crs(base.METRIC)
    minx, miny, maxx, maxy = P.total_bounds
    win = box(minx - pad_m, miny - pad_m, maxx + pad_m, maxy + pad_m)

    by_src = {'sidewalk': layers[0], 'landcover': layers[1], 'open_space': layers[2]}
    if osm:
        by_src['osm'] = (osm['tree'], osm['geoms'], osm['cls'])
    geoms, surf, srcs, tags = [], [], [], []
    items_by_src = {}
    skipped = 0
    for src, skip_unknown in gis_precedence(order or osm_priority):
        if src not in by_src:
            continue
        tree, g_all, c_all = by_src[src]
        if tree is None or not g_all:
            continue
        # OSM is queried on its lines but DRAWN as the buffered path polygons
        g_draw = osm['disp'] if src == 'osm' else g_all
        grow = (reach or {}).get(src, 0.0) if src != 'landcover' else 0.0
        if grow > 0 and src == 'osm':
            g_draw = osm['geoms']                    # grow the centreline, not the width
        # features just outside the window can reach into it once grown
        hits = tree.query(win.buffer(grow) if grow > 0 else win)
        # shapely 2 returns integer indices; shapely 1.8 returned geometries
        if len(hits) and not isinstance(hits[0], (int,)) and not hasattr(hits[0], '__index__'):
            pos = {id(g): k for k, g in enumerate(g_all)}
            hits = [pos[id(g)] for g in hits if id(g) in pos]
        for i in hits:
            i = int(i)
            k = c_all[i] if src != 'open_space' else 'grass'
            if skip_unknown and k not in ('paved', 'grass', 'loose'):
                skipped += 1
                continue
            g = (g_draw[i].buffer(grow) if grow > 0 else g_draw[i]).intersection(win)
            if g.is_empty:
                continue
            if simplify_m > 0:
                g = g.simplify(simplify_m, preserve_topology=True)
            if g.is_empty:
                continue
            t = osm['tag'][i] if src == 'osm' else ''
            geoms.append(g); surf.append(k); srcs.append(src); tags.append(t)
            items_by_src.setdefault(src, []).append((g, k, t))
    if not geoms:
        return None
    if overlap == 'resolve':
        # simplify happened per polygon BEFORE this, so the shared edges come out of the
        # difference itself and neighbouring pieces meet exactly, without gaps
        n_in = len(geoms)
        pieces = resolve_overlaps(items_by_src, gis_precedence(order or osm_priority))
        if not pieces:
            return None
        geoms = [p[0] for p in pieces]; surf = [p[1] for p in pieces]
        srcs = [p[2] for p in pieces]; tags = [p[3] for p in pieces]
        print(f'    [gis map] {n_in} polygons -> {len(geoms)} non-overlapping pieces '
              f'(overlaps resolved by precedence)')
    if len(geoms) > max_features:
        print(f'    [gis map] {len(geoms)} polygons near the track; keeping the '
              f'{max_features} largest (raise --gis-max to keep more)')
        order = sorted(range(len(geoms)), key=lambda k: -geoms[k].area)[:max_features]
        keep = sorted(order)                       # preserve precedence order
        geoms = [geoms[k] for k in keep]; surf = [surf[k] for k in keep]
        srcs = [srcs[k] for k in keep]; tags = [tags[k] for k in keep]
    gdf = gpd.GeoDataFrame({'surface': surf, 'src': srcs, 'tag': tags}, geometry=geoms,
                           crs=base.METRIC).to_crs('EPSG:4326')
    fc = json.loads(gdf.to_json())
    for f in fc['features']:
        f.pop('id', None)
        f['geometry']['coordinates'] = _round_coords(f['geometry']['coordinates'])
    return fc


def build_html(name, layers, agree, lines_on, scrim, basemap, weight, truth_fx=True,
               gis_view='footprint',
               gap_m=2.8, smooth_pts=6, raw_on=False, raw_pts=None,
               order=None, avail=None,
               gismap=None, gis_op=55, gis_overlay_on=False):
    payload = [{'name': nm, 'key': key, 'fc': fc, 'mode': mode, 'ramp': ramp, 'role': role}
               for (nm, fc, key, mode, ramp, role) in layers]
    return (HTML.replace('__NAME__', name)
            .replace('__AGREE__', agree)
            .replace('__LAYERS__', json.dumps(payload))
            .replace('__LINES_ON__', json.dumps(lines_on))
            .replace('__ROLES__', json.dumps(LINE_ROLES))
            .replace('__GISMAP__', json.dumps(gismap) if gismap else 'null')
            .replace('__GIS_OP__', str(int(gis_op)))
            .replace('__GIS_OVERLAY_ON__', 'true' if gis_overlay_on else 'false')
            .replace('__SCRIM__', str(scrim))
            .replace('__BASEMAP__', basemap)
            .replace('__WT__', str(weight))
            .replace('__TRUTH_FX__', 'true' if truth_fx else 'false')
            .replace('__GIS_VIEW__', 'Label view: features grown by their reach'
                     if gis_view == 'label' else 'Footprint view: polygons as published')
            .replace('__GAP_M__', str(float(gap_m)))
            .replace('__SMOOTH_W__', str(int(smooth_pts)))
            .replace('__RAW_ON__', 'true' if raw_on else 'false')
            .replace('__RAWPTS__', json.dumps([[round(p['lon'], 7), round(p['lat'], 7)]
                                               for p in raw_pts]) if raw_pts else 'null')
            .replace('__GIS_ORDER__', json.dumps(order or OSM_PRIORITY_ORDERS['first']))
            .replace('__GIS_AVAIL__', json.dumps(sorted(avail or GIS_SOURCES)))
            .replace('__STADIA_KEY__', STADIA_KEY))


def csv_for_gpx(gpx_path, csvs, pts=None, ranges=None):
    """The walk CSV for a GPX, by filename stamp, CONFIRMED against the clock.

    GPS/REC_<stamp>.gpx should pair with GPS/REC_<stamp>_GPS.csv - but a file can be
    misnamed, and then the names agree while the data does not. So a stamp match is only
    returned if the two actually overlap in time; otherwise this returns None with a
    warning and the caller falls back to overlap matching, which cannot make that mistake.
    """
    import os as _os
    stem = _os.path.splitext(_os.path.basename(gpx_path))[0]
    hit = None
    for p in csvs:
        b = _os.path.splitext(_os.path.basename(p))[0]
        if b == stem + '_GPS' or b == stem:
            hit = p
            break
    if hit is None or not pts or not ranges or not ranges.get(hit):
        return hit
    g0, g1 = pts[0]['tms'], pts[-1]['tms']
    c0, c1 = ranges[hit]
    if max(0, min(g1, c1) - max(g0, c0)) > 0:
        return hit
    import datetime as _dt
    f = lambda ms: _dt.datetime.fromtimestamp(ms / 1000, _dt.timezone.utc).strftime('%H:%M:%S')
    print(f'  [WARN] {stem}: {_os.path.basename(hit)} matches by NAME but not in time - '
          f'GPX {f(g0)}-{f(g1)} vs CSV {f(c0)}-{f(c1)} (no overlap). One of the two is '
          'misnamed; ignoring the name match.')
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('walks_dir', nargs='?', default='.')
    ap.add_argument('--sidewalk', default='sidewalk_inventory.geojson')
    ap.add_argument('--openspace', default='open_space.geojson')
    ap.add_argument('--osm', default='export.geojson',
                    help='OpenStreetMap highway ways with surface=* tags (an Overpass '
                         'export); used as a 4th GIS layer')
    ap.add_argument('--no-osm', action='store_true', help='ignore the OSM layer')
    ap.add_argument('--gis-reach', type=float, default=3.0,
                    help='metres: how far a GPS fix may be from a sidewalk or open-space '
                         'polygon and still take its surface. One value for all polygon '
                         'sources so none wins by reaching further (default 3, about the '
                         'GPS error)')
    ap.add_argument('--path-halfwidth', type=float, default=1.0,
                    help='added to the reach for OSM paths, whose distance is to the '
                         'centreline (default 1 m = half a 2 m path)')
    ap.add_argument('--lc-reach', type=float, default=1.5,
                    help='land-cover reach; it only matters at polygon edges (default 1.5)')
    ap.add_argument('--sw-reach', type=float, default=None, help='override: sidewalk reach')
    ap.add_argument('--os-reach', type=float, default=None, help='override: open-space reach')
    ap.add_argument('--osm-reach', type=float, default=None, help='override: OSM reach')
    ap.add_argument('--osm-tol', type=float, default=None, dest='osm_reach_legacy',
                    help=argparse.SUPPRESS)
    ap.add_argument('--gis-map-reach', action='store_true',
                    help='draw the GIS surface map in LABEL view: sidewalk, open-space and '
                         'OSM features grown by their reach, so the map shows the label a '
                         'GPS fix at each spot would get. Default is FOOTPRINT view, the '
                         'polygons as published.')
    ap.add_argument('--legacy-reach', action='store_true',
                    help='restore the old per-source reach (15/8/5/1.5 m) to reproduce '
                         'earlier outputs')
    ap.add_argument('--reach-sweep', action='store_true',
                    help='also compute GIS agreement and class shares at reaches '
                         f'{SWEEP_REACH} m -> gis_reach_sweep.csv')
    ap.add_argument('--gis-priority', default=None,
                    help='full GIS source order, highest priority first, e.g. '
                         '"sidewalk,osm,landcover,open_space". Overrides --osm-priority. '
                         'Applied to the GIS line, the saved geojson, summary.csv and the '
                         'surface map overlap resolution, so all of them agree.')
    ap.add_argument('--osm-priority',
                    choices=['first', 'after-sidewalk', 'after-landcover', 'last'],
                    default=None,
                    help="where OSM sits in the precedence. first: OSM > sidewalk > land "
                         "cover > open space. after-sidewalk: sidewalk > OSM > land cover "
                         "> open space. after-landcover: sidewalk > land cover > OSM > "
                         "open space. last: OSM only fills otherwise-unknown points.")
    ap.add_argument('--osm-compacted', choices=['loose', 'paved'], default='loose',
                    help="how to class surface=compacted (stone dust in most Boston parks)")
    ap.add_argument('--landcover', default=None,
                    help='land-cover tile .shp (or folder of LCLU_*.shp)')
    ap.add_argument('--out', default='walk_maps_satellite')
    ap.add_argument('--lines', default=','.join(DEFAULT_ON),
                    help='lines on at load: any of ' + ','.join(LINE_ROLES)
                         + '  (all are toggleable in the page regardless)')
    ap.add_argument('--basemap', default='Esri World Imagery',
                    help="basemap selected at load; use a barebones one "
                         "('CARTO Positron', 'Esri World Gray Canvas', "
                         "'Stadia Stamen Toner Lite') to read the GIS line without imagery")
    ap.add_argument('--barebones', action='store_true',
                    help="shorthand for --basemap 'CARTO Positron' --scrim 0")
    ap.add_argument('--gis-map', action='store_true',
                    help="open on the GIS surface classification AS the map, layered on "
                         "CARTO Positron (same as --basemap 'GIS surface + CARTO "
                         "Positron'). Every 'GIS surface + ...' basemap is in the "
                         "picker regardless.")
    ap.add_argument('--gis-overlay', action='store_true',
                    help='also draw the GIS polygons over the starting basemap (e.g. '
                         'over imagery)')
    ap.add_argument('--no-gis-map', action='store_true',
                    help='do not compute or embed GIS polygons (smaller, faster files)')
    ap.add_argument('--gis-pad-m', type=float, default=80.0,
                    help='how far around the track to include GIS polygons (metres)')
    ap.add_argument('--gis-opacity', type=int, default=55,
                    help='GIS polygon fill, 10-100 percent')
    ap.add_argument('--gis-overlap', choices=['resolve', 'stack'], default='resolve',
                    help="resolve: where sources overlap, show only the surface the "
                         "precedence picks (default). stack: draw every polygon "
                         "translucently, so overlaps show as mixed colour")
    ap.add_argument('--gis-max', type=int, default=8000,
                    help='cap on embedded GIS polygons per walk')
    ap.add_argument('--scrim', type=int, default=25,
                    help='white scrim over imagery, 0-90 percent (default 25)')
    ap.add_argument('--no-trim', action='store_true',
                    help='keep the unlabelled head and tail of each track. By default the '
                         'track is clipped to the span where the ground truth is known, '
                         'since the ends are GPX recorded before the first tag or after '
                         'the last and carry no data.')
    ap.add_argument('--pred-dir', default=None,
                    help="folder holding <stem>_pred_3class.geojson etc. Point it at "
                         "predict_walks.py's --out to choose which model's predictions are "
                         "drawn, e.g. GPS/walk_maps_cnn. Searched before --out and "
                         "<walks_dir>/walk_maps.")
    ap.add_argument('--map-match', action='store_true',
                    help='DRAW the lines on a track snapped to the mapped footways (OSM '
                         'export). Display only: labels still use the raw GPS fixes.')
    ap.add_argument('--match-sigma', type=float, default=4.0,
                    help='map matching: GPS noise scale in metres (default 4)')
    ap.add_argument('--match-radius', type=float, default=18.0,
                    help='map matching: max distance to a candidate path in metres (default 18)')
    ap.add_argument('--match-null-m', type=float, default=12.0,
                    help='map matching: fixes with no path closer than about this stay raw '
                         '(default 12 m)')
    ap.add_argument('--gap-m', type=float, default=2.8,
                    help='spacing between parallel lines in metres (default 2.8). The outer '
                         'lines sit this far off the GPS track, so lower it for figures.')
    ap.add_argument('--smooth-pts', type=int, default=6,
                    help='GPS fixes averaged either side of each point (default 6, ~13 s '
                         'of walking). 0 draws the raw track. Smoothing cuts bends.')
    ap.add_argument('--raw-gps', action='store_true',
                    help='start with the raw GPS fixes shown as dots (toggleable in the page)')
    ap.add_argument('--no-truth-outline', action='store_true',
                    help='start with the ground-truth black outline and shadow off '
                         '(toggleable in the page either way)')
    ap.add_argument('--weight', type=float, default=4.2,
                    help='line width; wider than the original 3.4 for imagery')
    a = ap.parse_args()

    lines_on = [r.strip() for r in a.lines.split(',') if r.strip()]
    bad = [r for r in lines_on if r not in LINE_ROLES]
    if bad:
        sys.exit(f'unknown line role(s) {bad}; choose from {LINE_ROLES}')
    if a.barebones:
        a.basemap, a.scrim = 'CARTO Positron', 0
    if a.gis_map:
        if a.no_gis_map:
            sys.exit('--gis-map needs the GIS polygons; drop --no-gis-map')
        a.basemap, a.scrim = 'GIS surface + CARTO Positron', 0
    a.gis_opacity = max(10, min(100, a.gis_opacity))
    if a.osm_reach_legacy is not None and a.osm_reach is None:
        a.osm_reach = a.osm_reach_legacy          # old --osm-tol still works
    # validate the source order NOW, so a typo fails before the slow GIS load
    if a.gis_priority:
        order = parse_priority(a.gis_priority)
        if a.osm_priority:
            print('  [note] --gis-priority given, so --osm-priority is ignored')
    else:
        order = list(OSM_PRIORITY_ORDERS[a.osm_priority or 'first'])
    a.scrim = max(0, min(90, a.scrim))

    def _res(p):
        if p and os.path.exists(p):
            return p
        c = os.path.join(a.walks_dir, os.path.basename(p)) if p else None
        return c if c and os.path.exists(c) else p

    a.sidewalk = _res(a.sidewalk); a.openspace = _res(a.openspace); a.osm = _res(a.osm)
    # GIS inputs usually live together (e.g. gis/). Any one that was not found in the
    # working folder or the walks folder is also looked for beside each of the others.
    # Before, only the OSM file did this, so --landcover gis/... found land cover while
    # sidewalk_inventory.geojson and open_space.geojson in the same folder were skipped.
    def _beside(name, refs):
        if not name or os.path.exists(name):
            return name
        for ref in refs:
            if not ref or not os.path.exists(ref):
                continue
            d = ref if os.path.isdir(ref) else os.path.dirname(ref)
            c = os.path.join(d, os.path.basename(name))
            if d and os.path.exists(c):
                return c
        return name
    for _ in range(2):          # twice, so a file found by one pass can anchor the next
        a.sidewalk = _beside(a.sidewalk, (a.landcover, a.osm, a.openspace))
        a.openspace = _beside(a.openspace, (a.landcover, a.osm, a.sidewalk))
        a.osm = _beside(a.osm, (a.landcover, a.sidewalk, a.openspace))
    if a.out == 'walk_maps_satellite':
        a.out = os.path.join(a.walks_dir, 'walk_maps_satellite')
    os.makedirs(a.out, exist_ok=True)
    print(f'walks dir: {a.walks_dir}\nout: {a.out}')
    print(f'lines on at load: {lines_on}   basemap: {a.basemap}   scrim: {a.scrim}%\n')

    print('Loading GIS layers (once)...')
    sw = base.read_layer(a.sidewalk)
    osp = base.read_layer(a.openspace)
    lc = base.read_landcover(a.landcover, a.walks_dir)
    sw_field = next((c for c in base.SW_FIELDS if sw is not None and c in sw.columns), None)
    lc_field = next((c for c in base.LC_FIELDS if lc is not None and c in lc.columns), None)
    layers = (base.build_tree(sw, [base.sw_class(v) for v in sw[sw_field]] if sw_field else []),
              base.build_tree(lc, [base.lc_class(v) for v in lc[lc_field]] if lc_field else []),
              base.build_tree(osp, ['grass'] * len(osp) if osp is not None else []))
    # name the prediction line after whichever model wrote it, so a CNN map is not
    # labelled 'XGBoost' by accident
    _model_label = 'model 3-class'
    for _d in (a.pred_dir, a.out, os.path.join(a.walks_dir, 'walk_maps')):
        _rj = os.path.join(_d, 'predict_walks_run.json') if _d else None
        if _rj and os.path.exists(_rj):
            try:
                _m = (json.load(open(_rj)) or {}).get('model')
                if _m:
                    _model_label = f'model 3-class ({"1D-CNN" if _m == "cnn" else "XGBoost"})'
                    print(f'  prediction layers from {_d} ({_m})')
            except Exception:
                pass
            break
    osm = None if a.no_osm else read_osm(a.osm, compacted=a.osm_compacted)
    if a.legacy_reach:
        reach = dict(LEGACY_REACH)
        print('  GIS reach: LEGACY per-source values (sidewalk 15, open space 8, OSM 5, '
              'land cover 1.5 m) - reproduces earlier outputs')
    else:
        reach = make_reach(a.gis_reach, a.path_halfwidth, a.lc_reach,
                           sidewalk=a.sw_reach, open_space=a.os_reach, osm=a.osm_reach)
        print('  GIS reach (m): ' + ', '.join(f'{k} {v:g}' for k, v in reach.items()))
    sweep_rows = []
    match_net = load_match_network(a.osm) if a.map_match else None
    if a.map_match and not match_net:
        print(f'  [WARNING] --map-match needs the OSM export ({a.osm!r}); drawing raw tracks')
    avail = {k for k, ok in (('sidewalk', sw is not None), ('landcover', lc is not None),
                             ('open_space', osp is not None), ('osm', osm is not None))
             if ok}
    print('  GIS source priority (highest first): '
          + ' > '.join(k + ('' if k in avail else ' (not loaded)') for k in order))
    _missing = [(n, f, why) for n, f, ok, why in (
        ('sidewalk inventory', a.sidewalk, sw is not None,
         'the authoritative sidewalk material layer; without it sidewalks fall back to '
         'land cover or OSM'),
        ('open space', a.openspace, osp is not None,
         'park extents; without it park lawns rely on land cover alone'))
        if not ok]
    for n, f, why in _missing:
        print(f'\n  [WARNING] {n.upper()} NOT LOADED (looked for {f!r}) - {why}.')
        print(f'            Pass it explicitly, e.g. --{"sidewalk" if n.startswith("side") else "openspace"} '
              f'gis/{os.path.basename(f or "")}')
    if lc is None:
        print('\n  [WARNING] NO LAND COVER LOADED. Land cover is the wall-to-wall layer, so '
              'most GIS coverage comes from it')
        print('            and without it far more points come out "unknown". Pass it '
              'explicitly, e.g.')
        print('              --landcover gis/LCLU_R07C19.shp      (a .shp, or a folder of '
              'LCLU_*.shp tiles)')
        print(f'            Without --landcover the script only searches '
              f'{os.path.abspath(a.walks_dir)} for LCLU_*.shp.\n')
    if osm is None and not a.no_osm:
        print('  [note] OSM path surfaces not loaded - pass --osm with the path to your '
              'Overpass export, or --no-osm to silence this')
    print(f'  sidewalk={len(layers[0][1])}, land cover={len(layers[1][1])}, '
          f'open space={len(layers[2][1])}\n')

    gpx_files = sorted(glob.glob(os.path.join(a.walks_dir, '*.gpx')))
    _all_csv = [p for p in glob.glob(os.path.join(a.walks_dir, '*.csv')) if base.is_insole_csv(p)]
    _gps = [p for p in _all_csv if '_GPS' in os.path.basename(p)]
    csvs = {p: base.csv_time_range(p) for p in (_gps or _all_csv)}
    print(f'{len(gpx_files)} GPX, {len(csvs)} insole CSV(s)\n')

    rows = []
    for gpx in gpx_files:
        pts, name = base.parse_gpx(gpx)
        stem = os.path.splitext(os.path.basename(gpx))[0]
        if len(pts) < 2:
            print(f'[skip] {stem}: <2 pts'); continue
        best = csv_for_gpx(gpx, csvs, pts, csvs)   # stamp match, confirmed by clock
        if best is None:                        # fallback: largest time overlap
            g0, g1 = pts[0]['tms'], pts[-1]['tms']
            ov = 0
            for p, r in csvs.items():
                if not r:
                    continue
                o = max(0, min(g1, r[1]) - max(g0, r[0]))
                if o > ov:
                    ov, best = o, p
            if best:
                print(f'[note] {stem}: no REC_<stamp>_GPS.csv; using {os.path.basename(best)} '
                      'by time overlap')
        tags = base.tagged_from_csv(best, pts) if best else [None] * len(pts)
        # ---- trim the unlabelled head and tail -------------------------------------
        # The GPX usually starts before the first tag and runs past the last, so the track
        # opens and closes with segments no layer can label. Those ends carry no data and
        # they stretch the map's extent, so the track is clipped to the span where the
        # GROUND TRUTH is known. This happens BEFORE the GIS lookup so every layer is
        # built on the trimmed track, and the prediction layers read from disk are cut by
        # the same indices below. --no-trim keeps the whole track.
        _npts_raw = len(pts)
        # The model layer can be unknown at the ends even where the ground truth is not:
        # its values come from insole STEPS matched onto fixes, and the last fixes of a
        # track often have no step within the matching window. Read it now, before the
        # trim, so the kept span is the part BOTH sources can label.
        _pred_tail = _pred_head = 0
        _pre = None
        for _d in (a.pred_dir, a.out, os.path.join(a.walks_dir, 'walk_maps')):
            _pp = os.path.join(_d, f'{stem}_pred_3class.geojson') if _d else None
            if _pp and os.path.exists(_pp):
                try:
                    _pre = [(f['properties'] or {}).get('surface')
                            for f in (json.load(open(_pp)).get('features') or [])]
                except Exception:
                    _pre = None
                break
        if _pre is not None and len(_pre) == len(tags):
            while (_pred_head < len(_pre)
                   and _pre[_pred_head] in (None, 'unknown')):
                _pred_head += 1
            while (_pred_tail < len(_pre) - _pred_head
                   and _pre[-1 - _pred_tail] in (None, 'unknown')):
                _pred_tail += 1
        elif _pre is not None:
            print(f'  [note] {stem}: prediction layer has {len(_pre)} segments for '
                  f'{len(tags)} - written for a different track, not used for trimming')
        _lo, _hi = 0, len(tags)
        if not a.no_trim and tags:
            while _lo < len(tags) and tags[_lo] is None:
                _lo += 1
            while _hi > _lo and tags[_hi - 1] is None:
                _hi -= 1
            # also drop ends the MODEL cannot label, so no layer starts or ends unknown
            _lo = max(_lo, _pred_head)
            _hi = min(_hi, len(tags) - _pred_tail)
            if _pred_head or _pred_tail:
                print(f'  {stem}: model layer unlabelled on {_pred_head} segment(s) at the '
                      f'start and {_pred_tail} at the end; trimming to the span both '
                      'sources cover')
            if _hi <= _lo:
                print(f'  [skip] {stem}: ground truth has no labelled segments'); continue
            if _lo or _hi < len(tags):
                _n0 = len(tags)
                pts = pts[_lo:_hi + 1]
                tags = tags[_lo:_hi]
                print(f'  {stem}: trimmed {_lo} unlabelled segment(s) from the start, '
                      f'{_n0 - _hi} from the end ({_hi - _lo} of {_n0} kept)')
        nearest = point_nearest([(p['lon'], p['lat']) for p in pts], layers, osm=osm)
        cands = apply_reach(nearest, reach)
        gis, gsrc = resolve_candidates(cands, order)
        if a.reach_sweep:
            for r_ in reach_sweep(nearest, tags, order, a.path_halfwidth, reach['landcover']):
                sweep_rows.append(dict(walk=stem, **r_))
        # ---- coverage checks, reported per walk ------------------------------------
        # Ground truth: a run of unknowns at the START means the GPX began recording
        # before the first surface marker, so the walk does not start where the tagging
        # does. Unknowns in the MIDDLE are the recorded pauses and are expected.
        _u = [i for i, v in enumerate(tags) if v is None]
        if _u:
            _lead = 0
            while _lead < len(tags) and tags[_lead] is None:
                _lead += 1
            _trail = 0
            while _trail < len(tags) and tags[-1 - _trail] is None:
                _trail += 1
            _mid = len(_u) - _lead - _trail
            _msg = []
            if _lead:
                _msg.append(f'{_lead} at the START')
            if _trail:
                _msg.append(f'{_trail} at the END')
            if _mid:
                _msg.append(f'{_mid} in the middle (recorded pauses)')
            _bad = _lead or _trail
            print(f'  {"[WARN]" if _bad else "  note"} {stem}: ground truth unlabelled on '
                  f'{len(_u)}/{len(tags)} segments - ' + ', '.join(_msg))
            if _bad:
                print('          a leading or trailing run means the GPX extends beyond the '
                      'tagged part of the walk; trim the GPX or tag from the first step.')
        # GIS: no source answered anywhere is a coverage problem, not a classification one
        _gu = sum(1 for v in gis if v in (None, 'unknown'))
        if _gu == len(gis):
            print(f'  [WARN] {stem}: GIS returned NOTHING for any segment. The walk is '
                  'probably outside the loaded layers - check that the land-cover tile and '
                  'the municipal sidewalk/open-space files cover these coordinates '
                  f'({pts[0]["lat"]:.5f}, {pts[0]["lon"]:.5f}).')
        elif _gu > 0.25 * len(gis):
            print(f'  [WARN] {stem}: GIS unclassified on {100*_gu/len(gis):.0f}% of segments '
                  '- likely partial coverage at the edge of a layer.')
        tagged_fc = base.seg_fc(pts, tags, 'tag')
        gis_fc = base.seg_fc(pts, gis, 'surface')
        json.dump(tagged_fc, open(os.path.join(a.out, f'{stem}_tagged.geojson'), 'w'))
        # DISPLAY geometry. The saved geojsons above, and every label, use the raw fixes;
        # only the drawn lines move onto the matched track.
        draw_fc = tagged_fc
        if match_net:
            mm, snapped = map_match([(p['lat'], p['lon']) for p in pts], match_net,
                                    sigma=a.match_sigma, radius=a.match_radius,
                                    null_m=a.match_null_m)
            draw_fc = base.seg_fc([dict(p, lat=q[0], lon=q[1]) for p, q in zip(pts, mm)],
                                  tags, 'tag')
            import math as _m
            mv = [_m.hypot((q[1] - p['lon']) * 111320 * _m.cos(_m.radians(p['lat'])),
                           (q[0] - p['lat']) * 111320) for p, q in zip(pts, mm)]
            mvs = sorted(v for v, sn in zip(mv, snapped) if sn)
            print(f'  [map-match] {stem}: {100*sum(snapped)/len(snapped):.0f}% of fixes snapped'
                  + (f', median move {mvs[len(mvs)//2]:.1f} m' if mvs else '')
                  + f'; {len(snapped)-sum(snapped)} left raw (no path fits)')
        json.dump(gis_fc, open(os.path.join(a.out, f'{stem}_surface.geojson'), 'w'))

        def _load(tag):
            for d in (a.pred_dir, a.out, os.path.join(a.walks_dir, 'walk_maps')):
                if not d:
                    continue
                pth = os.path.join(d, f'{stem}_{tag}.geojson')
                if os.path.exists(pth):
                    try:
                        fc_ = json.load(open(pth))
                    except Exception:
                        continue
                    f_ = fc_.get('features') or []
                    # these were written for the untrimmed track; cut them to match, and
                    # drop any unknown tail the model itself left
                    if _lo or _hi < len(f_):
                        if len(f_) >= _hi:
                            fc_['features'] = f_[_lo:_hi]
                        else:
                            print(f'  [warn] {stem}: {tag} has {len(f_)} segments for a '
                                  f'track of {_hi - _lo} - not trimmed, check it matches')
                    return fc_
            return None

        maplayers = [('truth', draw_fc, 'tag', 'cat', None, 'truth'),
                     ('GIS (+OSM)' if osm else 'GIS (3-layer)', gis_fc, 'surface',
                      'cat', None, 'gis')]
        fc = _load('pred_3class')
        if fc:
            maplayers.append((_model_label, fc, 'surface', 'cat', None, 'model3'))
        ev = _load('evenness'); hd = _load('hardness')
        if ev:
            maplayers.append(('evenness', ev, 'value', 'cont', RAMP_EVEN, 'evenness'))
        if hd:
            maplayers.append(('hardness', hd, 'value', 'cont', RAMP_HARD, 'hardness'))
        # plain: the bare track, no classification. Reuses the truth geometry and simply
        # ignores the values, so it is always available.
        maplayers.append(('plain track', draw_fc, 'tag', 'plain', None, 'plain'))

        have = {m[5] for m in maplayers}
        miss = [r for r in lines_on if r not in have]
        if miss:
            print(f'  [note] {stem}: no layer for {miss} - those toggles are disabled')

        gismap = None
        if not a.no_gis_map:
            try:
                gismap = gis_surface_fc(pts, layers, pad_m=a.gis_pad_m,
                                        max_features=a.gis_max, osm=osm,
                                        overlap=a.gis_overlap, order=order,
                                        reach=reach if a.gis_map_reach else None)
            except Exception as ex:
                print(f'  [gis map] {stem}: could not build polygons ({ex}); '
                      'continuing without them')
            if gismap:
                cnt = collections.Counter(f['properties']['src'] for f in gismap['features'])
                print(f'  [gis map] {stem}: {len(gismap["features"])} polygons {dict(cnt)}')
            elif a.basemap.startswith('GIS surface'):
                print(f'  [gis map] {stem}: no GIS polygons near the track - '
                      f'the page will fall back to the first basemap')
        html = build_html(f'{name} ({stem})', maplayers,
                          'agreement computed client-side', lines_on, a.scrim,
                          a.basemap, a.weight, gismap=gismap, gis_op=a.gis_opacity,
                          gis_overlay_on=a.gis_overlay,
                          truth_fx=not a.no_truth_outline,
                          gis_view='label' if a.gis_map_reach else 'footprint',
                          gap_m=a.gap_m, smooth_pts=max(0, a.smooth_pts), raw_on=a.raw_gps,
                          raw_pts=pts if match_net else None,
                          order=order, avail=avail)
        open(os.path.join(a.out, f'{stem}.html'), 'w', encoding='utf-8').write(html)

        CL = {'paved', 'grass', 'loose'}
        n = agn = 0
        for i in range(len(pts) - 1):
            t, gg = (tags[i] or 'unknown'), gis[i]
            if t in CL and gg in CL:
                n += 1; agn += (t == gg)
        gc = collections.Counter(gis); sc = collections.Counter(gsrc)
        tc = collections.Counter(t or 'unknown' for t in tags)
        rows.append({'walk': name, 'file': os.path.basename(gpx), 'points': len(pts),
                     # the geojsons cover GPX points [trim_start : trim_end+1]; anything
                     # reading them against the raw GPX must apply the same slice
                     'trim_start': _lo, 'trim_end': _hi, 'gpx_points': _npts_raw,
                     'paired_csv': (os.path.basename(best) if best else 'NONE'),
                     'truth_paved': tc.get('paved', 0), 'truth_grass': tc.get('grass', 0),
                     'truth_loose': tc.get('loose', 0),
                     'gis_paved': gc.get('paved', 0), 'gis_grass': gc.get('grass', 0),
                     'gis_loose': gc.get('loose', 0), 'gis_unknown': gc.get('unknown', 0),
                     'gis_from_osm': sc.get('osm', 0),
                     'agreement_pct': (round(100 * agn / n, 1) if n else None),
                     'agree_n': n})
        print(f'[ok] {stem:26s} pts={len(pts):4d} '
              f'csv={rows[-1]["paired_csv"]:26s} src={dict(sc)}')

    for r_ in rows:
        r_.update({f'reach_{k}_m': v for k, v in reach.items()})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    if sweep_rows:
        sw_df = pd.DataFrame(sweep_rows)
        pooled = (sw_df.groupby('reach_m')
                  .apply(lambda g: pd.Series({
                      'n': g['n'].sum(),
                      'agreement_pct': round((g['agreement_pct'].fillna(0) * g['n']).sum()
                                             / max(g['n'].sum(), 1), 1)}))
                  .reset_index())
        sw_df.to_csv(os.path.join(a.out, 'gis_reach_sweep.csv'), index=False)
        print('\nGIS agreement with ground truth vs reach (all walks pooled, by step):')
        for _, r_ in pooled.iterrows():
            print(f"  reach {r_['reach_m']:>4g} m: {r_['agreement_pct']:5.1f}%  (n={int(r_['n'])})")
        print(f"  per-walk detail, incl. GIS class shares -> {a.out}/gis_reach_sweep.csv")
    print(f"\nDone -> '{a.out}/' (one .html + geojsons per walk, plus summary.csv)")


if __name__ == '__main__':
    main()
