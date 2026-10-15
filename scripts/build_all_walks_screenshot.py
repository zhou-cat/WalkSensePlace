"""build_all_walks_screenshot.py - figure-ready walk maps.

Same pipeline, flags and outputs as build_all_walks_satellite.py (it imports that script
and changes only the page), plus:

  * the map is FRAMED to the extent of the GIS surface polygons: the map box takes that
    area's shape, is fitted to it exactly (fractional zoom), and cannot pan outside it.
    Every screenshot of a walk therefore has the same frame, whatever the basemap or
    lines, so panels of a multi-part figure line up.
  * a legend INSIDE the frame: surface colors always, plus the evenness / hardness scale
    whenever that line is on.
  * the map credits inside the frame, with a GIS data credit added.
  * SCREENSHOT MODE (button, or the S key): hides the info panel, layer picker and zoom
    buttons, leaving map, legend and credits.
  * the timeline drops its '(hover a line ...)' hint and takes the map's width.

    python scripts/build_all_walks_screenshot.py GPS --landcover gis/LCLU_R07C19.shp
    python scripts/build_all_walks_screenshot.py GPS --frame-width 1400 --start-clean

Output goes to <walks_dir>/walk_maps_screenshot unless --out is given. The frame is the
GIS polygon window - the track plus --gis-pad-m (default 80 m) - so use that flag to frame
tighter or looser. With --no-gis-map the frame falls back to the track, padded.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import build_all_walks_satellite as sat
except ImportError as e:
    sys.exit(f'build_all_walks_satellite.py must be in the same folder ({e})')

own = argparse.ArgumentParser(add_help=False)
own.add_argument('--frame-width', type=int, default=1100,
                 help='target map width in pixels (shrinks to fit the window)')
own.add_argument('--start-clean', action='store_true',
                 help='open pages already in screenshot mode')
own.add_argument('--text-scale', type=float, default=1.0,
                 help='enlarge the legend and map credits by this factor (e.g. 1.5) so they '
                      'stay readable once the screenshot is shrunk to figure size')
own.add_argument('--legend-pos', default='bottomleft',
                 choices=['topleft', 'topright', 'bottomleft', 'bottomright'])
A, rest = own.parse_known_args()
if '-h' in rest or '--help' in rest:
    own.print_help()
    print('\n...plus every flag of build_all_walks_satellite.py:\n')


def sw(s, a, b):
    assert s.count(a) == 1, f'page template changed; could not find: {a[:70]!r}'
    return s.replace(a, b)


H = sat.HTML
# ---- layout: a sized frame instead of a 74vh map ----
H = sw(H, "#map{height:74vh}#timeline{height:26vh;overflow-y:auto;overflow-x:hidden;background:#fafafa;border-top:1px solid #ddd}",
       ":root{--ts:__TEXT_SCALE__}"
       "body{background:#e9e9e9}"
       "#stage{display:flex;justify-content:center;align-items:flex-start;gap:8px;margin-top:12px}"
       "#tools{position:relative;width:44px;flex:0 0 44px}"
       "#tools .leaflet-control{float:none;margin:0 0 8px 0}"
       "#tools .leaflet-control-layers-expanded{position:absolute;left:0;top:0;z-index:4000;min-width:230px}"
       "#frame{position:relative;width:1100px;height:700px;background:#fff;box-shadow:0 0 0 1px #ccc}"
       "#map{width:100%;height:100%}"
       "#timeline{margin:0 auto 12px;background:#fff;overflow:hidden}"
       ".maplegend{background:rgba(255,255,255,.93);padding:6px 9px;border-radius:6px;font:12.5px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.25);color:#222}"
       ".leaflet-control-attribution{font-size:11px !important;zoom:var(--ts)}"
       ".maplegend{zoom:var(--ts)}"
       "#shotbtn{position:fixed;right:10px;bottom:10px;z-index:3000;padding:6px 10px;border-radius:7px;border:1px solid #bbb;background:#fff;cursor:pointer;font-size:12px}"
       ".shot .panel{display:none !important}"
       ".shot #shotbtn{opacity:.12}.shot #shotbtn:hover{opacity:1}")
H = sw(H, '<div id="map"></div><div id="timeline"></div>',
       '<div id="stage"><div id="tools"></div><div id="frame"><div id="map"></div></div>'
       '<div style="width:44px;flex:0 0 44px"></div></div><div id="timeline"></div>'
       '<button id="shotbtn" title="toggle screenshot mode (S)">screenshot mode (S)</button>')
# fractional zoom, so the frame fits the GIS extent exactly rather than the nearest level
H = sw(H, "L.control.layers(BASEMAPS,OVERLAYS,{position:'bottomleft',collapsed:true}).addTo(map);",
       "const LAYERS_CTL=L.control.layers(BASEMAPS,OVERLAYS,{position:'bottomleft',collapsed:true}).addTo(map);")
H = sw(H, "const map=L.map('map',{renderer:L.svg({padding:0.6}),preferCanvas:false});",
       "const map=L.map('map',{renderer:L.svg({padding:0.6}),preferCanvas:false,zoomSnap:0,zoomDelta:0.5,maxBoundsViscosity:1.0});")
# ---- timeline: no hover hint ----
H = sw(H, ",\n           `<text x=\"${padL+150}\" y=\"19\" font-size=\"11\" fill=\"#999\">(hover a line to locate it on the map)</text>`];",
       "];")
# ---- framing, legend and credits, right after the original fit ----
H = sw(H, "map.fitBounds(bd,{padding:[30,30],maxZoom:19});",
       """map.fitBounds(bd,{padding:[30,30],maxZoom:19});
// ---- screenshot framing: the map box takes the GIS window's shape and is locked to it ----
const FRAME_W=__FRAME_W__;
const FB=HAS_GIS?L.geoJSON(GISMAP).getBounds():bd.pad(0.15);
function sizeFrame(){
  const swc=FB.getSouthWest(), nec=FB.getNorthEast();
  const my=l=>Math.log(Math.tan(Math.PI/4+l*Math.PI/360));          // Web Mercator y
  const aspect=((nec.lng-swc.lng)*Math.PI/180)/(my(nec.lat)-my(swc.lat));
  let w=Math.min(FRAME_W,window.innerWidth-24-104), h=w/aspect;
  const maxH=window.innerHeight*0.74;
  if(h>maxH){h=maxH; w=h*aspect;}
  const fr=document.getElementById('frame');
  fr.style.width=Math.round(w)+'px'; fr.style.height=Math.round(h)+'px';
  document.getElementById('timeline').style.width=Math.round(w)+'px';
  map.invalidateSize(); map.setMinZoom(0); map.setMaxBounds(null);
  map.fitBounds(FB,{padding:[0,0],animate:false});
  map.setMinZoom(map.getZoom()); map.setMaxBounds(FB);
}
if(HAS_GIS) map.attributionControl.addAttribution(
  'GIS: MassGIS; City of Boston; &copy; OpenStreetMap contributors');
const LegendCtl=L.Control.extend({options:{position:'__LEGEND_POS__'},
  onAdd(){const d=L.DomUtil.create('div','maplegend'); d.id='maplegend';
          L.DomEvent.disableClickPropagation(d); return d;}});
new LegendCtl().addTo(map);
function renderMapLegend(){
  const el=document.getElementById('maplegend'); if(!el) return;
  const chip=k=>`<span style="display:inline-block;width:18px;height:7px;border-radius:3px;background:${palette()[k]};margin-right:7px;vertical-align:middle"></span>`;
  let h='<div style="font-weight:600;margin-bottom:1px">surface</div>'+
        ['paved','grass','loose','unknown'].map(k=>`<div>${chip(k)}${k}</div>`).join('');
  activeLayers().filter(Ly=>Ly.mode==='cont').forEach(Ly=>{
    const stops=rampFor(Ly).map(s=>s[1]).join(','), e=_ends[Ly.name]||['low','high'];
    h+=`<div style="font-weight:600;margin-top:7px">${Ly.name}</div>`+
       `<div style="width:160px;height:9px;border-radius:3px;background:linear-gradient(to right,${stops})"></div>`+
       `<div style="display:flex;justify-content:space-between;width:160px;font-size:11px;color:#444"><span>0 ${e[0]}</span><span>100 ${e[1]}</span></div>`;});
  el.innerHTML=h;
}""")
# ---- start-up: size the frame first, legend follows every refresh, screenshot toggle ----
H = sw(H, "applyScrim(); refreshAll(); window.addEventListener('resize',buildTimeline);",
       """const _refreshAll=refreshAll; refreshAll=function(){_refreshAll(); renderMapLegend();};
// zoom and layer controls move OUT of the map, into the toolbar beside the frame:
// still fully usable, never inside a capture
const TOOLS=document.getElementById('tools');
[map.zoomControl, LAYERS_CTL].forEach(c=>{ if(c&&c.getContainer) TOOLS.appendChild(c.getContainer()); });
sizeFrame(); applyScrim(); refreshAll();
window.addEventListener('resize',()=>{sizeFrame(); buildTimeline();});
const toggleShot=()=>document.body.classList.toggle('shot');
document.getElementById('shotbtn').addEventListener('click',toggleShot);
document.addEventListener('keydown',e=>{ if((e.key==='s'||e.key==='S')&&!/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) toggleShot(); });
if(__START_SHOT__) document.body.classList.add('shot');""")
H = (H.replace('__FRAME_W__', str(int(A.frame_width)))
      .replace('__LEGEND_POS__', A.legend_pos)
      .replace('__START_SHOT__', 'true' if A.start_clean else 'false')
      .replace('__TEXT_SCALE__', f'{max(0.5, A.text_scale):g}'))
sat.HTML = H

# default output folder: walk_maps_screenshot, so the satellite maps are not overwritten
_orig_parse = argparse.ArgumentParser.parse_args


def _parse(self, *x, **k):
    ns = _orig_parse(self, *x, **k)
    if getattr(ns, 'out', None) == 'walk_maps_satellite' and hasattr(ns, 'walks_dir'):
        ns.out = os.path.join(ns.walks_dir, 'walk_maps_screenshot')
    return ns


argparse.ArgumentParser.parse_args = _parse
sys.argv = [sys.argv[0]] + rest

if __name__ == '__main__':
    sat.main()
