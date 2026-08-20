"""
v2 — sample the CITY OF BOSTON SIDEWALK INVENTORY (per-sidewalk material) along a GPX
track, offline. No network: you download the GeoJSON once in a browser, this reads it.

1) Download the GeoJSON (public domain) from:
   https://data.boston.gov/dataset/57b57bc6-6344-48ca-9316-95961213a38e/resource/2faee1d9-484a-4f3a-b42f-d5c3b6663f75/download/sidewalk_inventory.geojson
   Save it as  sidewalk_inventory.geojson  in your WalkSensePlace folder.

2) Run:
   pip install shapely gpxpy              # already installed for you
   python sample_surface_boston_v2.py activity_22645980458.gpx sidewalk_inventory.geojson

Outputs track_surface.geojson (per-segment 'surface' in your classes) + a profile.
Zero network calls.

Notes vs v1 (OSM):
- Boston's file stores material in a column that is NOT called 'surface'. This script
  AUTO-DETECTS the material field (tries common names, else picks the best text column)
  and prints the raw material values it saw + how each mapped, so you can verify/adjust.
- The whole city is in the file (~large). We pre-filter to sidewalks near the track for speed.
"""
import sys, json, math
import gpxpy
from shapely.geometry import LineString, Point, shape

GPX      = sys.argv[1] if len(sys.argv) > 1 else 'activity_22645980458.gpx'
SW_JSON  = sys.argv[2] if len(sys.argv) > 2 else 'sidewalk_inventory.geojson'
TOL_M    = 20     # nearest-sidewalk tolerance (m); a bit wider than OSM b/c sidewalks flank roads
NEAR_M   = 300    # pre-filter: keep sidewalks whose vertices fall within this of the track bbox

# --- Boston MATERIAL codes -> your classes (from the data dictionary) ---
#   CC = Cement Concrete, BC = Bituminous Concrete (asphalt), GB = Granite Block,
#   BR = Brick, OT = Other.  All built sidewalk surfaces map to 'paved'; OT/blank -> unknown.
BOSTON_CODES = {
    'cc':'paved',   # cement concrete
    'bc':'paved',   # bituminous concrete (asphalt)
    'gb':'paved',   # granite block
    'br':'paved',   # brick
    'cb':'paved',   # concrete/brick block
    'bl':'paved',   # bluestone / block
    'ot':'unknown', # other (could be anything -> leave unknown)
}
# extra spellings / full words, just in case the export uses them
PAVED = {'concrete','cement','cement concrete','asphalt','bituminous','bituminous concrete',
         'brick','granite','granite block','stone','pavers','paver','bluestone','flagstone','slate'}
GRASS = {'grass','turf','lawn'}
LOOSE = {'dirt','earth','soil','gravel','stone dust','stonedust','unpaved','sand','cinder'}
def to_class(v):
    if v is None: return 'unknown'
    s = str(v).strip().lower()
    if not s or s in {'none','null','n/a','na','#n/a',''}: return 'unknown'
    if s in BOSTON_CODES: return BOSTON_CODES[s]          # two-letter codes (CC/BC/GB/BR/OT)
    if s in PAVED: return 'paved'
    if s in GRASS: return 'grass'
    if s in LOOSE: return 'loose'
    if any(k in s for k in ['concret','asphalt','brick','granit','paver','stone','slate','cement','bitum']): return 'paved'
    if 'grass' in s or 'turf' in s or 'lawn' in s: return 'grass'
    if any(k in s for k in ['dirt','gravel','sand','soil','earth','cinder','dust','unpaved']): return 'loose'
    return 'unknown'

CANDIDATE_FIELDS = ['MATERIAL','material','Material','surface','SURFACE']   # Boston uses 'MATERIAL'

# --- track ---
with open(GPX) as f:
    g = gpxpy.parse(f)
pts = [(p.longitude, p.latitude) for tr in g.tracks for seg in tr.segments for p in seg.points]
print(f"{len(pts)} track points")
clat = sum(p[1] for p in pts)/len(pts); clon = sum(p[0] for p in pts)/len(pts)
def xy(lon, lat):
    return ((lon-clon)*math.cos(math.radians(clat))*111320.0, (lat-clat)*110540.0)
# track bbox in metres for pre-filter
txy = [xy(*p) for p in pts]
minx=min(a for a,_ in txy)-NEAR_M; maxx=max(a for a,_ in txy)+NEAR_M
miny=min(b for _,b in txy)-NEAR_M; maxy=max(b for _,b in txy)+NEAR_M

# --- load Boston sidewalks ---
try:
    gj = json.load(open(SW_JSON, encoding='utf-8'))
except FileNotFoundError:
    sys.exit(f"'{SW_JSON}' not found. Download the Sidewalk Inventory GeoJSON first (see header).")
feats = gj.get('features', [])
print(f"{len(feats)} sidewalk features in file")
if not feats:
    sys.exit("No features in file.")

# --- detect coordinate system: Boston's shapefile-derived GeoJSON is usually MA State
# Plane (feet, EPSG 2249), NOT lat/lon. Sniff the first coordinate and reproject if needed. ---
def _first_xy(features):
    for ft in features:
        gm = ft.get('geometry') or {}
        cs = gm.get('coordinates')
        while isinstance(cs, list) and cs and isinstance(cs[0], list):
            cs = cs[0]
        if isinstance(cs, list) and len(cs) >= 2 and all(isinstance(v,(int,float)) for v in cs[:2]):
            return cs[0], cs[1]
    return None
fx = _first_xy(feats)
is_lonlat = fx is not None and -180 <= fx[0] <= 180 and -90 <= fx[1] <= 90
reproject = None
if not is_lonlat:
    print(f"coords look projected (first vertex ~{fx}); reprojecting to lat/lon ...")
    try:
        from pyproj import Transformer
        # EPSG:2249 = NAD83 / Massachusetts Mainland (US survey feet) — Boston's standard
        _t = Transformer.from_crs("EPSG:2249", "EPSG:4326", always_xy=True)
        reproject = lambda x, y: _t.transform(x, y)   # returns (lon, lat)
    except Exception as ex:
        sys.exit("Coordinates are projected (not lat/lon) and pyproj isn't available to "
                 f"reproject: {ex}\nInstall it with:  pip install pyproj")
else:
    print(f"coords are lat/lon (first vertex ~{fx})")

# --- auto-detect the material field ---
props0 = feats[0].get('properties', {}) or {}
field = next((f for f in CANDIDATE_FIELDS if f in props0), None)
if field is None:
    # pick the text column with the most distinct short string values (likely 'material')
    import collections
    scores = {}
    for k in props0:
        vals = [str((ft.get('properties') or {}).get(k)) for ft in feats[:2000]]
        txt = [v for v in vals if v and not v.replace('.','',1).replace('-','',1).isdigit()]
        uniq = set(v.lower() for v in txt if len(v) <= 25)
        if txt: scores[k] = len(uniq)
    field = max(scores, key=scores.get) if scores else None
print(f"using material field: {field!r}   (available columns: {list(props0)[:12]}{'...' if len(props0)>12 else ''})")
if field is None:
    sys.exit("Could not find a material column. Paste me one feature's properties and I'll set it.")

# --- build sidewalk lines (metric), pre-filtered near the track, with mapped class ---
import collections
from shapely.ops import transform as _shp_transform

def _to_local(x, y, z=None):
    lon, lat = (reproject(x, y) if reproject is not None else (x, y))
    return xy(lon, lat)                       # -> local metres

raw_counter = collections.Counter()
type_counter = collections.Counter()
ways = []
for ft in feats:
    geom = ft.get('geometry')
    if not geom: continue
    val = (ft.get('properties') or {}).get(field)
    raw_counter[str(val)] += 1
    cls = to_class(val)
    try:
        g2 = shape(geom)                      # Polygon / MultiPolygon / LineString / Multi*
        if g2.is_empty: continue
        gl = _shp_transform(_to_local, g2)    # reproject (if needed) + to local metres, any type
    except Exception:
        continue
    type_counter[g2.geom_type] += 1
    ways.append((cls, gl))
print(f"{len(ways)} sidewalk geometries built (whole file)  types={dict(type_counter)}")

# show the raw material values so you can verify the mapping
print("\nRaw material values seen (top 15):")
for v, n in raw_counter.most_common(15):
    print(f"  {v!r:32s} -> {to_class(v):8s}  ({n})")

# --- nearest sidewalk per track point (spatial index over the whole city) ---
from shapely.strtree import STRtree
geoms = [ln for _, ln in ways]
tree = STRtree(geoms)
classes = []; dists = []
for pt in pts:
    P = Point(xy(*pt))
    idx = tree.nearest(P)                     # index of nearest sidewalk line
    ln = geoms[idx]; d = ln.distance(P)
    dists.append(d)
    classes.append(ways[idx][0] if d <= TOL_M else 'unknown')

import statistics as _st
dsort = sorted(dists)
print("\nDistance from each track point to the NEAREST city sidewalk (metres):")
print(f"  min {min(dists):.0f} | median {_st.median(dists):.0f} | "
      f"90th pct {dsort[int(0.9*len(dsort))-1]:.0f} | max {max(dists):.0f}")
within = sum(d <= TOL_M for d in dists)
print(f"  within TOL_M={TOL_M} m: {within}/{len(dists)} points")
if within == 0:
    print("  --> the walk never runs within {} m of a CITY sidewalk. This route is off the\n"
          "      PWD sidewalk grid (Northeastern campus / Fenway park paths), so this dataset\n"
          "      cannot classify it. Use the OSM export (includes campus/park paths) instead,\n"
          "      or raise TOL_M if you just want the nearest street's material.".format(TOL_M))

from collections import Counter
c = Counter(classes); tot = len(classes)
print("\nSurface profile along the walk (per GPS point):")
for k in ['paved','grass','loose','unknown']:
    print(f"  {k:8s} {c.get(k,0):4d}  ({100*c.get(k,0)/tot:4.0f}%)")
print("\n(Sidewalk inventories are built surfaces -> expect paved-heavy, little/no loose; "
      "raise TOL_M if too many 'unknown' where the track runs mid-road away from sidewalks.)")

feats_out = [{'type':'Feature','properties':{'surface': classes[i]},
              'geometry':{'type':'LineString','coordinates':[list(pts[i]), list(pts[i+1])]}}
             for i in range(len(pts)-1)]
json.dump({'type':'FeatureCollection','features':feats_out}, open('track_surface.geojson','w'))
print("\nwrote track_surface.geojson  ->  load it in your compare map.")
