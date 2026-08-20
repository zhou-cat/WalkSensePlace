"""
OFFLINE surface sampler — no network at all.
Reads a LOCAL OSM surface GeoJSON (exported from overpass-turbo in your browser)
and matches it to your GPX track.

1) In a browser go to https://overpass-turbo.eu , zoom to your walk area, run:
       way({{bbox}})["surface"];
       out geom;
   then Export -> download as GeoJSON  ->  save as  osm_surface.geojson

2) Run:
       pip install shapely gpxpy          # (already installed for you)
       python sample_surface_LOCAL.py activity_22645980458.gpx osm_surface.geojson

Outputs track_surface.geojson + a printed surface profile. Zero network calls.
"""
import sys, json, math
import gpxpy
from shapely.geometry import LineString, Point, shape

GPX      = sys.argv[1] if len(sys.argv) > 1 else 'activity_22645980458.gpx'
OSM_JSON = sys.argv[2] if len(sys.argv) > 2 else 'osm_surface.geojson'
TOL_M    = 12    # nearest-way tolerance (metres); raise to ~20 if too many 'unknown'

PAVED = {'asphalt','concrete','concrete:plates','concrete:lanes','paved','paving_stones',
         'sett','cobblestone','unhewn_cobblestone','bricks','brick','metal','wood'}
GRASS = {'grass','grass_paver'}
LOOSE = {'unpaved','dirt','earth','ground','gravel','fine_gravel','compacted','pebblestone',
         'sand','mud','woodchips'}
def to_class(s):
    if not s: return 'unknown'
    s = s.lower()
    return 'paved' if s in PAVED else 'grass' if s in GRASS else 'loose' if s in LOOSE else 'unknown'

# --- track ---
with open(GPX) as f:
    g = gpxpy.parse(f)
pts = [(p.longitude, p.latitude) for tr in g.tracks for seg in tr.segments for p in seg.points]
print(f"{len(pts)} track points")
clat = sum(p[1] for p in pts)/len(pts); clon = sum(p[0] for p in pts)/len(pts)
def xy(lon, lat):
    return ((lon-clon)*math.cos(math.radians(clat))*111320.0, (lat-clat)*110540.0)

# --- local OSM surface ways ---
try:
    gj = json.load(open(OSM_JSON, encoding='utf-8'))
except FileNotFoundError:
    sys.exit(f"'{OSM_JSON}' not found. Export it from overpass-turbo first (see header).")
ways = []
for feat in gj.get('features', []):
    surf = (feat.get('properties') or {}).get('surface')
    if not surf: continue
    geom = shape(feat['geometry'])
    lines = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
    for ln in lines:
        if ln.geom_type != 'LineString': continue
        ways.append((to_class(surf), LineString([xy(x, y) for x, y in ln.coords])))
print(f"{len(ways)} surface-tagged ways loaded from {OSM_JSON}")
if not ways:
    sys.exit("No 'surface' features in the file — re-run the overpass-turbo query and export again.")

# --- nearest surface way per point ---
classes = []
for pt in pts:
    P = Point(xy(*pt)); bc, bd = 'unknown', 1e9
    for c, ln in ways:
        d = ln.distance(P)
        if d < bd: bd, bc = d, c
    classes.append(bc if bd <= TOL_M else 'unknown')

from collections import Counter
c = Counter(classes); tot = len(classes)
print("\nSurface profile along the walk (per GPS point):")
for k in ['paved','grass','loose','unknown']:
    print(f"  {k:8s} {c.get(k,0):4d}  ({100*c.get(k,0)/tot:4.0f}%)")
print("\n(High 'unknown' = OSM has no surface tag on those sidewalks; raise TOL_M or accept the gap.)")

feats = [{'type':'Feature','properties':{'surface': classes[i]},
          'geometry':{'type':'LineString','coordinates':[list(pts[i]), list(pts[i+1])]}}
         for i in range(len(pts)-1)]
json.dump({'type':'FeatureCollection','features':feats}, open('track_surface.geojson','w'))
print("\nwrote track_surface.geojson")
