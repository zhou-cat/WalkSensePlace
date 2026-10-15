"""
sample_surface_landcover.py — sample the MassGIS 2016 Land Cover/Land Use POLYGON
shapefile along a GPX track. Reads the shapefile directly (CRS handled automatically).

Use the SHAPEFILE download (not the CSV — the CSV has no geometry). Unzip it so you have
LANDCOVER_LANDUSE_POLY.shp (plus its .dbf/.shx/.prj) in a folder.

Run:
    pip install geopandas pyogrio gpxpy      # pyogrio ships GDAL; do NOT pip install fiona
    python sample_surface_landcover.py activity_22645980458.gpx  LANDCOVER_LANDUSE_POLY.shp

Writes track_surface.geojson (per-segment 'surface') for your compare map, and prints the
raw land-cover values it saw + how each mapped so you can verify/adjust the mapping.

Note: the statewide file is large, so this reads ONLY the polygons near your walk (bbox
pre-filter) — fast even on the full download.
"""
import sys, json
import gpxpy
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

GPX = sys.argv[1] if len(sys.argv) > 1 else 'activity_22645980458.gpx'
SHP = sys.argv[2] if len(sys.argv) > 2 else 'LANDCOVER_LANDUSE_POLY.shp'
PAD_DEG = 0.004     # ~450 m bbox padding around the walk

# candidate land-cover / land-use field names in the MassGIS schema (auto-detected)
LC_FIELDS = ['COVERNAME','COVERTYPE','COVER_DESC','LC_DESC','LANDCOVER','COVERCODE',
             'LCLU','LU_DESC','USEGENDESC','LANDUSE','LU','DESCRIPT','CLASS','LCTYPE']

def to_class(v):
    """Map a land-cover/land-use description to grass / paved / loose. Order matters:
    'developed open space' is grassy, so open-space is checked before 'developed'."""
    s = str(v).strip().lower()
    if not s or s in ('none','nan','null'): return 'unknown'
    if any(k in s for k in ['open space','grass','pasture','cultivat','agric','recreation','golf','lawn','cemeter']):
        return 'grass'
    if any(k in s for k in ['impervious','pavement','paved','road','building','commercial','industrial',
                            'transportation','parking','high density','residential']):
        return 'paved'
    if any(k in s for k in ['bare','soil','sand','barren','quarr','excavat']):
        return 'loose'
    # forest / tree / shrub / water / wetland -> not a walking-surface class we model
    return 'unknown'

# ---- track ----
with open(GPX) as f:
    g = gpxpy.parse(f)
pts = [(p.longitude, p.latitude) for tr in g.tracks for seg in tr.segments for p in seg.points]
if not pts: sys.exit("No track points in GPX.")
print(f"{len(pts)} track points")
minx = min(p[0] for p in pts) - PAD_DEG; maxx = max(p[0] for p in pts) + PAD_DEG
miny = min(p[1] for p in pts) - PAD_DEG; maxy = max(p[1] for p in pts) + PAD_DEG

# ---- read ONLY polygons near the walk (bbox in the shapefile's own CRS) ----
def read_near(shp_path):
    """Read only the polygons near the walk from any shapefile (CRS handled), return lat/lon."""
    crs = gpd.read_file(shp_path, rows=1).crs
    bnative = tuple(gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)],
                                     crs="EPSG:4326").to_crs(crs).total_bounds)
    return gpd.read_file(shp_path, bbox=bnative).to_crs("EPSG:4326")

print(f"reading {SHP} near the walk ...")
gdf = read_near(SHP)
if len(gdf) == 0:
    sys.exit("No polygons found near the track — check that the shapefile covers this area.")

# ---- If the given file has no land-cover field, look for a sibling shapefile that does ----
if not any(c in gdf.columns for c in LC_FIELDS):
    import glob as _g
    here = os.path.dirname(os.path.abspath(SHP)) or '.'
    for cand in _g.glob(os.path.join(here, '**', '*.shp'), recursive=True):
        if os.path.abspath(cand) == os.path.abspath(SHP): continue
        try: ccols = list(gpd.read_file(cand, rows=1).columns)
        except Exception: continue
        if any(c in ccols for c in LC_FIELDS):
            print(f"found land-cover data in sibling shapefile: {cand}")
            gdf = read_near(cand); break

# ---- If this is the MassGIS TILE INDEX (columns tilename/shp_link), fetch the real tile data ----
if 'shp_link' in gdf.columns and not any(c in gdf.columns for c in LC_FIELDS):
    links = [u for u in gdf['shp_link'].dropna().unique().tolist() if u]
    print(f"\nThis file is the LCLU TILE INDEX, not the data. {len(links)} tile(s) cover your walk:")
    for u in links: print("   ", u)
    import urllib.request, zipfile, io, glob as _glob
    tmp = os.path.join(os.path.dirname(os.path.abspath(SHP)) or '.', '_lclu_tiles')
    os.makedirs(tmp, exist_ok=True)
    tiles = []
    for u in links:
        try:
            print(f"  downloading {u} ...")
            data = urllib.request.urlopen(u, timeout=90).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z: z.extractall(tmp)
        except Exception as ex:
            print(f"    download failed: {ex}")
    shps = _glob.glob(os.path.join(tmp, '**', '*.shp'), recursive=True)
    for sp in shps:
        try: tiles.append(read_near(sp))
        except Exception: pass
    if not tiles:
        sys.exit("\nCould not auto-download the tile data (network may block it).\n"
                 "Open the shp_link URL(s) above in your browser, download + unzip the tile,\n"
                 "then re-run pointing at the tile's own .shp, e.g.:\n"
                 f"    python {os.path.basename(sys.argv[0])} {os.path.basename(GPX)} <downloaded_tile>.shp")
    gdf = pd.concat(tiles, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs="EPSG:4326")
    print(f"loaded {len(gdf)} land-cover polygons from {len(tiles)} tile(s)")

print(f"{len(gdf)} land-cover polygons near the walk")

# ---- detect the land-cover field ----
field = next((c for c in LC_FIELDS if c in gdf.columns), None)
if field is None:
    # fall back to the text column with the most distinct short values
    best, nbest = None, -1
    for c in gdf.columns:
        if c == 'geometry': continue
        vals = gdf[c].astype(str)
        if vals.str.len().mean() > 40: continue
        u = vals.str.lower().nunique()
        if 1 < u < 60 and u > nbest and not vals.str.fullmatch(r'-?\d+(\.\d+)?').all():
            best, nbest = c, u
    field = best
print(f"using land-cover field: {field!r}   (columns: {list(gdf.columns)[:12]}...)")
if field is None:
    sys.exit("Couldn't find a land-cover field — paste me the column names and I'll set it.")

# ---- spatial join: which polygon each GPS point falls in ----
gpts = gpd.GeoDataFrame({'pid': range(len(pts))},
                        geometry=[Point(xy) for xy in pts], crs="EPSG:4326")
joined = gpd.sjoin(gpts, gdf[[field, 'geometry']], how='left', predicate='within')
joined = joined.sort_values('pid').drop_duplicates('pid')           # one polygon per point
raw = joined[field].reindex(range(len(pts))).values
classes = [to_class(v) for v in raw]

# ---- report ----
import collections
prof = collections.Counter(classes); tot = len(classes)
print("\nSurface profile along the walk (per GPS point):")
for k in ['paved','grass','loose','unknown']:
    print(f"  {k:8s} {prof.get(k,0):4d}  ({100*prof.get(k,0)/tot:4.0f}%)")
print("\nraw land-cover values seen -> mapping:")
for v, n in collections.Counter(str(x) for x in raw).most_common(15):
    print(f"  {v[:34]:34s} -> {to_class(v):8s} ({n})")

# ---- write per-segment GeoJSON ----
feats = [{'type':'Feature','properties':{'surface': classes[i]},
          'geometry':{'type':'LineString','coordinates':[list(pts[i]), list(pts[i+1])]}}
         for i in range(len(pts)-1)]
json.dump({'type':'FeatureCollection','features':feats}, open('track_surface.geojson','w'))
print("\nwrote track_surface.geojson -> load it in your compare map.")
