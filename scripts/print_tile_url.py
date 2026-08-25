"""Print the MassGIS land-cover tile download URL(s) covering a walk.
    python print_tile_url.py REC_20260519_161001.gpx LANDCOVER_LANDUSE_POLY.shp
Then open the printed URL in a browser, download + unzip the tile, and run:
    python sample_surface_landcover.py REC_20260519_161001.gpx <tile>.shp
"""
import sys, gpxpy, geopandas as gpd
from shapely.geometry import box

GPX = sys.argv[1] if len(sys.argv) > 1 else 'REC_20260519_161001.gpx'
IDX = sys.argv[2] if len(sys.argv) > 2 else 'LANDCOVER_LANDUSE_POLY.shp'

g = gpxpy.parse(open(GPX))
P = [(p.longitude, p.latitude) for t in g.tracks for s in t.segments for p in s.points]
xs = [a for a, b in P]; ys = [b for a, b in P]
bb = box(min(xs)-0.004, min(ys)-0.004, max(xs)+0.004, max(ys)+0.004)

idx = gpd.read_file(IDX).to_crs("EPSG:4326")
hit = idx[idx.intersects(bb)]
print(f"{len(hit)} tile(s) cover the walk:")
for _, r in hit.iterrows():
    print("  tile:", r.get('tilename'), "\n   url:", r.get('shp_link'))
