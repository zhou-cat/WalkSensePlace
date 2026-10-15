"""Check the sidewalk and open-space files the walk-map scripts expect.

    python scripts/check_gis_inputs.py GPS

Reports, for each file: whether the exact filename exists (and any near-misses, such as a
hidden double extension from renaming in Windows Explorer), whether it parses, how many
features and of what geometry, and - for the sidewalk inventory - whether the material
field the classifier needs is present and what the features would be classified as.
Reads the GeoJSON directly, so it runs even where geopandas is not installed.
"""
import collections
import glob
import json
import os
import sys

WALKS = sys.argv[1] if len(sys.argv) > 1 else 'GPS'
EXPECTED = {
    'sidewalk_inventory.geojson': 'sidewalk',
    'open_space.geojson': 'open_space',
}
SW_FIELDS = ['MATERIAL', 'material', 'Material', 'surface', 'SURFACE']   # as build_all_walks.py
OLD = {'sidewalk': 24727, 'open_space': 577}     # feature counts of the previous versions

# the classifier itself, if build_all_walks.py is importable; else counts raw values only
sw_class = None
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_all_walks as _b
    sw_class = _b.sw_class
except Exception as e:
    print(f'(build_all_walks.py not importable here: {e}; classes will not be checked)\n')


def near_misses(name):
    stem = name.split('.')[0].lower()
    key = stem.split('_')[0]                       # 'sidewalk' / 'open'
    out = []
    for p in glob.glob(os.path.join(WALKS, '*')):
        b = os.path.basename(p)
        if key in b.lower() and b != name:
            out.append(b)
    return out


ok_all = True
for name, kind in EXPECTED.items():
    path = os.path.join(WALKS, name)
    print(f'=== {path}')
    if not os.path.exists(path):
        ok_all = False
        print('  NOT FOUND under that exact name.')
        nm = near_misses(name)
        if nm:
            print('  Similar files in the folder (repr shows hidden characters/extensions):')
            for b in nm:
                print(f'    {b!r}   ({os.path.getsize(os.path.join(WALKS, b))/1e6:.1f} MB)')
            print(f'  -> rename the right one to exactly {name!r}. In Explorer, turn on '
                  'View > File name extensions first, or the rename can add a second '
                  '".geojson".')
        else:
            print('  No similarly named file here either.')
        print()
        continue
    size = os.path.getsize(path) / 1e6
    try:
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
    except Exception as e:
        ok_all = False
        print(f'  exists ({size:.1f} MB) but does NOT parse as JSON: {e}')
        print('  -> re-download it as GeoJSON (not CSV, KML or a zipped shapefile)\n')
        continue
    feats = d.get('features') or []
    if d.get('type') != 'FeatureCollection' or not feats:
        ok_all = False
        print(f'  parses, but is not a GeoJSON FeatureCollection with features '
              f'(type={d.get("type")!r}, {len(feats)} features)\n')
        continue
    geom = collections.Counter((f.get('geometry') or {}).get('type') for f in feats)
    crs = (d.get('crs') or {}).get('properties', {}).get('name', 'none given (WGS84 assumed)')
    first = feats[0].get('geometry') or {}
    c = first.get('coordinates')
    while isinstance(c, list) and c and isinstance(c[0], list):
        c = c[0]
    print(f'  OK: {len(feats):,} features ({size:.1f} MB), geometry {dict(geom)}, CRS {crs}')
    print(f'      first coordinate {c}  <- should look like [-71.x, 42.x] for Boston in WGS84')
    if OLD.get(kind):
        print(f'      previous version had {OLD[kind]:,} features')
    if not set(geom) <= {'Polygon', 'MultiPolygon'}:
        print('  [!] expected polygons; other geometry types will not behave as before')
    if kind == 'sidewalk':
        cols = set(feats[0].get('properties', {}))
        fld = next((k for k in SW_FIELDS if k in cols), None)
        if not fld:
            ok_all = False
            print(f'  [!] NO MATERIAL FIELD. Looked for {SW_FIELDS}; the file has: '
                  f'{sorted(cols)}')
            print('      The layer would load, but every sidewalk would be "unknown". Tell '
                  'me the new field name and its values and I will map them.')
        else:
            vals = collections.Counter(str(f['properties'].get(fld)).strip() for f in feats)
            print(f'      material field: {fld!r}; most common values: {vals.most_common(8)}')
            if sw_class:
                cls = collections.Counter(sw_class(f['properties'].get(fld)) for f in feats)
                tot = sum(cls.values())
                print('      classified as: ' + ', '.join(
                    f'{k} {v:,} ({100*v/tot:.0f}%)' for k, v in cls.most_common()))
                if cls.get('unknown', 0) / tot > 0.5:
                    ok_all = False
                    print('  [!] over half unknown - the material codes may have changed')
    print()

print('ALL GOOD - rebuild the maps.' if ok_all else
      'Fix the items marked above, then rebuild the maps.')
