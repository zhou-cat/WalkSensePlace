"""Which loaded GIS layers actually cover each walk?

    python scripts/check_gis_coverage.py GPS --landcover gis/LCLU_R07C19.shp

For every GPX it prints the walk's bounding box and, for each layer, whether that layer's
own extent contains it. A walk outside a layer's extent gets no label from it - that is a
coverage gap, not a classification failure, and no change of reach or priority will fix it.

Needs geopandas (same as the map scripts).
"""
import argparse
import glob
import os
import xml.etree.ElementTree as ET


def gpx_bbox(path):
    lat, lon = [], []
    for pt in ET.parse(path).getroot().iter():
        if pt.tag.split('}')[-1] != 'trkpt':
            continue
        lat.append(float(pt.attrib['lat'])); lon.append(float(pt.attrib['lon']))
    return (min(lon), min(lat), max(lon), max(lat)) if lat else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('walks_dir')
    ap.add_argument('--landcover', default=None)
    ap.add_argument('--sidewalk', default='sidewalk_inventory.geojson')
    ap.add_argument('--openspace', default='open_space.geojson')
    ap.add_argument('--osm', default='export.geojson')
    a = ap.parse_args()

    import geopandas as gpd
    from shapely.geometry import box

    def extent(path, label):
        """A layer's extent in WGS84, or None if it is missing."""
        if not path:
            return None
        if not os.path.exists(path):
            for d in (a.walks_dir, os.path.dirname(a.landcover or ''), '.'):
                c = os.path.join(d, os.path.basename(path)) if d else None
                if c and os.path.exists(c):
                    path = c
                    break
            else:
                print(f'  {label:18s} NOT FOUND ({path})')
                return None
        try:
            g = gpd.read_file(path)
            if g.crs is not None and str(g.crs).upper() not in ('EPSG:4326', 'OGC:CRS84'):
                g = g.to_crs('EPSG:4326')
            b = g.total_bounds
            print(f'  {label:18s} {len(g):7d} features   '
                  f'lon {b[0]:.4f}..{b[2]:.4f}  lat {b[1]:.4f}..{b[3]:.4f}')
            return box(*b)
        except Exception as e:
            print(f'  {label:18s} could not read: {e}')
            return None

    print('loaded layers and their extents:')
    layers = {'land cover': extent(a.landcover, 'land cover'),
              'sidewalk': extent(a.sidewalk, 'sidewalk'),
              'open space': extent(a.openspace, 'open space'),
              'OSM': extent(a.osm, 'OSM')}

    print('\nwalk coverage:')
    bad = []
    for gpx in sorted(glob.glob(os.path.join(a.walks_dir, '*.gpx'))):
        stem = os.path.splitext(os.path.basename(gpx))[0]
        bb = gpx_bbox(gpx)
        if bb is None:
            print(f'  {stem}: no track points'); continue
        w = box(*bb)
        hits = []
        for name, ext in layers.items():
            if ext is None:
                hits.append(f'{name}:--')
            elif ext.contains(w):
                hits.append(f'{name}:yes')
            elif ext.intersects(w):
                hits.append(f'{name}:PARTIAL')
            else:
                hits.append(f'{name}:NO')
        covered = [h for h in hits if h.endswith(':yes')]
        flag = '' if covered else '   <-- NO layer covers this walk'
        if not covered:
            bad.append(stem)
        print(f'  {stem}  ({bb[1]:.4f}, {bb[0]:.4f})  ' + '  '.join(hits) + flag)

    if bad:
        print(f'\n{len(bad)} walk(s) outside every loaded layer: {bad}')
        print('  Land cover is tiled 10 km x 10 km (TILENAME RxxCyy), so a walk a few km '
              'from the others can need a different tile.')
        print('  The sidewalk inventory and open space layers are CITY OF BOSTON only, so '
              'they cover nothing outside the city limits whatever tile you load.')


if __name__ == '__main__':
    main()
