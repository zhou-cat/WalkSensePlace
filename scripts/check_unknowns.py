"""Where are the 'unknown' segments in each layer, and do the layers agree?"""
import glob, json, os, sys

gis_dir = sys.argv[1] if len(sys.argv) > 1 else 'GPS/walk_maps_satellite'
pred_dir = sys.argv[2] if len(sys.argv) > 2 else 'GPS/walk_maps'

def vals(path, key):
    if not os.path.exists(path):
        return None
    return [(f['properties'] or {}).get(key) or 'unknown'
            for f in json.load(open(path, encoding='utf-8'))['features']]

tot = {'truth': [0, 0], 'gis': [0, 0], 'model': [0, 0], 'shared': 0}
for p in sorted(glob.glob(os.path.join(gis_dir, '*_tagged.geojson'))):
    stamp = os.path.basename(p)[:-len('_tagged.geojson')]
    t = vals(p, 'tag')
    g = vals(os.path.join(gis_dir, stamp + '_surface.geojson'), 'surface')
    m = vals(os.path.join(pred_dir, stamp + '_pred_3class.geojson'), 'surface')
    if not (t and g and m):
        print(f'{stamp}: missing a layer, skipped'); continue
    n = min(len(t), len(g), len(m))
    U = lambda v: {i for i, x in enumerate(v[:n]) if x == 'unknown'}
    ut, ug, um = U(t), U(g), U(m)
    both = um & ug
    print(f'{stamp}: {n} segs | unknown: truth {len(ut)}, GIS {len(ug)}, model {len(um)} '
          f'| model-unknown that is also GIS-unknown: {len(both)}/{len(um) or 1}')
    for k, u in (('truth', ut), ('gis', ug), ('model', um)):
        tot[k][0] += len(u); tot[k][1] += n
    tot['shared'] += len(both)

print('\npooled:')
for k in ('truth', 'gis', 'model'):
    a, b = tot[k]
    print(f'  {k:6s} {a:5d} / {b} = {100*a/max(b,1):.2f}% unknown')
print(f'  of the model\'s unknowns, {tot["shared"]} are also GIS unknowns')