"""Surface composition of every walk, as one CSV per walk plus a combined one.

    python scripts/make_map_prop.py GPS --gis-dir GPS/walk_maps_satellite

Writes, into --out (default the GIS folder):
    map_prop_long_<stamp>.csv   Surface, Source, Share   one row per surface x source
    map_prop_long_all.csv       the same with a Walk column, for plotting all walks at once
    map_prop_wide.csv           surfaces as rows, sources as columns, one block per walk

Shares are computed over the SEGMENTS of each walk's geojsons, which is what the maps draw
and what the earlier hand-made CSVs counted, so the numbers are comparable.

The three sources:
    Ground truth      <stamp>_tagged.geojson    the walk file's own surface tags
    GIS               <stamp>_surface.geojson   the public layers, at the reach in summary.csv
    Prediction model  <stamp>_pred_3class.geojson
All three come from the map output, so whatever model and reach produced those files is
what this reports - run predict_walks.py and build_all_walks_satellite.py first.
"""
import argparse
import csv
import glob
import json
import os
import sys

SURFACES = ['grass', 'loose', 'paved', 'unknown']
SOURCES = [('Ground truth', '_tagged.geojson', 'tag'),
           ('GIS', '_surface.geojson', 'surface'),
           ('Prediction model', '_pred_3class.geojson', 'surface')]


def shares(path, key, keep_unknown):
    """Share of this layer's segments on each surface, or None if the file is missing."""
    if not os.path.exists(path):
        return None
    feats = json.load(open(path, encoding='utf-8')).get('features', [])
    counts = {}
    for f in feats:
        v = (f.get('properties') or {}).get(key)
        v = str(v).strip().lower() if v is not None else 'unknown'
        if v in ('', 'none', 'nan', '__exclude__'):
            v = 'unknown'
        counts[v] = counts.get(v, 0) + 1
    if not keep_unknown:
        counts.pop('unknown', None)
    tot = sum(counts.values())
    if not tot:
        return None
    return {s: counts.get(s, 0) / tot for s in SURFACES}, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('walks_dir')
    ap.add_argument('--gis-dir', default=None,
                    help='folder with the walk geojsons (default <walks_dir>/walk_maps_satellite)')
    ap.add_argument('--pred-dir', default=None,
                    help='folder with *_pred_3class.geojson, if not --gis-dir '
                         '(default: try --gis-dir, then <walks_dir>/walk_maps)')
    ap.add_argument('--out', default=None, help='output folder (default --gis-dir)')
    ap.add_argument('--keep-unknown', action='store_true',
                    help="keep the 'unknown' class in the shares. GIS leaves some steps "
                         "unclassified, so dropping it renormalises the other three and "
                         "hides a real GIS weakness; the earlier hand-made CSVs dropped it.")
    ap.add_argument('--decimals', type=int, default=4)
    a = ap.parse_args()
    a.gis_dir = a.gis_dir or os.path.join(a.walks_dir, 'walk_maps_satellite')
    a.out = a.out or a.gis_dir
    pred_dirs = [d for d in (a.pred_dir, a.gis_dir,
                             os.path.join(a.walks_dir, 'walk_maps')) if d]

    summ = os.path.join(a.gis_dir, 'summary.csv')
    if os.path.exists(summ):
        head = open(summ, encoding='utf-8').readline().strip().split(',')
        rc = [c for c in head if c.startswith('reach_')]
        print('GIS layers built with: ' + (', '.join(rc) if rc else
              'reach NOT recorded - built before the shared reach; rebuild first'))
    else:
        print(f'[warning] no summary.csv in {a.gis_dir}: cannot confirm how the GIS '
              'layers were built')

    stamps = sorted({os.path.basename(p)[:-len('_tagged.geojson')]
                     for p in glob.glob(os.path.join(a.gis_dir, '*_tagged.geojson'))})
    if not stamps:
        sys.exit(f'no *_tagged.geojson in {a.gis_dir} - run build_all_walks_satellite.py first')

    os.makedirs(a.out, exist_ok=True)
    all_rows, wide = [], []
    for stamp in stamps:
        got, note = {}, []
        for name, suffix, key in SOURCES:
            dirs = pred_dirs if 'pred' in suffix else [a.gis_dir]
            res = None
            for d in dirs:
                res = shares(os.path.join(d, stamp + suffix), key, a.keep_unknown)
                if res:
                    break
            if res is None:
                note.append(f'no {name}')
                continue
            got[name] = res
        if not got:
            print(f'[skip] {stamp}: no layers found'); continue
        rows = []
        for name, _s, _k in SOURCES:
            if name not in got:
                continue
            sh, tot = got[name]
            for s in SURFACES:
                if s == 'unknown' and not a.keep_unknown:
                    continue
                rows.append(dict(Surface=s, Source=name, Share=round(sh[s], a.decimals)))
            all_rows += [dict(Walk=stamp, **r) for r in rows[-len(SURFACES) if a.keep_unknown
                                                            else -3:]]
            wide.append(dict(Walk=stamp, Source=name, Segments=tot,
                             **{s: round(sh[s], a.decimals) for s in SURFACES}))
        out = os.path.join(a.out, f'map_prop_long_{stamp[-6:]}.csv')
        with open(out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['Surface', 'Source', 'Share'])
            w.writeheader(); w.writerows(rows)
        counts = ' '.join(f'{n.split()[0][:5]}={got[n][1]}' for n in got)
        print(f'  {stamp}: {len(rows)} rows  [{counts}]'
              + (f'   ({", ".join(note)})' if note else ''))

    if all_rows:
        with open(os.path.join(a.out, 'map_prop_long_all.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['Walk', 'Surface', 'Source', 'Share'])
            w.writeheader(); w.writerows(all_rows)
    if wide:
        with open(os.path.join(a.out, 'map_prop_wide.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['Walk', 'Source', 'Segments'] + SURFACES)
            w.writeheader(); w.writerows(wide)
    print(f'\n{len(stamps)} walk(s) -> {a.out}/map_prop_long_*.csv, map_prop_long_all.csv, '
          'map_prop_wide.csv')
    print('plot with: python scripts/make_paper_figures.py --out figures --comp-dir '
          + a.out)


if __name__ == '__main__':
    main()
