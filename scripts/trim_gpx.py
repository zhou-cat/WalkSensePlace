"""Trim each GPX to the span its ground-truth tags cover.

    python scripts/trim_gpx.py GPS                 # report only, writes nothing
    python scripts/trim_gpx.py GPS --write          # write the trimmed GPX files
    python scripts/trim_gpx.py GPS --write --suffix ""   # overwrite in place (backs up)

A GPX normally starts recording before the first surface tag is placed and keeps running
after the last, so the track has fixes at both ends that no layer can label. Trimming them
HERE, once, means predict_walks.py, build_all_walks*.py and fill_gis_table.py all read the
same track and none of them needs to know about an offset.

The kept span is [first surface marker, last surface marker] from the walk's own
GPS/REC_<stamp>_GPS.csv, intersected with the insole recording's own time range. Fixes
inside the span are untouched, including those in recorded pauses: a pause is real data.

Original files are renamed to <name>.gpx.orig unless --suffix keeps them separate.
"""
import argparse
import glob
import os
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd

NS = 'http://www.topografix.com/GPX/1/1'
# tag columns that mark a SURFACE (transition/corrupt are not surfaces)
SURFACE_COLS = {'asphalt', 'concrete', 'grass', 'compact ground', 'dirt', 'gravel',
                'stone dust', 'sand', 'mulch', 'brick', 'paving stones'}
EXCLUDE_COLS = {'transition', 'wait', 'corrupt'}


def tag_span(csv_path):
    """The span the ground truth actually covers: (start, end, rec_start, rec_end).

    A marker marks where a segment BEGINS, not where it ends, so the tagged span runs from
    the first surface marker to the END OF THE RECORDING - the walker keeps walking on the
    last tagged surface until the recording stops. (Trimming at the last marker instead
    would cut off the whole final segment, which on our walks was 10-50% of the track.)

    The one exception is a recording that ends inside an excluded region: if the final
    marker of any kind is a transition or wait, the truth ends where that region opens.
    """
    d = pd.read_csv(csv_path, low_memory=False)
    ts = pd.to_numeric(d['timestamp'], errors='coerce')
    surf, excl = [], []
    for col in d.columns:
        key = col.strip().lower()
        if key not in SURFACE_COLS and key not in EXCLUDE_COLS:
            continue
        hit = d.index[d[col].astype(str).str.strip().str.lower() == 'x']
        tsv = [int(ts.loc[i]) for i in hit if pd.notna(ts.loc[i])]
        (surf if key in SURFACE_COLS else excl).extend(tsv)
    if not surf:
        return None
    c0, c1 = int(ts.min()), int(ts.max())
    start = min(surf)
    # ends at the recording's end, unless an excluded region opens after the last surface
    end = c1
    later = [t for t in excl if t > max(surf)]
    if later:
        end = min(later)
    return start, end, c0, c1


def read_gpx(path):
    ET.register_namespace('', NS)
    tree = ET.parse(path)
    pts = []
    for seg in tree.getroot().iter():
        if seg.tag.split('}')[-1] != 'trkseg':
            continue
        for pt in list(seg):
            if pt.tag.split('}')[-1] != 'trkpt':
                continue
            t = None
            for c in pt:
                if c.tag.split('}')[-1] == 'time':
                    t = (c.text or '').strip()
            tms = (int(datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp() * 1000)
                   if t else None)
            pts.append((seg, pt, tms))
    return tree, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('walks_dir')
    ap.add_argument('--write', action='store_true', help='actually write the trimmed files')
    ap.add_argument('--suffix', default='',
                    help="appended before .gpx. The DEFAULT is '' - overwrite in place, "
                         'keeping the original as .gpx.orig - because a suffix leaves two '
                         'GPX files per walk and the map scripts then treat the trimmed '
                         'one as a separate walk with no prediction layers.')
    ap.add_argument('--pad-s', type=float, default=0.0,
                    help='keep this many seconds either side of the tagged span')
    a = ap.parse_args()

    tot_before = tot_after = 0
    _dupe_warn = [False]
    for gpx in sorted(glob.glob(os.path.join(a.walks_dir, '*.gpx'))):
        if a.suffix and gpx.endswith(a.suffix + '.gpx'):
            continue                                   # do not re-trim our own output
        stem = os.path.splitext(os.path.basename(gpx))[0]
        csvs = [os.path.join(a.walks_dir, stem + s) for s in ('_GPS.csv', '.csv')]
        csv_path = next((c for c in csvs if os.path.exists(c)), None)
        if not csv_path:
            print(f'{stem}: no walk CSV, skipped'); continue
        span = tag_span(csv_path)
        if not span:
            print(f'{stem}: no surface markers in {os.path.basename(csv_path)}, skipped')
            continue
        t0, t1, c0, c1 = span
        lo = max(t0, c0) - int(a.pad_s * 1000)
        hi = min(t1, c1) + int(a.pad_s * 1000)
        tree, pts = read_gpx(gpx)
        timed = [p for p in pts if p[2] is not None]
        if not timed:
            print(f'{stem}: GPX has no timestamps, skipped'); continue
        g0, g1 = timed[0][2], timed[-1][2]
        if max(0, min(g1, c1) - max(g0, c0)) <= 0:
            f = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime('%H:%M:%S')
            print(f'{stem}: [WARN] GPX {f(g0)}-{f(g1)} does not overlap '
                  f'{os.path.basename(csv_path)} {f(c0)}-{f(c1)} - MISMATCHED PAIR, skipped')
            continue
        drop = [p for p in pts if p[2] is None or p[2] < lo or p[2] > hi]
        keep = len(pts) - len(drop)
        tot_before += len(pts); tot_after += keep
        head = sum(1 for p in timed if p[2] < lo)
        tail = sum(1 for p in timed if p[2] > hi)
        if not drop:
            print(f'{stem}: {len(pts)} fixes, nothing to trim')
            continue
        print(f'{stem}: {len(pts)} -> {keep} fixes  (-{head} before the first tag, '
              f'-{tail} after the last)')
        if not a.write:
            continue
        if keep < 2:
            print('   [WARN] fewer than 2 fixes would remain - not written'); continue
        for seg, pt, _ in drop:
            seg.remove(pt)
        out = gpx if not a.suffix else os.path.join(
            a.walks_dir, stem + a.suffix + '.gpx')
        if out == gpx:
            shutil.copy2(gpx, gpx + '.orig')
        tree.write(out, encoding='utf-8', xml_declaration=True)
        print(f'   -> {os.path.basename(out)}'
              + ('  (original kept as .gpx.orig)' if out == gpx else ''))
        if out != gpx:
            _dupe_warn[0] = True

    if tot_before:
        print(f'\ntotal {tot_before} -> {tot_after} fixes '
              f'({100*(tot_before-tot_after)/tot_before:.1f}% trimmed)')
    if a.write and _dupe_warn[0]:
        print(f"\n[WARN] the trimmed files carry the suffix '{a.suffix}', so GPS/ now holds "
              'TWO GPX files per walk. The map scripts glob *.gpx and will treat the '
              'trimmed ones as separate walks with no prediction, evenness or hardness '
              'layers. Either delete the originals and rename, or rerun with --suffix "".')
    if a.write:
        print('\nNEXT: the geojsons in walk_maps/ were written for the UNTRIMMED track.\n'
              '  1. delete them:  del GPS\\walk_maps\\*.geojson\n'
              '  2. rerun predict_walks.py\n'
              '  3. rerun build_all_walks_satellite.py with --no-trim (the GPX is already '
              'trimmed)')
    if not a.write:
        print('report only - rerun with --write to apply')


if __name__ == '__main__':
    main()
