"""How far does each walk's GPS track sit from the mapped paths?

    python scripts/gps_path_offset.py GPS --osm gis/export.geojson
    python scripts/gps_path_offset.py GPS --osm gis/export.geojson --plots

For every GPX in the folder: the distance from each fix to the nearest mapped footway /
path / cycleway in the OSM export, summarized per walk, plus a per-leg drift test. A leg
whose shape matches a mapped path but sits a few metres off it (small residual after a
best-fit shift) is GPS displacement; a leg no shift can fit may be off the mapped network.

Writes gps_path_offset.csv (one row per walk) and, with --plots, one PNG per walk.
Pure Python + matplotlib: no geopandas needed. Distances use a local flat projection,
which is accurate to well under 1% over a walk-sized area.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st

FOOT = {'footway', 'path', 'pedestrian', 'cycleway', 'track', 'steps', 'bridleway'}


def read_gpx(p):
    s = open(p, encoding='utf-8').read()
    return [(float(a), float(b)) for a, b in
            re.findall(r'<trkpt[^>]*lat="([^"]+)"[^>]*lon="([^"]+)"', s)] or \
           [(float(b), float(a)) for a, b in
            re.findall(r'<trkpt[^>]*lon="([^"]+)"[^>]*lat="([^"]+)"', s)]


def load_paths(path):
    d = json.load(open(path, encoding='utf-8'))
    ways = []
    for f in d['features']:
        g = f.get('geometry') or {}
        if f['properties'].get('highway') not in FOOT:
            continue
        lines = ([g['coordinates']] if g.get('type') == 'LineString'
                 else g['coordinates'] if g.get('type') == 'MultiLineString' else [])
        for c in lines:
            ways.append((c, f['properties'].get('surface') or ''))
    return ways


def project(lat0, lon0):
    mx = 111320 * math.cos(math.radians(lat0))
    return lambda lat, lon: ((lon - lon0) * mx, (lat - lat0) * 111320)


def seg_dist(p, a, b):
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def nearest(p, segs, shift=(0, 0)):
    q = (p[0] + shift[0], p[1] + shift[1])
    return min((seg_dist(q, a, b) for a, b in segs), default=float('inf'))


def leg_test(xy, segs, n=40, R=10.0, step=1.0):
    """Per n-fix leg: median distance as recorded, best-shift residual, and that shift."""
    out = []
    for i in range(0, len(xy), n):
        idx = range(i, min(i + n, len(xy)))
        med0 = st.median(nearest(xy[k], segs) for k in idx)
        best = (med0, 0.0, 0.0)
        r = int(R / step)
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                sh = (a * step, b * step)
                m = st.median(nearest(xy[k], segs, sh) for k in idx)
                if m < best[0]:
                    best = (m, sh[0], sh[1])
        out.append((i, idx[-1], med0) + best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('walks_dir')
    ap.add_argument('--osm', required=True, help='Overpass export (export.geojson)')
    ap.add_argument('--plots', action='store_true')
    ap.add_argument('--leg', type=int, default=40, help='fixes per leg for the drift test')
    a = ap.parse_args()

    ways = load_paths(a.osm)
    rows = []
    for gpx in sorted(glob.glob(os.path.join(a.walks_dir, '*.gpx'))):
        ll = read_gpx(gpx)
        if len(ll) < 10:
            continue
        to = project(*ll[0])
        xy = [to(*p) for p in ll]
        xs, ys = [p[0] for p in xy], [p[1] for p in xy]
        segs = []
        for c, _ in ways:
            P = [to(lat, lon) for lon, lat in c]
            if any(min(xs) - 60 < x < max(xs) + 60 and min(ys) - 60 < y < max(ys) + 60 for x, y in P):
                segs += list(zip(P[:-1], P[1:]))
        if not segs:
            print(f'{os.path.basename(gpx)}: no mapped paths nearby, skipped')
            continue
        d = [nearest(p, segs) for p in xy]
        q = lambda v, f: sorted(v)[int(f * (len(v) - 1))]
        legs = leg_test(xy, segs, n=a.leg)
        fit = [l for l in legs if l[3] <= 1.5]           # legs a shift puts on a path
        unfit = [l for l in legs if l[3] > 1.5]
        sh = [(l[4], l[5]) for l in fit]
        drift = (st.mean(math.hypot(x - st.mean(s[0] for s in sh), y - st.mean(s[1] for s in sh))
                         for x, y in sh) if len(sh) > 1 else 0.0)
        row = dict(walk=os.path.basename(gpx), fixes=len(xy),
                   median_m=round(st.median(d), 1), p75_m=round(q(d, .75), 1),
                   p90_m=round(q(d, .9), 1), max_m=round(max(d), 1),
                   pct_over_5m=round(100 * sum(x > 5 for x in d) / len(d)),
                   pct_over_10m=round(100 * sum(x > 10 for x in d) / len(d)),
                   legs=len(legs), legs_path_shaped=len(fit),
                   shift_drift_m=round(drift, 1),
                   unfit_legs=';'.join(f'{l[0]}-{l[1]}' for l in unfit))
        rows.append(row)
        print(f"{row['walk']:28s} median {row['median_m']:4.1f} m | >5 m {row['pct_over_5m']:3d}% "
              f"| >10 m {row['pct_over_10m']:3d}% | {row['legs_path_shaped']}/{row['legs']} legs "
              f"path-shaped, shifts drift {row['shift_drift_m']} m"
              + (f" | no path fits fixes {row['unfit_legs']}" if unfit else ''))
        if a.plots:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.add_collection(LineCollection(segs, colors='#9aa7b8', linewidths=2.2, zorder=1))
            sc = ax.scatter(xs, ys, c=d, cmap='plasma', vmin=0, vmax=12, s=12, zorder=3)
            for i in range(0, len(xy), a.leg):
                ax.annotate(str(i), xy[i], textcoords='offset points', xytext=(6, 4), fontsize=8)
            ax.plot(*xy[0], 'o', ms=10, mfc='#2E7D32', mec='white', zorder=5)
            ax.plot(*xy[-1], 'o', ms=10, mfc='#C62828', mec='white', zorder=5)
            ax.set_xlim(min(xs) - 25, max(xs) + 25); ax.set_ylim(min(ys) - 25, max(ys) + 25)
            ax.set_aspect('equal'); ax.set_xlabel('m east of start'); ax.set_ylabel('m north of start')
            fig.colorbar(sc, ax=ax, shrink=.8).set_label('distance to nearest mapped path (m)')
            ax.set_title(f"{row['walk']}  (green start, red end)", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(a.walks_dir, os.path.splitext(row['walk'])[0] + '_gps_offset.png'), dpi=120)
            plt.close(fig)

    if rows:
        out = os.path.join(a.walks_dir, 'gps_path_offset.csv')
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        med = st.median(r['median_m'] for r in rows)
        print(f'\n{len(rows)} walks; median of per-walk medians {med:.1f} m -> {out}')


if __name__ == '__main__':
    main()
