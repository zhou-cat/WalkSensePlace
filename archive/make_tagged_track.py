"""
Build the GROUND-TRUTH tagged track (track_tagged.geojson) for the surface maps.

The ground truth = a GPX (coordinates) + the matching insole recording CSV (surface
tags), time-matched so each GPS point gets the surface you tagged at that instant.
Both files MUST be the same walk (their timestamps must overlap).

Usage:
    python make_tagged_track.py  <walk.gpx>  <REC_*.csv>
e.g.
    python make_tagged_track.py  activity_22645980458.gpx  REC_20260424_190906_GPS.csv

Writes track_tagged.geojson (per-segment 'tag' in your classes). Paste its contents
into the map's TAGGED variable (see note printed at the end).
"""
import sys, json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import numpy as np, pandas as pd

GPX = sys.argv[1] if len(sys.argv) > 1 else 'activity_22645980458.gpx'
CSV = sys.argv[2] if len(sys.argv) > 2 else 'REC_20260424_190906_GPS.csv'

# --- surface markers in the insole CSV -> your classes ---
SURFACE_MAP = {
    'grass': 'grass',
    'dirt': 'loose', 'compact ground': 'loose', 'compacted ground': 'loose',
    'concrete': 'paved', 'asphalt': 'paved', 'exposed aggregate': 'paved', 'brick': 'paved',
}
EXCLUDE_TAGS = {'transition', 'wait'}     # act as boundaries but leave the region unknown
SENSOR_COLS = set(['sole_id','timestamp','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z',
                   'magn_x','magn_y','magn_z','corrupt'] + [f'pressure_{i:02d}' for i in range(1,13)])

# --- 1) read insole CSV, build time-ordered surface events (carry-forward) ---
d = pd.read_csv(CSV, low_memory=False)
if 'timestamp' not in d.columns:
    sys.exit("CSV has no 'timestamp' column.")
d = d.sort_values('timestamp').reset_index(drop=True)
events = []
for col in d.columns:
    if col in SENSOR_COLS: continue
    key = col.strip().lower()
    tag = None if key in EXCLUDE_TAGS else SURFACE_MAP.get(key)
    if key in EXCLUDE_TAGS: tag = '__EXC__'
    elif tag is None: continue
    for idx in d.index[d[col].astype(str).str.strip() == 'x']:
        events.append((int(d.loc[idx, 'timestamp']), tag))
events = sorted(set(events))
if not events:
    sys.exit("No surface markers found in the CSV (expected columns like grass/dirt/concrete/... "
             "with an 'x' in tagged rows).")
ev_ts  = np.array([e[0] for e in events])
ev_lab = [e[1] for e in events]
def tag_at(ts_ms):
    i = np.searchsorted(ev_ts, ts_ms, side='right') - 1
    if i < 0: return None
    lab = ev_lab[i]
    return None if lab == '__EXC__' else lab

# --- 2) read GPX (time + lat/lon) ---
root = ET.parse(GPX).getroot(); strip = lambda t: t.split('}')[-1]
pts = []
for pt in root.iter():
    if strip(pt.tag) != 'trkpt': continue
    lat = float(pt.attrib['lat']); lon = float(pt.attrib['lon']); t = None
    for c in pt:
        if strip(c.tag) == 'time': t = (c.text or '').strip()
    if t is None: continue
    tms = int(datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp() * 1000)
    pts.append({'lat': lat, 'lon': lon, 'tms': tms})
if not pts:
    sys.exit("No <trkpt> points with <time> in the GPX.")

# --- sanity: do the two clocks overlap? ---
g0, g1 = pts[0]['tms'], pts[-1]['tms']; c0, c1 = int(ev_ts.min()), int(d['timestamp'].max())
overlap = max(0, min(g1, c1) - max(g0, c0)) / 1000.0
def _iso(ms): return datetime.fromtimestamp(ms/1000, tz=timezone.utc).isoformat()
print(f"GPX span : {_iso(g0)} -> {_iso(g1)}")
print(f"CSV span : {_iso(c0)} -> {_iso(c1)}")
if overlap <= 0:
    print("WARNING: the GPX and CSV time ranges DO NOT overlap — this GPX and this insole "
          "recording are not the same walk. Every point will be 'unknown'.")
else:
    print(f"overlap  : {overlap:.0f}s")

# --- 3) tag each GPX point, write per-segment GeoJSON ---
for p in pts: p['tag'] = tag_at(p['tms'])
from collections import Counter
print("tag distribution:", dict(Counter(p['tag'] or 'unknown' for p in pts)))
feats = [{'type': 'Feature',
          'properties': {'tag': pts[i]['tag'] or 'unknown'},
          'geometry': {'type': 'LineString',
                       'coordinates': [[pts[i]['lon'], pts[i]['lat']],
                                       [pts[i+1]['lon'], pts[i+1]['lat']]]}}
         for i in range(len(pts)-1)]
json.dump({'type': 'FeatureCollection', 'features': feats}, open('track_tagged.geojson', 'w'))
print(f"\nwrote track_tagged.geojson ({len(feats)} segments)")
print("-> paste its contents into the map's `const TAGGED = (...)` variable, save, reload.")
