"""
predict_walks.py — run the trained model over each walk's raw recording and write a
time-matched prediction track (the "3rd line") for the maps. NO pasting/copying.

For each walk it predicts on the recording named like the walk's insole CSV but WITHOUT
the trailing "_GPS" (e.g. REC_20260424_190906.csv), and writes, into --out:
    <gpx_stem>_pred_3class.geojson   (grass/loose/paved)
    <gpx_stem>_pred_2class.geojson   (soft/hard)
build_all_walks.py then draws these automatically as extra parallel lines.

It reuses YOUR notebook's exact feature extractor (execs its cells) so features match
the model. Trees (XGBoost) are used if installed, else HistGradientBoosting as a stand-in.

Usage:
    pip install xgboost scikit-learn scipy pandas numpy gpxpy
    python predict_walks.py --notebook ground_condition_xgboost_v16_merged_3class.ipynb \
        --train-dir "C:/Users/Sam/WalkSensePlace/Individual Users" \
        --walks-dir "C:/Users/Sam/WalkSensePlace/GPS" \
        --out       "C:/Users/Sam/WalkSensePlace/GPS/walk_maps"
"""
import argparse, os, glob, json, sys
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime

META = {'foot','step_id','surface','contact_time_sec','source_user','source_recording','t_ms'}

# ---------- load the notebook's real extractor (config + helpers + extractor cells) ----------
def load_extractor(nb_path):
    nb = json.load(open(nb_path, encoding='utf-8')); S = lambda c: ''.join(c['source'])
    codes = [S(c) for c in nb['cells'] if c['cell_type'] == 'code']
    cfg     = next(s for s in codes if 'DROP_FEATURES =' in s)
    helpers = next(s for s in codes if 'def label_surface_segments' in s)
    extr    = next(s for s in codes if 'def extract_step_features' in s)
    # add a per-step timestamp (t_ms) for aligning predictions to the GPX; feature logic untouched
    extr = extr.replace(
        "        rows.append(row)",
        "        try:\n"
        "            row['t_ms'] = float(pd.to_numeric(df['timestamp'], errors='coerce').iloc[(p1+p2)//2])\n"
        "        except Exception:\n"
        "            row['t_ms'] = float('nan')\n"
        "        rows.append(row)", 1)
    ns = {'__name__': 'wsp_extract'}
    # neutralise a few notebook side effects that need files/plots we don't want here
    for cell, tag in [(cfg,'cfg'), (helpers,'helpers'), (extr,'extr')]:
        exec(compile(cell, tag, 'exec'), ns)
    need = ['label_surface_segments','preprocess','extract_step_features','DROP_FEATURES']
    missing = [n for n in need if n not in ns]
    if missing: sys.exit(f"notebook missing {missing} — is this the v16 extractor notebook?")
    ns.setdefault('compute_session_stats', lambda raw: None)
    return ns

# ---------- build the labelled training table ----------
def find_recordings(root):
    files = glob.glob(os.path.join(root, '**', 'REC_*.csv'), recursive=True)
    if not files: files = glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)
    return sorted(f for f in files if '_GPS' not in os.path.basename(f))  # train on raw recs

def build_training(ns, train_dir):
    lss, pp, extract = ns['label_surface_segments'], ns['preprocess'], ns['extract_step_features']
    css = ns.get('compute_session_stats', lambda r: None)
    frames = []
    for f in find_recordings(train_dir):
        try: raw = pd.read_csv(f, low_memory=False)
        except Exception: continue
        if 'sole_id' not in raw.columns: continue
        ss = css(raw)
        for sole, foot in [(1,'right'), (2,'left')]:
            sub = raw[raw['sole_id'] == sole].copy()
            lab = lss(sub)
            if lab is None or len(lab) == 0: continue
            fe = extract(pp(lab), foot_label=foot, session_stats=ss)
            if len(fe):
                fe['source_recording'] = os.path.basename(f); frames.append(fe)
    if not frames: sys.exit(f"No labelled steps built from {train_dir}. Check the path / REC_*.csv files.")
    full = pd.concat(frames, ignore_index=True).dropna(subset=['surface'])
    full = full[full['surface'] != 'unknown']
    return full

def make_model():
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return Pipeline([('s', StandardScaler()),
                         ('x', XGBClassifier(n_estimators=350, learning_rate=0.05, max_depth=5,
                                             subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                             eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0))]), True
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=350, random_state=42), False

def train(full, drop_features, remap2=False):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    df = full.copy()
    if remap2: df['surface'] = np.where(df['surface'] == 'grass', 'soft', 'hard')
    feats = [c for c in df.columns if c not in META and c not in set(drop_features)]
    X = df[feats].fillna(0).values; le = LabelEncoder(); y = le.fit_transform(df['surface'].values)
    model, _ = make_model()
    try:
        model.fit(X, y, **{f'{model.steps[-1][0]}__sample_weight': compute_sample_weight('balanced', y)}) \
            if hasattr(model, 'steps') else model.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
    except Exception:
        model.fit(X, y)
    return model, le, feats

# ---------- predict on one raw recording ----------
def predict_recording(ns, csv_path, model, le, feats):
    lss, pp, extract = ns['label_surface_segments'], ns['preprocess'], ns['extract_step_features']
    css = ns.get('compute_session_stats', lambda r: None)
    raw = pd.read_csv(csv_path, low_memory=False); ss = css(raw)
    frames = []
    for sole, foot in [(1,'right'), (2,'left')]:
        sub = raw[raw['sole_id'] == sole].copy()
        lab = lss(sub)
        dfp = pp(lab if lab is not None else sub)
        fe = extract(dfp, foot_label=foot, session_stats=ss)
        if len(fe): frames.append(fe)
    if not frames: return []
    pred_df = pd.concat(frames, ignore_index=True)
    for c in feats:
        if c not in pred_df.columns: pred_df[c] = 0.0
    X = pred_df[feats].fillna(0).values
    yhat = le.inverse_transform(model.predict(X))
    return list(zip(pd.to_numeric(pred_df['t_ms'], errors='coerce').values, yhat))

# ---------- GPX + time-match ----------
def parse_gpx(path):
    root = ET.parse(path).getroot(); strip = lambda t: t.split('}')[-1]; pts = []
    for pt in root.iter():
        if strip(pt.tag) != 'trkpt': continue
        lat = float(pt.attrib['lat']); lon = float(pt.attrib['lon']); t = None
        for c in pt:
            if strip(c.tag) == 'time': t = (c.text or '').strip()
        if not t: continue
        tms = int(datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp() * 1000)
        pts.append({'lon': lon, 'lat': lat, 'tms': tms})
    return pts

def match_to_track(pts, step_preds, max_gap_ms=1500):
    if not step_preds: return ['unknown']*len(pts)
    st = np.array([s[0] for s in step_preds], float); lab = [s[1] for s in step_preds]
    order = np.argsort(st); st = st[order]; lab = [lab[i] for i in order]
    out = []
    for p in pts:
        j = int(np.searchsorted(st, p['tms']))
        cands = [k for k in (j-1, j) if 0 <= k < len(st)]
        if not cands: out.append('unknown'); continue
        k = min(cands, key=lambda k: abs(st[k]-p['tms']))
        out.append(lab[k] if abs(st[k]-p['tms']) <= max_gap_ms else 'unknown')
    return out

def seg_fc(pts, vals, key='surface'):
    return {'type':'FeatureCollection','features':[
        {'type':'Feature','properties':{key: vals[i]},
         'geometry':{'type':'LineString','coordinates':[[pts[i]['lon'],pts[i]['lat']],
                                                        [pts[i+1]['lon'],pts[i+1]['lat']]]}}
        for i in range(len(pts)-1)]}

def paired_gps_csv(gpx_pts, walks_dir):
    if not gpx_pts: return None
    g0, g1 = gpx_pts[0]['tms'], gpx_pts[-1]['tms']; best, ov = None, 0
    for f in glob.glob(os.path.join(walks_dir, '*.csv')):
        try:
            ts = pd.to_numeric(pd.read_csv(f, usecols=['timestamp'])['timestamp'], errors='coerce').dropna()
        except Exception:
            continue
        if not len(ts): continue
        o = max(0, min(g1, int(ts.max())) - max(g0, int(ts.min())))
        if o > ov: ov, best = o, f
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notebook', required=True, help='v16 extractor notebook (.ipynb)')
    ap.add_argument('--train-dir', required=True, help='folder of labelled REC_*.csv (Individual Users)')
    ap.add_argument('--walks-dir', default='.')
    ap.add_argument('--pred-dir', default=None, help='folder holding the raw (non-_GPS) recordings, if not alongside the walks')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    a.out = a.out or os.path.join(a.walks_dir, 'walk_maps'); os.makedirs(a.out, exist_ok=True)

    print("loading extractor from notebook ..."); ns = load_extractor(a.notebook)
    print("building training features ..."); full = build_training(ns, a.train_dir)
    print(f"  {len(full)} labelled steps, surfaces {dict(full.surface.value_counts())}")
    drop = ns.get('DROP_FEATURES', [])
    print("training 3-class model ..."); m3, le3, f3 = train(full, drop, remap2=False)
    print("training 2-class model ..."); m2, le2, f2 = train(full, drop, remap2=True)

    for gpx in sorted(glob.glob(os.path.join(a.walks_dir, '*.gpx'))):
        stem = os.path.splitext(os.path.basename(gpx))[0]
        pts = parse_gpx(gpx)
        if len(pts) < 2: print(f"[skip] {stem}: <2 pts"); continue
        gps_csv = paired_gps_csv(pts, a.walks_dir)
        if not gps_csv:
            print(f"[warn] {stem}: no overlapping insole CSV -> can't predict"); continue
        base = os.path.basename(gps_csv).replace('_GPS', '')          # the raw recording name
        search = [os.path.join(os.path.dirname(gps_csv), base)]
        for d in [a.pred_dir, a.walks_dir, a.train_dir]:
            if d: search += glob.glob(os.path.join(d, '**', base), recursive=True)
        pred_csv = next((c for c in search if c and os.path.exists(c)), None)
        if pred_csv is None:
            pred_csv = gps_csv   # fallback: the _GPS file has the full sensor stream too
            print(f"[note] {stem}: raw '{base}' not found; predicting on {os.path.basename(gps_csv)} instead")
        for tag, (mdl, le, feats) in [('3class',(m3,le3,f3)), ('2class',(m2,le2,f2))]:
            preds = predict_recording(ns, pred_csv, mdl, le, feats)
            vals = match_to_track(pts, preds)
            json.dump(seg_fc(pts, vals), open(os.path.join(a.out, f"{stem}_pred_{tag}.geojson"), 'w'))
        print(f"[ok] {stem}: predicted from {os.path.basename(pred_csv)} "
              f"(steps: {len(predict_recording(ns, pred_csv, m3, le3, f3))})")
    print(f"\nDone -> prediction geojsons in '{a.out}'. Run build_all_walks.py to draw them.")

if __name__ == '__main__':
    main()
