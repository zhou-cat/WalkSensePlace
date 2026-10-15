"""
predict_walks.py - per-walk, step-by-step lines for the maps, time-matched to each GPX:
    <stem>_pred_3class.geojson   grass/loose/paved   (3-class model)       [categorical]
    <stem>_pred_2class.geojson   soft/hard           (2-class model)       [categorical]
    <stem>_hardness.geojson      P(firm) x 100       (logistic regression) [continuous 0-100]
    <stem>_evenness.geojson      evenness 0-100      (unsupervised scaler) [continuous 0-100]
The map scripts (build_all_walks*.py) draw these automatically.

Follows the paper's pipeline, as defined in the v21 notebook it loads:
  * SEGMENTATION from the notebook (default 'events'): steps run from one initial contact
    to the next, from Ground Contact/<user>/<REC>_groundcontact.csv. Recordings without an
    event file are skipped rather than segmented another way (EVENTS_REQUIRED).
  * the notebook's feature set: META + DROP_FEATURES + TAILORED_DROP excluded (or KEEP_ONLY,
    if set), event-only features dropped under peaks, constant and exactly-duplicated
    columns dropped.
  * the notebook's XGBoost settings and soft class weighting (CLASS_WEIGHT_STRENGTH).
  * evenness from the evenness notebook's event-segmented scaler
    (evenness_scaler_events.json); a scaler fitted under a different segmentation is
    refused rather than misapplied.
  * each step is placed on the track at its INITIAL CONTACT.

Walk-in-training: by default the models train on every recording, so a walk whose
recording is also under --train-dir is predicted IN-SAMPLE. Each walk's status is
printed; --holdout walk retrains without that walk's recording (its walker's other walks
stay in), --holdout user without any of the walker's recordings.

Usage (from the project folder):
    python scripts/predict_walks.py --notebook notebooks/ground_condition_xgboost_v21_merged_3class.ipynb ^
        --train-dir "Individual Users" --walks-dir GPS [--holdout walk]
"""
import argparse, os, glob, json, sys, re, ast
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime

META = {'foot', 'step_id', 'surface', 'contact_time_sec', 'source_user', 'source_recording', 't_ms'}
EVENT_ONLY_FEATURES = ['stance_time_sec', 'stance_frac_true']
SOFT = {'grass', 'soft'}
COMPL_FEATS = ['accel_jerk_ratio', 'accel_hf_frac', 'accel_spec_centroid', 'accel_zcr',
               'gyro_hf_frac', 'gu_stance_frac', 'stance_duration', 'loading_rate_norm']
# the v21 / comparison-notebook XGBoost settings
XGB_PARAMS = dict(max_depth=4, n_estimators=250, learning_rate=0.05, subsample=0.9,
                  colsample_bytree=0.8, eval_metric='mlogloss', random_state=42, verbosity=0)


def rec_key(path):
    """A recording's name with any '_GPS' suffix removed (REC_x_GPS.csv -> REC_x.csv), so a
    walk's copy and its training twin are recognised as the same recording."""
    b = os.path.basename(path); stem, ext = os.path.splitext(b)
    return (stem[:-4] if stem.endswith('_GPS') else stem) + ext


# ======================= notebook extractor + contact events =======================
def _strip_glob(p):
    """'Individual Users/*' -> 'Individual Users' (the folder above any wildcard)."""
    parts = re.split(r'[\\/]', p)
    keep = []
    for x in parts:
        if any(ch in x for ch in '*?['):
            break
        keep.append(x)
    return os.sep.join(keep) if keep else '.'


def find_gc_dir(explicit, train_dir, nb_path):
    cands = [explicit] if explicit else []
    cands += [os.path.join(os.getcwd(), 'Ground Contact'),
              os.path.join(os.path.dirname(os.path.abspath(_strip_glob(train_dir))), 'Ground Contact'),
              os.path.join(os.path.dirname(os.path.abspath(nb_path)), '..', 'Ground Contact')]
    return next((os.path.normpath(c) for c in cands if c and os.path.isdir(c)), None)


def load_extractor(nb_path, gc_dir=None, train_dir='.'):
    nb = json.load(open(nb_path, encoding='utf-8')); S = lambda c: ''.join(c['source'])
    codes = [S(c) for c in nb['cells'] if c['cell_type'] == 'code']
    cfg     = next(s for s in codes if 'DROP_FEATURES =' in s)
    helpers = next(s for s in codes if 'def label_surface_segments' in s)
    extr    = next(s for s in codes if 'def extract_step_features' in s)
    # time-stamp each step at its INITIAL CONTACT (window start), where the step lands
    extr = extr.replace("        rows.append(row)",
        "        try:\n            row['t_ms'] = float(pd.to_numeric(df['timestamp'], errors='coerce').iloc[p1])\n"
        "        except Exception:\n            row['t_ms'] = float('nan')\n        rows.append(row)", 1)
    ns = {'__name__': 'wsp_extract'}
    for cell, tag in [(cfg, 'cfg'), (helpers, 'helpers'), (extr, 'extr')]:
        exec(compile(cell, tag, 'exec'), ns)
    ns.setdefault('compute_session_stats', lambda raw: None)
    ns.setdefault('SEGMENTATION', 'peaks')
    ns.setdefault('EVENTS_REQUIRED', False)
    # KEEP_ONLY lives in a later cell; read its literal value if it is set there
    for s in codes:
        m = re.search(r'^KEEP_ONLY\s*=\s*(\[.*?\])', s, re.M | re.S)
        if m:
            try: ns['KEEP_ONLY'] = ast.literal_eval(m.group(1))
            except Exception: pass
            break
    # the notebook's GC_DIR is relative to notebooks/; resolve it from here instead
    gd = find_gc_dir(gc_dir, train_dir, nb_path)
    if gd: ns['GC_DIR'] = gd
    ns['_GC_INDEX'] = None
    # register as the default for train_surface(full, drop, remap2), so scripts importing
    # this module (fill_gis_table.py, time_inference.py) train the same model as the maps
    global _NS
    _NS = ns
    return ns


def _gc_index(ns):
    if ns.get('_GC_INDEX') is None:
        idx = {}
        gd = ns.get('GC_DIR')
        if gd and os.path.isdir(gd):
            for f in glob.glob(os.path.join(gd, '**', '*_groundcontact.csv'), recursive=True):
                idx[os.path.basename(f)[:-len('_groundcontact.csv')] + '.csv'] = \
                    os.path.basename(os.path.dirname(f))
        ns['_GC_INDEX'] = idx
    return ns['_GC_INDEX']


def recording_events(ns, rec_path, sole_id):
    """Contact events for one sole of a recording, via the notebook's contact_events, or
    None. The user folder is found from the event file itself, so it works for walk
    recordings outside Individual Users too."""
    if ns.get('SEGMENTATION') != 'events' or 'contact_events' not in ns:
        return None
    key = rec_key(rec_path); user = _gc_index(ns).get(key)
    if user is None:
        return None
    return ns['contact_events'](user, os.path.join('x', key), sole_id)


def has_events(ns, rec_path):
    return rec_key(rec_path) in _gc_index(ns)


def find_recordings(root):
    files = glob.glob(os.path.join(root, '**', 'REC_*.csv'), recursive=True) or \
            glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)
    return sorted(f for f in files if '_GPS' not in os.path.basename(f)
                  and not f.endswith('_groundcontact.csv'))


def _extract_recording(ns, raw, rec_path, ss, labelled_only):
    lss, pp, extract = ns['label_surface_segments'], ns['preprocess'], ns['extract_step_features']
    frames = []
    soles = [(1, 'right'), (2, 'left')] if labelled_only else \
            [(s, 'right' if str(s) in ('1', '1.0') else 'left' if str(s) in ('2', '2.0') else f'sole{s}')
             for s in raw['sole_id'].dropna().unique()]
    for sole, foot in soles:
        sub = raw[raw['sole_id'] == sole].copy(); lab = lss(sub)
        if labelled_only and (lab is None or len(lab) == 0):
            continue
        dfp = pp(lab if lab is not None else sub)
        ev = recording_events(ns, rec_path, int(float(sole)))
        fe = extract(dfp, foot_label=foot, session_stats=ss, events=ev)
        if len(fe): frames.append(fe)
    return frames


def build_training(ns, train_dir):
    css = ns.get('compute_session_stats', lambda r: None); frames = []; skipped = []
    need = ns.get('SEGMENTATION') == 'events' and ns.get('EVENTS_REQUIRED', True)
    for f in find_recordings(train_dir):
        if need and not has_events(ns, f):
            skipped.append(os.path.basename(f)); continue
        try: raw = pd.read_csv(f, low_memory=False)
        except Exception: continue
        if 'sole_id' not in raw.columns: continue
        ss = css(raw)
        for fe in _extract_recording(ns, raw, f, ss, labelled_only=True):
            fe['source_recording'] = os.path.basename(f); frames.append(fe)
    if skipped:
        print(f"  skipped {len(skipped)} training recording(s) with no contact-event file: "
              f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
    if not frames: sys.exit(f"No labelled steps from {train_dir}.")
    full = pd.concat(frames, ignore_index=True).dropna(subset=['surface'])
    return full[full['surface'] != 'unknown']


def extract_features(ns, csv_path):
    css = ns.get('compute_session_stats', lambda r: None)
    raw = pd.read_csv(csv_path, low_memory=False)
    if 'sole_id' not in raw.columns: return pd.DataFrame()
    frames = _extract_recording(ns, raw, csv_path, css(raw), labelled_only=False)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ================================ models =================================
def make_model():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(**XGB_PARAMS)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250,
                                              random_state=42)


def soft_sample_weight(y, strength):
    """The notebook's class weighting: (n / (k * count)) ** strength. strength 0.5 is
    halfway (on a log scale) between no weighting and fully balanced."""
    if strength <= 0: return None
    cs, cnt = np.unique(y, return_counts=True); n, k = len(y), len(cs)
    w = {c: (n / (k * ct)) ** strength for c, ct in zip(cs, cnt)}
    return np.array([w[v] for v in y])


def feature_columns(df, drop, ns=None):
    """The notebook's feature selection, plus its guards."""
    ns = ns or {}
    keep_only = ns.get('KEEP_ONLY') or []
    if keep_only:
        feats = [c for c in dict.fromkeys(keep_only) if c in df.columns]
    else:
        excl = set(META) | set(drop) | set(ns.get('TAILORED_DROP', []))
        feats = [c for c in df.columns if c not in excl]
    if ns.get('SEGMENTATION', 'events') != 'events':
        feats = [c for c in feats if c not in EVENT_ONLY_FEATURES]
    out, seen = [], set()
    for c in feats:
        col = pd.to_numeric(df[c], errors='coerce')
        if col.nunique(dropna=True) <= 1:
            continue                                    # constant
        h = tuple(pd.util.hash_pandas_object(col.fillna(-12345.6789), index=False))
        if h in seen:
            continue                                    # exact duplicate (cycle_duration)
        seen.add(h); out.append(c)
    return out


def train_surface(full, drop, remap2=False, ns=None):
    from sklearn.preprocessing import LabelEncoder
    ns = ns if ns is not None else _NS
    df = full.copy()
    if remap2: df['surface'] = np.where(df['surface'] == 'grass', 'soft', 'hard')
    feats = feature_columns(df, drop, ns)
    X = df[feats].fillna(0).values; le = LabelEncoder(); y = le.fit_transform(df['surface'].values)
    m = make_model(); w = soft_sample_weight(y, (ns or {}).get('CLASS_WEIGHT_STRENGTH', 0.5))
    try: m.fit(X, y, sample_weight=w)
    except TypeError: m.fit(X, y)
    return m, le, feats


_NS = {}   # set in main(); lets train_surface(full, drop) keep its old 3-argument form


def train_hardness(full, weights='balanced'):
    """P(firm) from compliance features. weights='balanced' reproduces the earlier maps;
    'none' keeps the training data's own class balance - see --hardness-weights."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import Pipeline
    compl = [c for c in COMPL_FEATS if c in full.columns]
    y = (~full['surface'].isin(SOFT)).astype(int).values
    pipe = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(
        max_iter=1000, class_weight='balanced' if weights == 'balanced' else None))])
    return pipe.fit(full[compl].fillna(0).values, y), compl


# ======================= unsupervised evenness (events-aware) =======================
from scipy.signal import find_peaks
from scipy.stats import norm
EV_MIN, EV_MAX = 15, 150
EV_SENSORS = [f'pressure_{i:02d}' for i in range(1, 13)]
EV_WITHIN = ['orient_range_a', 'orient_range_b', 'cop_wander', 'press_jitter', 'gyro_std']
EV_STEP = ['stance_dur', 'loading_rate', 'impact', 'tiltA', 'tiltB']
EV_FEATS = EV_WITHIN + [f'{c}_roll3_std' for c in EV_STEP]


def ev_find_steps(tot):
    mx = np.nanmax(tot) if np.nanmax(tot) > 0 else 1.0
    pk, _ = find_peaks(tot, distance=25, prominence=mx * 0.15); return pk


def ev_extract_sole(df, foot, ns=None, events=None):
    """As evenness_score_unsupervised: windows from contact events when given (the
    notebook's event_windows), else total-pressure peaks."""
    df = df.reset_index(drop=True)
    if len(df) < EV_MIN: return pd.DataFrame()
    P = df[EV_SENSORS].apply(pd.to_numeric, errors='coerce').fillna(0).values; tot = P.sum(1)
    ax = pd.to_numeric(df['accel_x'], errors='coerce').fillna(0).values
    ay = pd.to_numeric(df['accel_y'], errors='coerce').fillna(0).values
    az = pd.to_numeric(df['accel_z'], errors='coerce').fillna(0).values
    G = df[['gyro_x', 'gyro_y', 'gyro_z']].apply(pd.to_numeric, errors='coerce').fillna(0).values
    Gc = G - np.median(G, 0)
    amag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2); gmag = np.sqrt((Gc ** 2).sum(1))
    tA = np.degrees(np.arctan2(ax, np.hypot(ay, az))); tB = np.degrees(np.arctan2(az, np.hypot(ax, ay)))
    ts = pd.to_numeric(df['timestamp'], errors='coerce').values if 'timestamp' in df.columns else np.arange(len(df))
    if events is not None and ns is not None and 'event_windows' in ns:
        pk = np.asarray(ns['event_windows'](df, events)[0], dtype=int)
    else:
        pk = ev_find_steps(tot)
    rows = []; o = 0
    for i in range(len(pk) - 1):
        a, b = int(pk[i]), int(pk[i + 1]); n = b - a
        if n < EV_MIN or n > EV_MAX: continue
        Pw = P[a:b]; totw = tot[a:b]; sc = totw.mean() + 1e-9; wam = amag[a:b]; wgm = gmag[a:b]
        Qn = Pw / (Pw.sum(1, keepdims=True) + 1e-9); t = int(np.argmin(wgm))
        rows.append(dict(foot=foot, order=o, t_ms=float(ts[a]),          # initial contact
            orient_range_a=float(tA[a:b].max() - tA[a:b].min()), orient_range_b=float(tB[a:b].max() - tB[a:b].min()),
            cop_wander=float(np.mean(np.abs(np.diff(Qn, axis=0)).sum(1))) if n > 1 else 0.0,
            press_jitter=float(np.mean(np.std(np.diff(Pw, n=2, axis=0), axis=0)) / sc) if n > 2 else 0.0,
            gyro_std=float(np.mean([np.std(Gc[a:b, k]) for k in range(3)])),
            stance_dur=float((totw > 0.5 * sc).mean()), loading_rate=float(np.max(np.diff(totw)) / sc),
            impact=float(wam[:max(1, n // 3)].max() / (np.median(wam) + 1e-9)),
            tiltA=float(tA[a:b][t]), tiltB=float(tB[a:b][t]))); o += 1
    return pd.DataFrame(rows)


def ev_extract(csv_path, ns=None):
    raw = pd.read_csv(csv_path, low_memory=False)
    if 'sole_id' not in raw.columns: return pd.DataFrame()
    parts = []
    for sole in raw['sole_id'].dropna().unique():
        foot = 'right' if str(sole) in ('1', '1.0') else 'left' if str(sole) in ('2', '2.0') else f'sole{sole}'
        ev = recording_events(ns, csv_path, int(float(sole))) if ns else None
        if ns and ns.get('SEGMENTATION') == 'events' and ev is None:
            continue                     # never mix segmentations inside one scaler's use
        st = ev_extract_sole(raw[raw['sole_id'] == sole], foot, ns, ev)
        if len(st): parts.append(st)
    if not parts: return pd.DataFrame()
    S = pd.concat(parts, ignore_index=True).sort_values(['foot', 'order'])
    for c in EV_STEP:
        S[f'{c}_roll3_std'] = S.groupby('foot')[c].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    return S


def _robust_cs(x):
    m = float(np.median(x)); q = float(np.subtract(*np.percentile(x, [75, 25])))
    return m, (q if q > 0 else (float(np.std(x)) or 1.0))


def fit_even_scaler(df, segmentation):
    cs = {c: list(_robust_cs(df[c].values)) for c in EV_FEATS if c in df.columns}
    Z = np.column_stack([np.clip((df[c].values - cs[c][0]) / (cs[c][1] + 1e-9), -4, 4) for c in cs]); U = Z.mean(1)
    um, us = _robust_cs(U)
    return {'segmentation': segmentation, 'features': list(cs.keys()), 'feat_center_scale': cs,
            'U_center': um, 'U_scale': us}


def apply_even_scaler(df, scaler):
    feats = scaler['features']; cs = scaler['feat_center_scale']
    for c in feats:
        if c not in df: df[c] = 0.0
    Z = np.column_stack([np.clip((df[c].values - cs[c][0]) / (cs[c][1] + 1e-9), -4, 4) for c in feats]); U = Z.mean(1)
    return np.clip(100 * norm.cdf(-(U - scaler['U_center']) / (scaler['U_scale'] + 1e-9)), 0, 100)


def find_even_scaler(explicit, train_dir, walks_dir, segmentation):
    """The evenness notebooks write evenness_scaler_<seg>.json (or _3class_<seg>). A scaler
    recording a different segmentation - or none, i.e. built before the switch - is
    refused, because its centres do not apply to differently cut windows."""
    names = [f'evenness_scaler_{segmentation}.json', f'evenness_scaler_3class_{segmentation}.json',
             'evenness_scaler.json']
    roots = [_strip_glob(train_dir), walks_dir, '.']
    cands = [explicit] if explicit else []
    for d in roots:
        for sub in ('Charts_Graphs', os.path.join('..', 'Charts_Graphs'), ''):
            cands += [os.path.join(d, sub, n) for n in names]
    refused, seen = [], set()
    for c in cands:
        if not (c and os.path.exists(c)): continue
        rp = os.path.realpath(c)
        if rp in seen: continue
        seen.add(rp)
        try: sc = json.load(open(c, encoding='utf-8'))
        except Exception: continue
        seg = sc.get('segmentation', 'peaks')           # pre-switch scalers were peaks
        if seg == segmentation:
            return sc, c, refused
        refused.append(f'{c} (segmentation={seg})')
    return None, None, refused


# ============================ GPS and output =============================
def parse_gpx(path):
    root = ET.parse(path).getroot(); strip = lambda t: t.split('}')[-1]; pts = []
    for pt in root.iter():
        if strip(pt.tag) != 'trkpt': continue
        lat = float(pt.attrib['lat']); lon = float(pt.attrib['lon']); t = None
        for c in pt:
            if strip(c.tag) == 'time': t = (c.text or '').strip()
        if not t: continue
        pts.append({'lon': lon, 'lat': lat, 'tms': int(datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp() * 1000)})
    return pts


def match_to_track(pts, step_vals, max_gap_ms=1500, fill='unknown'):
    if not step_vals: return [fill] * len(pts)
    st = np.array([s[0] for s in step_vals], float); v = [s[1] for s in step_vals]
    o = np.argsort(st); st = st[o]; v = [v[i] for i in o]; out = []
    for p in pts:
        j = int(np.searchsorted(st, p['tms'])); cand = [k for k in (j - 1, j) if 0 <= k < len(st)]
        if not cand: out.append(fill); continue
        k = min(cand, key=lambda k: abs(st[k] - p['tms']))
        out.append(v[k] if abs(st[k] - p['tms']) <= max_gap_ms else fill)
    return out


def seg_fc(pts, vals, key):
    feats = []
    for i in range(len(pts) - 1):
        val = vals[i]
        if isinstance(val, float) and np.isnan(val): val = None
        feats.append({'type': 'Feature', 'properties': {key: val},
            'geometry': {'type': 'LineString', 'coordinates': [[pts[i]['lon'], pts[i]['lat']], [pts[i + 1]['lon'], pts[i + 1]['lat']]]}})
    return {'type': 'FeatureCollection', 'features': feats}


def paired_gps_csv(gpx_pts, walks_dir):
    """The insole CSV in the walks folder whose time range overlaps the GPX most."""
    if not gpx_pts: return None
    g0, g1 = gpx_pts[0]['tms'], gpx_pts[-1]['tms']; best, ov = None, 0
    for f in glob.glob(os.path.join(walks_dir, '*.csv')):
        if f.endswith('_groundcontact.csv'): continue
        try: ts = pd.to_numeric(pd.read_csv(f, usecols=['timestamp'])['timestamp'], errors='coerce').dropna()
        except Exception: continue
        if not len(ts): continue
        o = max(0, min(g1, int(ts.max())) - max(g0, int(ts.min())))
        if o > ov: ov, best = o, f
    return best


# ================================= main ==================================
def main():
    global _NS
    ap = argparse.ArgumentParser()
    ap.add_argument('--notebook', required=True); ap.add_argument('--train-dir', required=True)
    ap.add_argument('--walks-dir', default='.'); ap.add_argument('--pred-dir', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--gc-dir', default=None, help="'Ground Contact' folder (auto-found if omitted)")
    ap.add_argument('--holdout', choices=['none', 'walk', 'user'], default='none',
                    help="none: train on everything (a walk in --train-dir is predicted "
                         "in-sample); walk: retrain without that walk's recording; user: "
                         "retrain without any of the walker's recordings")
    ap.add_argument('--hardness-weights', choices=['balanced', 'none'], default='balanced',
                    help="class weights for P(firm). 'balanced' (default, as before) "
                         "assumes firm and soft are equally common, which pulls every value "
                         "toward soft when firm steps are the majority; 'none' keeps the "
                         "data's own balance")
    ap.add_argument('--evenness-scaler', default=None, help='path to an evenness scaler json')
    a = ap.parse_args(); a.out = a.out or os.path.join(a.walks_dir, 'walk_maps'); os.makedirs(a.out, exist_ok=True)

    print("loading extractor ..."); ns = load_extractor(a.notebook, a.gc_dir, a.train_dir); _NS = ns
    seg = ns.get('SEGMENTATION', 'peaks')
    print(f"  segmentation: {seg}" + (f"  (events from {ns.get('GC_DIR')})" if seg == 'events' else ''))
    if seg == 'events' and not _gc_index(ns):
        sys.exit("SEGMENTATION='events' but no contact-event files were found. Pass --gc-dir "
                 "pointing at your 'Ground Contact' folder.")
    print("building training features ..."); full = build_training(ns, a.train_dir)
    print(f"  {len(full)} labelled steps {dict(full.surface.value_counts())}")
    drop = ns.get('DROP_FEATURES', [])
    user_of = {rec_key(f): os.path.basename(os.path.dirname(f)) for f in find_recordings(a.train_dir)}
    train_keys = full['source_recording'].map(rec_key)
    users = train_keys.map(user_of)

    all_classes = set(full['surface'])

    def fit_all(mask=None):
        sub = full if mask is None else full[mask.values]
        missing = all_classes - set(sub['surface'])
        if missing:
            # a held-out model that never saw a class can never predict it - refuse rather
            # than fall back to the in-sample model, which would hide the leakage again
            return None, sorted(missing)
        m3 = train_surface(sub, drop, False, ns); m2 = train_surface(sub, drop, True, ns)
        hp = train_hardness(sub, a.hardness_weights)
        return (m3, m2, hp), None

    print("training surface models (3-class, 2-class) and hardness P(firm) ...")
    base_models, _ = fit_all()
    print(f"  3-class model: {len(base_models[0][2])} features, XGBoost {XGB_PARAMS['n_estimators']} trees "
          f"depth {XGB_PARAMS['max_depth']}, class-weight strength {ns.get('CLASS_WEIGHT_STRENGTH', 0.5)}")
    print(f"  hardness weights: {a.hardness_weights}   holdout: {a.holdout}")

    even_scaler, src, refused = find_even_scaler(a.evenness_scaler, a.train_dir, a.walks_dir, seg)
    for r in refused:
        print(f"  evenness: refused {r} - built under a different segmentation")
    if even_scaler:
        print(f"evenness: using scaler from {src}")
    else:
        print(f"evenness: no {seg} scaler found -> fitting one on the training recordings ...")
        ev_tr = [ev_extract(f, ns) for f in find_recordings(a.train_dir)]
        ev_tr = [d for d in ev_tr if len(d)]
        if not ev_tr: sys.exit("could not build evenness training features")
        even_scaler = fit_even_scaler(pd.concat(ev_tr, ignore_index=True), seg)

    def lab(pdf, model):
        m, le, feats = model
        X = pdf.reindex(columns=feats).fillna(0).values
        return le.inverse_transform(m.predict(X))

    def hard(pdf, hp):
        pipe, compl = hp
        return pipe.predict_proba(pdf.reindex(columns=compl).fillna(0).values)[:, 1] * 100

    info = dict(segmentation=seg, gc_dir=ns.get('GC_DIR'), holdout=a.holdout,
                hardness_weights=a.hardness_weights, xgb=XGB_PARAMS,
                class_weight_strength=ns.get('CLASS_WEIGHT_STRENGTH', 0.5),
                n_features_3class=len(base_models[0][2]), evenness_scaler=src or 'fitted here',
                step_time='initial contact', walks={})
    for gpx in sorted(glob.glob(os.path.join(a.walks_dir, '*.gpx'))):
        stem = os.path.splitext(os.path.basename(gpx))[0]; pts = parse_gpx(gpx)
        if len(pts) < 2: print(f"[skip] {stem}: <2 pts"); continue
        gps = paired_gps_csv(pts, a.walks_dir)
        if not gps: print(f"[warn] {stem}: no insole recording overlaps this GPX"); continue
        key = rec_key(gps)
        search = [os.path.join(os.path.dirname(gps), key)]
        for d in [a.pred_dir, a.walks_dir, _strip_glob(a.train_dir)]:
            if d: search += glob.glob(os.path.join(d, '**', key), recursive=True)
        rec = next((c for c in search if c and os.path.exists(c)), gps)
        if seg == 'events' and not has_events(ns, rec):
            print(f"[skip] {stem}: no contact-event file for {key} - not predicting with a "
                  "different segmentation"); continue
        in_train = key in set(train_keys); u = user_of.get(key)
        if a.holdout == 'walk' and in_train:
            models, missing = fit_all(train_keys != key); how = 'held out (walk)'
        elif a.holdout == 'user' and u is not None:
            models, missing = fit_all(users != u); how = f'held out (user {u})'
        else:
            missing = None
            models = base_models
            how = 'IN-SAMPLE (walk is in the training set)' if in_train else 'not in training'
        if missing:
            print(f"[skip] {stem}: holding it out leaves no {missing} steps in training, so "
                  "that model could never predict them - not written"); continue
        pdf = extract_features(ns, rec)
        if not len(pdf): print(f"[warn] {stem}: 0 steps from {os.path.basename(rec)}"); continue
        t = pd.to_numeric(pdf['t_ms'], errors='coerce').values

        def write(vals, tag, keyname, fill):
            mv = match_to_track(pts, list(zip(t, vals)), fill=fill)
            json.dump(seg_fc(pts, mv, keyname), open(os.path.join(a.out, f"{stem}_{tag}.geojson"), 'w'))
        write(lab(pdf, models[0]), 'pred_3class', 'surface', 'unknown')
        write(lab(pdf, models[1]), 'pred_2class', 'surface', 'unknown')
        write(hard(pdf, models[2]), 'hardness', 'value', float('nan'))
        evdf = ev_extract(rec, ns)
        if len(evdf):
            ev_vals = apply_even_scaler(evdf, even_scaler); ev_t = pd.to_numeric(evdf['t_ms'], errors='coerce').values
            evm = match_to_track(pts, list(zip(ev_t, ev_vals)), fill=float('nan'))
            json.dump(seg_fc(pts, evm, 'value'), open(os.path.join(a.out, f"{stem}_evenness.geojson"), 'w'))
        else:
            print(f"[warn] {stem}: 0 evenness steps")
        info['walks'][stem] = dict(recording=rec, in_training=in_train, model=how, steps=int(len(pdf)))
        print(f"[ok] {stem}: {len(pdf)} steps from {rec} | {how}")
    json.dump(info, open(os.path.join(a.out, 'predict_walks_run.json'), 'w'), indent=1, default=str)
    print(f"\nDone -> geojsons in '{a.out}' (settings in predict_walks_run.json).")


if __name__ == '__main__':
    main()
