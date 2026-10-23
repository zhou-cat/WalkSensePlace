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

TWO FILES PER WALK, AND WHAT EACH IS FOR
    Individual Users/<user>/REC_<stamp>.csv   TRAINING copy: every tag, including the
                                             transition/wait regions the step gate needs.
                                             Models and the evenness scaler are fitted on
                                             these and nothing else.
    GPS/REC_<stamp>_GPS.csv                   WALK copy: the same recording tagged for
                                             surface underfoot only. This is the file the
                                             walk is PREDICTED from, and whose tags are
                                             the ground truth drawn on the map.
    GPS/REC_<stamp>.gpx                       the track the steps are placed on.
The two CSVs hold identical sensor data and differ only in their tag markers, so the walk
copy is never used for training and the training copy is never used for prediction.
(Earlier versions substituted the training copy when predicting a walk, which drew the
map from the wrong tagging regime.)

Walk-in-training: the two copies are the same recording, so training on
Individual Users/<user>/REC_<stamp>.csv and predicting GPS/REC_<stamp>_GPS.csv means
predicting steps the model has already seen. Each walk's status is printed;
--holdout walk refits without that recording (the walker's other recordings stay in),
--holdout user without any of the walker's recordings.

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


def walk_stamp(path):
    """The recording stamp shared by a walk's three files, e.g. REC_20260519_155900.

    GPS/REC_x_GPS.csv, GPS/REC_x.gpx and Individual Users/<u>/REC_x.csv all reduce to
    REC_x. Used ONLY to recognise that they describe the same walk - for pairing the GPX,
    for finding the contact-event file, and for holding the recording out of training.
    It never causes one file to stand in for another.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[:-4] if stem.endswith('_GPS') else stem


def rec_key(path):
    """Back-compat for scripts that import this module (fill_gis_table, time_inference)."""
    return walk_stamp(path) + '.csv'


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
                idx[os.path.basename(f)[:-len('_groundcontact.csv')]] = \
                    os.path.basename(os.path.dirname(f))
        ns['_GC_INDEX'] = idx
    return ns['_GC_INDEX']


def recording_events(ns, rec_path, sole_id):
    """Contact events for one sole of a recording, via the notebook's contact_events, or
    None. The user folder is found from the event file itself, so it works for walk
    recordings outside Individual Users too."""
    if ns.get('SEGMENTATION') != 'events' or 'contact_events' not in ns:
        return None
    key = walk_stamp(rec_path); user = _gc_index(ns).get(key)
    if user is None:
        return None
    return ns['contact_events'](user, os.path.join('x', key + '.csv'), sole_id)


def has_events(ns, rec_path):
    return walk_stamp(rec_path) in _gc_index(ns)


def find_recordings(root):
    """TRAINING recordings only: Individual Users/<user>/REC_*.csv.

    _GPS copies are excluded here by design - they are walk files, not training files.
    """
    files = glob.glob(os.path.join(root, '**', 'REC_*.csv'), recursive=True) or \
            glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)
    return sorted(f for f in files if '_GPS' not in os.path.basename(f)
                  and not f.endswith('_groundcontact.csv'))


def find_walk_csv(stamp, walks_dir):
    """The walk copy for a stamp: GPS/REC_<stamp>_GPS.csv, else GPS/REC_<stamp>.csv."""
    for name in (stamp + '_GPS.csv', stamp + '.csv'):
        p = os.path.join(walks_dir, name)
        if os.path.exists(p):
            return p
    hits = [p for p in glob.glob(os.path.join(walks_dir, '**', stamp + '*.csv'), recursive=True)
            if not p.endswith('_groundcontact.csv')]
    return hits[0] if hits else None


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


def walk_truth(ns, csv_path):
    """Per-step ground truth from a WALK file's own tags, as (t_ms, surface) pairs.

    The walk copy tags surface underfoot only, so steps are labelled by the surface marker
    they fall under without the training gate's transition and partial-label rules.
    """
    raw = pd.read_csv(csv_path, low_memory=False)
    if 'sole_id' not in raw.columns:
        return []
    out = []
    for sole in raw['sole_id'].dropna().unique():
        sub = raw[raw['sole_id'] == sole].copy()
        lab = ns['label_surface_segments'](sub)
        if lab is None or 'surface' not in lab.columns:
            continue
        dfp = ns['preprocess'](lab)
        ev = recording_events(ns, csv_path, int(float(sole)))
        if ns.get('SEGMENTATION') == 'events' and ev is None:
            continue
        tot = dfp[[c for c in dfp.columns if c.startswith('pressure_')]].to_numpy(float).sum(1)
        ts = pd.to_numeric(dfp['timestamp'], errors='coerce').values
        if ns.get('SEGMENTATION') == 'events' and ev is not None:
            st, en = ns['event_windows'](dfp, ev)
        else:
            pk = ns['find_step_windows'](dfp)
            st, en = (pk[:-1], pk[1:]) if len(pk) >= 2 else ([], [])
        sv = dfp['surface'].values
        _excl = {ns.get('EXCLUDE_LABEL', '__exclude__'), 'unknown', ''}
        for a_, b_ in zip(np.asarray(st, int), np.asarray(en, int)):
            lab_in = [v for v in sv[a_:b_] if isinstance(v, str) and v and v not in _excl]
            if not lab_in:
                continue        # unlabelled, or inside a transition the walk file marks
            vals, cnts = np.unique(lab_in, return_counts=True)
            out.append((float(ts[a_]), str(vals[int(np.argmax(cnts))])))
    return out


def extract_features(ns, csv_path):
    css = ns.get('compute_session_stats', lambda r: None)
    raw = pd.read_csv(csv_path, low_memory=False)
    if 'sole_id' not in raw.columns: return pd.DataFrame()
    frames = _extract_recording(ns, raw, csv_path, css(raw), labelled_only=False)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ============================ the 1D-CNN option ==========================
# The CNN learns from raw step windows instead of per-step features, so it needs its own
# extractor cells from the CNN notebook: SEQ_CHANNELS/T_LEN, _channel_matrix,
# _resample_zscore, step_windows_from_df (cell 5) and Conv1dNet/train_fold (cell 6).
# Segmentation, the step gate and the contact events are shared with the feature pipeline,
# so both models cut the same windows.
def _defs_only(cell):
    """Keep only imports, constants, functions and classes from a notebook cell.

    The CNN cells end with driver code that builds the notebook's own arrays; running it
    here would fail (no data in scope) or do work twice. Parsing the cell and re-emitting
    just the definitions takes the pieces without the side effects.
    """
    import ast as _ast
    tree = _ast.parse(cell)
    keep = []
    for node in tree.body:
        if isinstance(node, (_ast.Import, _ast.ImportFrom, _ast.FunctionDef,
                             _ast.AsyncFunctionDef, _ast.ClassDef)):
            keep.append(node)
        elif isinstance(node, _ast.Assign) and all(
                isinstance(t, _ast.Name) for t in node.targets):
            try:                      # constants only: T_LEN = 64, SEQ_CHANNELS = [...]
                _ast.literal_eval(node.value); keep.append(node)
            except Exception:
                pass
        elif isinstance(node, _ast.Try):
            keep.append(node)         # the usual "try: import torch" guard
    return _ast.unparse(_ast.Module(body=keep, type_ignores=[]))


def load_cnn(nb_path, ns):
    """Add the CNN notebook's window builder and model to an existing namespace."""
    nb = json.load(open(nb_path, encoding='utf-8'))
    codes = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    seq = next(s for s in codes if 'def step_windows_from_df' in s)
    mdl = next(s for s in codes if 'class Conv1dNet' in s)
    cfg = next(s for s in codes if 'T_LEN' in s)
    for name, cell in (('cnn-cfg', cfg), ('cnn-seq', seq), ('cnn-model', mdl)):
        try:
            exec(compile(_defs_only(cell), name, 'exec'), ns)
        except ModuleNotFoundError as e:
            if 'torch' in str(e):
                raise SystemExit('--model cnn needs PyTorch. Check with:\n'
                                 '  python -c "import torch; print(torch.__version__, '
                                 'torch.cuda.is_available())"')
            raise
        except Exception as e:                       # the config cell may lack context
            if name != 'cnn-cfg':
                raise RuntimeError(f'could not load {name} from {nb_path}: {e}')
    if 'device' not in ns:          # a computed assignment, so not kept by _defs_only
        import torch
        ns['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    for need in ('step_windows_from_df', 'Conv1dNet', 'train_fold', 'T_LEN', 'SEQ_CHANNELS'):
        if need not in ns:
            raise RuntimeError(f"{nb_path} did not define {need}")
    return ns


def _gate_counter():
    """The counter dict step_surface_label expects, fresh per recording."""
    return {'seen': 0, 'transition': 0, 'partial': 0, 'kept': 0, 'excluded': 0}


def cnn_windows(ns, csv_path, labelled_only):
    """Raw step windows (and labels, and step times) from one recording."""
    raw = pd.read_csv(csv_path, low_memory=False)
    if 'sole_id' not in raw.columns:
        return np.empty((0, ns['T_LEN'], len(ns['SEQ_CHANNELS'])), np.float32), [], []
    Xs, ys, ts = [], [], []
    soles = [(1, 'right'), (2, 'left')] if labelled_only else \
            [(sv, '') for sv in raw['sole_id'].dropna().unique()]
    for sole, _foot in soles:
        sub = raw[raw['sole_id'] == sole].copy()
        lab = ns['label_surface_segments'](sub)
        if labelled_only and (lab is None or not len(lab)):
            continue
        dfp = ns['preprocess'](lab if lab is not None else sub)
        ev = recording_events(ns, csv_path, int(float(sole)))
        if ns.get('SEGMENTATION') == 'events' and ev is None:
            continue
        # NOTE: step_windows_from_df's `stats` is the step-gate COUNTER dict that
        # step_surface_label increments, not the session statistics. Passing session
        # stats here raises KeyError('excluded').
        W, L = ns['step_windows_from_df'](dfp, _gate_counter(), events=ev)
        if not len(W):
            continue
        tms = pd.to_numeric(dfp['timestamp'], errors='coerce').values
        st = (ns['event_windows'](dfp, ev)[0] if (ns.get('SEGMENTATION') == 'events' and ev is not None)
              else ns['find_step_windows'](dfp)[:-1])
        Xs.append(np.asarray(W, np.float32)); ys.extend(list(L))
        ts.extend([float(tms[int(i)]) for i in np.asarray(st, int)[:len(W)]])
    if not Xs:
        return np.empty((0, ns['T_LEN'], len(ns['SEQ_CHANNELS'])), np.float32), [], []
    return np.concatenate(Xs, 0), ys, ts


def cnn_build_training(ns, train_dir):
    """Windows + labels + source recording for every TRAINING recording."""
    X, y, rec = [], [], []
    need = ns.get('SEGMENTATION') == 'events' and ns.get('EVENTS_REQUIRED', True)
    for f in find_recordings(train_dir):
        if need and not has_events(ns, f):
            continue
        Xi, yi, _ = cnn_windows(ns, f, labelled_only=True)
        keep = [k for k, v in enumerate(yi) if isinstance(v, str) and v and v != 'unknown']
        if not keep:
            continue
        X.append(Xi[keep]); y += [yi[k] for k in keep]
        rec += [os.path.basename(f)] * len(keep)
    if not X:
        sys.exit(f'no labelled CNN windows from {train_dir}')
    return np.concatenate(X, 0), np.array(y, dtype=object), np.array(rec, dtype=object)


def cnn_train(ns, X, y, strength=0.5, seed=42, max_epochs=None, label=''):
    """Fit Conv1dNet with an internal validation split for early stopping.

    Training is silent inside train_fold, so the caller prints before and after: on CPU a
    fit over ~20k windows takes minutes, and without this it looks like a hang.
    """
    import time as _time, torch
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder(); yi = le.fit_transform(y)
    rng = np.random.default_rng(seed); idx = rng.permutation(len(yi))
    cut = max(1, int(0.15 * len(idx))); va, tr = idx[:cut], idx[cut:]
    cs, cnt = np.unique(yi[tr], return_counts=True)
    w = np.ones(len(le.classes_), np.float32)
    for c, ct in zip(cs, cnt):
        w[c] = (len(yi[tr]) / (len(cs) * ct)) ** strength
    kw = {} if max_epochs is None else {'max_epochs': int(max_epochs)}
    print(f"    CNN fit{label}: {len(tr)} train / {len(va)} val windows on "
          f"{ns.get('device', 'cpu')}"
          + (f", max {max_epochs} epochs" if max_epochs else '')
          + ' ... ', end='', flush=True)
    _t0 = _time.time()
    model = ns['train_fold'](X[tr], yi[tr], X[va], yi[va], len(le.classes_), w, **kw)
    print(f'done in {_time.time() - _t0:.0f}s', flush=True)
    return model, le


def cnn_predict(ns, model, X):
    import torch
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 256):
            xb = torch.tensor(X[i:i + 256]).to(ns.get('device', 'cpu'))
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out) if out else np.array([], int)


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
    ap.add_argument('--model', choices=['xgboost', 'cnn'], default='xgboost',
                    help="which model labels the walks' surface class. 'cnn' needs "
                         '--cnn-notebook and PyTorch; hardness and evenness are unchanged.')
    ap.add_argument('--cnn-notebook', default=None,
                    help='the CNN comparison notebook (required with --model cnn)')
    ap.add_argument('--cnn-epochs', type=int, default=None,
                    help="cap the CNN's training epochs (the notebook's default is 80). "
                         'With --holdout walk the CNN is refit once per walk, so on CPU '
                         'the full run can take an hour or more; 25 is a reasonable cap '
                         'for a first pass.')
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
    if a.model == 'cnn':
        if not a.cnn_notebook:
            sys.exit('--model cnn requires --cnn-notebook')
        print('loading CNN ...'); load_cnn(a.cnn_notebook, ns)
        print(f"  CNN: {len(ns['SEQ_CHANNELS'])} channels x {ns['T_LEN']} samples per step"
              f"  (device {ns.get('device', 'cpu')})")
    seg = ns.get('SEGMENTATION', 'peaks')
    print(f"  segmentation: {seg}" + (f"  (events from {ns.get('GC_DIR')})" if seg == 'events' else ''))
    if seg == 'events' and not _gc_index(ns):
        sys.exit("SEGMENTATION='events' but no contact-event files were found. Pass --gc-dir "
                 "pointing at your 'Ground Contact' folder.")
    print("building training features ..."); full = build_training(ns, a.train_dir)
    print(f"  {len(full)} labelled steps {dict(full.surface.value_counts())}")
    drop = ns.get('DROP_FEATURES', [])
    user_of = {walk_stamp(f): os.path.basename(os.path.dirname(f))
               for f in find_recordings(a.train_dir)}
    train_keys = full['source_recording'].map(walk_stamp)
    users = train_keys.map(user_of)

    all_classes = set(full['surface'])

    cnnX = cnnY = cnnR = cnn_stamps = None
    _fit_label = ['']          # set per walk, so each fit says which walk it is for
    if a.model == 'cnn':
        print('building CNN training windows ...')
        cnnX, cnnY, cnnR = cnn_build_training(ns, a.train_dir)
        cnn_stamps = np.array([walk_stamp(r) for r in cnnR], dtype=object)
        print(f'  {len(cnnX)} windows {dict(pd.Series(cnnY).value_counts())}')

    def fit_all(mask=None):
        sub = full if mask is None else full[mask.values]
        missing = all_classes - set(sub['surface'])
        if missing:
            # a held-out model that never saw a class can never predict it - refuse rather
            # than fall back to the in-sample model, which would hide the leakage again
            return None, sorted(missing)
        # hardness and evenness always come from the feature pipeline; --model only
        # decides what labels the 3-class surface layer
        m2 = train_surface(sub, drop, True, ns)
        hp = train_hardness(sub, a.hardness_weights)
        if a.model == 'cnn':
            if mask is None:
                keep = np.ones(len(cnnY), bool)
            else:
                # the stamp set is built ONCE; computing it inside the comprehension
                # rebuilt it per window (21k windows x 21k rows = minutes of nothing)
                _keep_stamps = set(sub['source_recording'].map(walk_stamp))
                keep = np.fromiter((st in _keep_stamps for st in cnn_stamps),
                                   bool, len(cnn_stamps))
            if not keep.any() or (all_classes - set(cnnY[keep])):
                return None, sorted(all_classes - set(cnnY[keep])) or ['(no windows)']
            m3 = ('cnn',) + cnn_train(ns, cnnX[keep], cnnY[keep],
                                      ns.get('CLASS_WEIGHT_STRENGTH', 0.5),
                                      max_epochs=a.cnn_epochs, label=_fit_label[0])
        else:
            m3 = train_surface(sub, drop, False, ns)
        return (m3, m2, hp), None

    print("training surface models (3-class, 2-class) and hardness P(firm) ...")
    base_models, _ = fit_all()
    if a.model == 'cnn':
        print(f"  3-class model: 1D-CNN on raw windows "
              f"({len(ns['SEQ_CHANNELS'])} channels x {ns['T_LEN']} samples), "
              f"class-weight strength {ns.get('CLASS_WEIGHT_STRENGTH', 0.5)}")
    else:
        print(f"  3-class model: {len(base_models[0][2])} features, XGBoost "
              f"{XGB_PARAMS['n_estimators']} trees depth {XGB_PARAMS['max_depth']}, "
              f"class-weight strength {ns.get('CLASS_WEIGHT_STRENGTH', 0.5)}")
    print(f"  hardness weights: {a.hardness_weights}   holdout: {a.holdout}")
    if a.model == 'cnn' and a.holdout != 'none':
        _nw = len(glob.glob(os.path.join(a.walks_dir, '*.gpx')))
        print(f"  NOTE: --holdout {a.holdout} refits the CNN once per walk ({_nw} walks). "
              f"Each fit is silent until it finishes; use --cnn-epochs to cap it.")

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

    def lab(pdf, model, Xseq=None):
        if model[0] == 'cnn':
            _, m, le = model
            if Xseq is None or not len(Xseq):
                return np.array([], dtype=object)
            return le.inverse_transform(cnn_predict(ns, m, Xseq))
        m, le, feats = model
        X = pdf.reindex(columns=feats).fillna(0).values
        return le.inverse_transform(m.predict(X))

    def hard(pdf, hp):
        pipe, compl = hp
        return pipe.predict_proba(pdf.reindex(columns=compl).fillna(0).values)[:, 1] * 100

    info = dict(segmentation=seg, gc_dir=ns.get('GC_DIR'), holdout=a.holdout,
                hardness_weights=a.hardness_weights, xgb=XGB_PARAMS, model=a.model,
                class_weight_strength=ns.get('CLASS_WEIGHT_STRENGTH', 0.5),
                n_features_3class=(None if a.model == 'cnn' else len(base_models[0][2])), evenness_scaler=src or 'fitted here',
                step_time='initial contact', walks={})
    for gpx in sorted(glob.glob(os.path.join(a.walks_dir, '*.gpx'))):
        stem = os.path.splitext(os.path.basename(gpx))[0]; pts = parse_gpx(gpx)
        if len(pts) < 2: print(f"[skip] {stem}: <2 pts"); continue
        # The WALK copy is what gets predicted. Prefer the stamp-matched file in the walks
        # folder; fall back to time-overlap only if the naming does not match.
        stamp = walk_stamp(gpx)
        rec = find_walk_csv(stamp, a.walks_dir) or paired_gps_csv(pts, a.walks_dir)
        if not rec:
            print(f"[warn] {stem}: no walk CSV found for {stamp} in {a.walks_dir}"); continue
        if seg == 'events' and not has_events(ns, rec):
            print(f"[skip] {stem}: no contact-event file for {stamp} - not predicting with a "
                  "different segmentation"); continue
        # The training twin is the SAME recording, so holding the walk out must drop it.
        in_train = stamp in set(train_keys); u = user_of.get(stamp)
        if a.holdout == 'walk' and in_train:
            _fit_label[0] = f' for {stem} (walk held out)'
            models, missing = fit_all(train_keys != stamp); how = 'held out (walk)'
        elif a.holdout == 'user' and u is not None:
            _fit_label[0] = f' for {stem} (user {u} held out)'
            models, missing = fit_all(users != u); how = f'held out (user {u})'
        else:
            missing = None
            models = base_models
            how = ('IN-SAMPLE (this recording is also in the training set)' if in_train
                   else 'not in training')
        if missing:
            print(f"[skip] {stem}: holding it out leaves no {missing} steps in training, so "
                  "that model could never predict them - not written"); continue
        pdf = extract_features(ns, rec)
        if not len(pdf): print(f"[warn] {stem}: 0 steps from {os.path.basename(rec)}"); continue
        t = pd.to_numeric(pdf['t_ms'], errors='coerce').values
        Xw = tw = None
        if a.model == 'cnn':
            Xw, _, tw = cnn_windows(ns, rec, labelled_only=False)
            if not len(Xw):
                print(f"[warn] {stem}: 0 CNN windows - skipping"); continue

        def write(vals, tag, keyname, fill):
            mv = match_to_track(pts, list(zip(t, vals)), fill=fill)
            json.dump(seg_fc(pts, mv, keyname), open(os.path.join(a.out, f"{stem}_{tag}.geojson"), 'w'))
        # NOTE: the ground-truth layer (<stem>_tagged.geojson) is written by
        # build_all_walks.py, which reads the same GPS/REC_*_GPS.csv and tags each GPS fix
        # directly. Writing it here too would put two differently-derived versions under
        # one name in the same folder. walk_truth() is kept for callers that want the
        # per-step truth (fill_gis_table.py).
        if a.model == 'cnn':
            p3 = lab(pdf, models[0], Xw)
            mv = match_to_track(pts, list(zip(tw, p3)), fill='unknown')
            json.dump(seg_fc(pts, mv, 'surface'),
                      open(os.path.join(a.out, f'{stem}_pred_3class.geojson'), 'w'))
        else:
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
        info['walks'][stem] = dict(walk_csv=rec, in_training=in_train, fitted=how,
                                   steps=int(len(pdf)),
                                   cnn_windows=(int(len(Xw)) if Xw is not None else None))
        print(f"[ok] {stem}: {len(pdf)} steps from {os.path.basename(rec)} "
              f"(walk copy) | model {a.model} | {how}")
    json.dump(info, open(os.path.join(a.out, 'predict_walks_run.json'), 'w'), indent=1, default=str)
    print(f"\nDone -> geojsons in '{a.out}' (settings in predict_walks_run.json).")


if __name__ == '__main__':
    main()
