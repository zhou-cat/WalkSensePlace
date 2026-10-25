"""Fill the per-step agreement table: GIS label vs 3-class model vs ground truth.

    python scripts/fill_gis_table.py --notebook notebooks/ground_condition_xgboost_v21_merged_3class.ipynb ^
        --train-dir "Individual Users" --walks-dir GPS --gis-dir GPS/walk_maps_satellite

Uses predict_walks.py's own extractor, training set and model, so the "as mapped" model is
the one the walk maps show. The model is scored three ways:

  as mapped            trained on every recording, INCLUDING the walk being scored - the
                       walk-map setting. Scoring a model on its own training steps measures
                       how well it fits them, not how well it labels a new walk.
  leave-one-walk-out   trained on every recording EXCEPT the walk being scored. The
                       walker's other walks stay in, so the user is 'in the sample' while
                       the walk is not. This is the honest version of the walk-map setting.
  leave-one-user-out   trained without ANY of the walker's recordings - the paper's main
                       evaluation (Results).

Unit: one insole step. Truth is the step's surface from the inline tags, with the same
step gate as training; the model labels the step; the GIS label is read at the GPS fix
nearest the step's initial contact (within 1.5 s, as the maps do). Both sources are
scored on the SAME steps. Steps with no GPS fix within 1.5 s are excluded from both.

Writes gis_table_per_walk.csv, gis_table_pooled.csv and gis_table_rows.tex.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_walks as pw  # noqa: E402

CL3 = ['grass', 'loose', 'paved']
MODES = ['as mapped', 'leave-one-walk-out', 'leave-one-user-out']
TRUTH = 'Ground truth'          # composition only; it has no agreement with itself


def metrics(yt, yp):
    from sklearn.metrics import f1_score
    yt = np.asarray(yt, dtype=object); yp = np.asarray(yp, dtype=object)
    if not len(yt):
        return dict(n=0)
    f1 = f1_score(yt, yp, labels=CL3, average=None, zero_division=0)
    # Composition over the SAME scored steps as the agreement, so every row shares one
    # denominator and the rows are directly comparable. (make_map_prop.py instead counts
    # each layer's own segments, so its denominators differ between sources.) Shares do not
    # sum to 100 where a source leaves steps unlabelled; that remainder is unlabeled_pct.
    share = {f'share_{c}': float(100 * np.mean(yp == c)) for c in CL3}
    return dict(n=len(yt), accuracy=float((yt == yp).mean()), macro_f1=float(f1.mean()),
                f1_grass=float(f1[0]), f1_loose=float(f1[1]), f1_paved=float(f1[2]),
                unlabeled_pct=float(100 * np.mean(yp == 'unknown')), **share)


def gis_at_steps(pts, gis_seg_vals, step_t, max_gap_ms=1500):
    """GIS label at the GPS fix nearest each step (within max_gap_ms), else None.
    gis_seg_vals[i] is the label of segment i, which starts at GPS point i."""
    pv = list(gis_seg_vals) + ([gis_seg_vals[-1]] if gis_seg_vals else [])   # per point
    vals = pw.match_to_track([{'tms': t} for t in step_t],
                             [(p['tms'], v) for p, v in zip(pts, pv)],
                             max_gap_ms=max_gap_ms, fill=None)
    return vals


def predict(model, pdf, Xseq=None):
    """Labels from either model kind. The CNN's tuple is ('cnn', model, encoder)."""
    if model[0] == 'cnn':
        _, m, le = model
        if Xseq is None or not len(Xseq):
            return None
        return le.inverse_transform(pw.cnn_predict(_NS, m, Xseq))
    m, le, feats = model
    X = pdf.copy()
    for c in feats:
        if c not in X:
            X[c] = 0.0
    return le.inverse_transform(m.predict(X[feats].fillna(0).values))


_NS = {}          # the extractor namespace, needed by predict() for the CNN device


def rec_key(path):
    """A recording's name with any '_GPS' suffix removed: REC_x_GPS.csv -> REC_x.csv.
    The walks folder holds the _GPS copies, the training folder the unsuffixed ones, so
    recordings are matched on this key - otherwise a walk never matches its own training
    recording, leave-one-walk-out silently equals 'as mapped', and the user is unknown."""
    b = os.path.basename(path)
    stem, ext = os.path.splitext(b)
    return (stem[:-4] if stem.endswith('_GPS') else stem) + ext


def _unused_find_walk_recording(gpx_pts, walks_dir, pred_dir, train_dir):
    """The insole recording for a walk: its unsuffixed REC_x.csv, searched in --pred-dir,
    --train-dir and the walks folder; falls back to the paired REC_x_GPS.csv."""
    gps = pw.paired_gps_csv(gpx_pts, walks_dir)
    if not gps:
        return None
    base = rec_key(gps)
    search = [os.path.join(os.path.dirname(gps), base)]
    for d in [pred_dir, train_dir, walks_dir]:
        if d:
            search += glob.glob(os.path.join(d, '**', base), recursive=True)
    return next((c for c in search if c and os.path.exists(c)), gps)


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '--'
    return f'{x:.1f}' if pct else f'{x:.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--notebook', required=True, help='same notebook predict_walks.py uses')
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--walks-dir', default='GPS')
    ap.add_argument('--pred-dir', default=None, help='as in predict_walks.py')
    ap.add_argument('--gis-dir', default=None,
                    help='folder with <stem>_surface.geojson (default <walks-dir>/walk_maps_satellite)')
    ap.add_argument('--out', default=None, help='output folder (default --gis-dir)')
    ap.add_argument('--model', choices=['xgboost', 'cnn'], default='xgboost',
                    help='which model the table scores (the GIS row is unaffected)')
    ap.add_argument('--cnn-notebook', default=None,
                    help='the CNN comparison notebook (required with --model cnn)')
    ap.add_argument('--cnn-epochs', type=int, default=None,
                    help="cap the CNN's epochs; it is refit once per regime per walk")
    a = ap.parse_args()
    a.gis_dir = a.gis_dir or os.path.join(a.walks_dir, 'walk_maps_satellite')
    a.out = a.out or a.gis_dir

    # which reach built the GIS labels? (written by build_all_walks_satellite.py)
    TRIM = {}
    summ = os.path.join(a.gis_dir, 'summary.csv')
    if os.path.exists(summ):
        sm = pd.read_csv(summ)
        if {'walk', 'trim_start', 'trim_end'} <= set(sm.columns):
            # build_all_walks_satellite.py trims each track to the span where the ground
            # truth is known, so its geojsons cover GPX points [trim_start : trim_end+1].
            # Without applying the same slice, the segment counts disagree and every
            # trimmed walk looks like a GPX mismatch.
            TRIM = {str(r.walk): (int(r.trim_start), int(r.trim_end))
                    for r in sm.itertuples() if pd.notna(r.trim_start)}
            _nt = sum(1 for v in TRIM.values() if v[0] or v[1])
            if _nt:
                print(f'  {_nt} walk(s) were trimmed; applying the same slice here')
        rc = [c for c in sm.columns if c.startswith('reach_')]
        print('GIS labels built with reach (m): ' +
              (', '.join(f"{c[6:-2]} {sm[c].iloc[0]:g}" for c in rc) if rc else
               'NOT RECORDED - built before the shared reach; rebuild first'))
    else:
        print(f'[warning] no summary.csv in {a.gis_dir}: cannot confirm how the GIS labels were built')

    global _NS
    print('loading extractor ...'); ns = pw.load_extractor(a.notebook); _NS = ns
    if a.model == 'cnn':
        if not a.cnn_notebook:
            sys.exit('--model cnn requires --cnn-notebook')
        print('loading CNN ...'); pw.load_cnn(a.cnn_notebook, ns)
        print(f"  CNN on {ns.get('device', 'cpu')}")
    print('building training features ...'); full = pw.build_training(ns, a.train_dir)
    user_of = {pw.walk_stamp(f): os.path.basename(os.path.dirname(f))
               for f in pw.find_recordings(a.train_dir)}
    # Kept OUTSIDE the training table: train_surface treats every column not in META as a
    # feature, so a user column inside it would be trained on (or crash, being text).
    train_keys = full['source_recording'].map(pw.walk_stamp)
    users = train_keys.map(user_of)
    drop = ns.get('DROP_FEATURES', [])
    print(f'  {len(full)} labelled steps, {full.source_recording.nunique()} recordings, '
          f'users {sorted(users.dropna().unique())}')
    all_classes = set(full['surface'])
    cnnX = cnnY = cnn_stamps = None
    if a.model == 'cnn':
        print('building CNN training windows ...')
        cnnX, cnnY, cnnR = pw.cnn_build_training(ns, a.train_dir)
        cnn_stamps = np.array([pw.walk_stamp(r) for r in cnnR], dtype=object)
        print(f'  {len(cnnX)} windows')

    def fit(mask, what):
        """Fit the chosen model on the masked training set, or None if a class is absent."""
        sub = full if mask is None else full[mask.values]
        missing = sorted(all_classes - set(sub['surface']))
        if not len(sub) or missing:
            print(f'    [{what}] skipped: holding out leaves '
                  + ('no training data' if not len(sub) else f'no {missing} steps'))
            return None
        if a.model == 'cnn':
            stamps = set(sub['source_recording'].map(pw.walk_stamp))
            keep = np.fromiter((st in stamps for st in cnn_stamps), bool, len(cnn_stamps))
            if not keep.any() or (all_classes - set(cnnY[keep])):
                print(f'    [{what}] skipped: CNN windows missing a class')
                return None
            return ('cnn',) + pw.cnn_train(ns, cnnX[keep], cnnY[keep],
                                           ns.get('CLASS_WEIGHT_STRENGTH', 0.5),
                                           max_epochs=a.cnn_epochs, label=f' [{what}]')
        return pw.train_surface(sub, drop, False)

    m_all = fit(None, 'as mapped')

    def _old_held_out(mask, what):
        # a held-out training set that is empty, or lacks a class, gives a model that can
        # never predict that class: report and skip that row rather than score it
        sub = full[mask.values]
        missing = sorted(all_classes - set(sub['surface']))
        if not len(sub) or missing:
            print(f'    [{what}] skipped: holding out leaves '
                  + ('no training data' if not len(sub) else f'no {missing} steps'))
            return None
        return pw.train_surface(sub, drop, False)

    per_walk, pooled = [], {k: ([], []) for k in [TRUTH, 'GIS'] + MODES}
    for gpx in sorted(glob.glob(os.path.join(a.walks_dir, '*.gpx'))):
        stem = os.path.splitext(os.path.basename(gpx))[0]
        pts = pw.parse_gpx(gpx)
        _tr = TRIM.get(stem)
        if _tr:
            lo_, hi_ = _tr
            if hi_ + 1 <= len(pts):
                pts = pts[lo_:hi_ + 1]
            else:
                print(f'[warn] {stem}: summary.csv says points {lo_}-{hi_} but the GPX has '
                      f'{len(pts)} - not slicing'); 
        gpath = os.path.join(a.gis_dir, f'{stem}_surface.geojson')
        if len(pts) < 2 or not os.path.exists(gpath):
            print(f'[skip] {stem}: ' + ('<2 GPS points' if len(pts) < 2 else f'no {gpath}'))
            continue
        # the WALK copy (GPS/REC_<stamp>_GPS.csv) is what gets scored; the training copy
        # under Individual Users is for fitting only. They are the same recording with two
        # tagging regimes, so substituting one for the other reads the wrong tags.
        stamp = pw.walk_stamp(gpx)
        rec = pw.find_walk_csv(stamp, a.walks_dir)
        if not rec:
            print(f'[skip] {stem}: no recording found'); continue
        pdf = pw.extract_features(ns, rec)
        if not len(pdf) or 'surface' not in pdf:
            print(f'[skip] {stem}: no tagged steps in {os.path.basename(rec)}'); continue
        pdf = pdf[pdf['surface'].isin(CL3)].reset_index(drop=True)
        t = pd.to_numeric(pdf['t_ms'], errors='coerce').values
        gis_seg = [f['properties'].get('surface') or 'unknown'
                   for f in json.load(open(gpath))['features']]
        if len(gis_seg) != len(pts) - 1:
            print(f'[warn] {stem}: GIS file has {len(gis_seg)} segments for {len(pts)} GPS '
                  f'point(s)'
                  + ('' if _tr else ' and summary.csv records no trim for it')
                  + ' - the layer and the GPX do not line up, skipped'); continue
        g = gis_at_steps(pts, gis_seg, t)
        keep = np.array([v is not None for v in g])
        if not keep.any():
            print(f'[skip] {stem}: no step has a GPS fix within 1.5 s'); continue
        sub = pdf[keep].reset_index(drop=True)
        yt = sub['surface'].values
        gis = np.array([v for v in g if v is not None], dtype=object)

        rb = stamp; u = user_of.get(rb)
        in_train = rb in set(train_keys)
        Xw = None
        if a.model == 'cnn':
            Xw_all, _, tw = pw.cnn_windows(ns, rec, labelled_only=False)
            # keep only the windows that survived the same step filter as `sub`
            Xw = Xw_all[:len(sub)] if len(Xw_all) >= len(sub) else None
            if Xw is None:
                print(f'    [warn] {stem}: {len(Xw_all)} CNN windows for {len(sub)} steps '
                      '- cannot align, skipping the CNN rows'); 
        preds = {'as mapped': predict(m_all, sub, Xw) if m_all else None}
        if in_train:
            mw = fit(train_keys != rb, 'leave-one-walk-out')
            preds['leave-one-walk-out'] = predict(mw, sub, Xw) if mw else None
        else:
            preds['leave-one-walk-out'] = preds['as mapped']   # walk never trained on
        mu = fit(users != u, 'leave-one-user-out') if u is not None else None
        preds['leave-one-user-out'] = predict(mu, sub, Xw) if mu else None
        print(f'[ok] {stem}: {len(sub)} steps ({int((~keep).sum())} without GPS) | '
              f'features from {rec} | recording {rb} {"IS" if in_train else "is NOT"} in training'
              + (f', user {u}' if u else ', user unknown - no leave-one-user-out'))

        for src, yp in [('Ground truth', yt), ('GIS', gis)] + [(m, preds[m]) for m in MODES]:
            if yp is None or len(yp) != len(yt):
                continue
            per_walk.append(dict(walk=stem, source=src, in_training=in_train, **metrics(yt, yp)))
            pooled[src][0].extend(yt); pooled[src][1].extend(yp)

    if not per_walk:
        sys.exit('no walks evaluated')
    pw_df = pd.DataFrame(per_walk)
    po_df = pd.DataFrame([dict(source=k, **metrics(*v)) for k, v in pooled.items() if v[0]])
    os.makedirs(a.out, exist_ok=True)
    pw_df.to_csv(os.path.join(a.out, 'gis_table_per_walk.csv'), index=False)
    po_df.to_csv(os.path.join(a.out, 'gis_table_pooled.csv'), index=False)

    label = {TRUTH: 'Ground truth', 'GIS': 'GIS (4 sources)',
             'as mapped': '3-class model, as mapped',
             'leave-one-walk-out': '3-class model, leave-one-walk-out',
             'leave-one-user-out': '3-class model, leave-one-user-out'}
    print(f"\nPooled over {pw_df.walk.nunique()} walks, by step:")
    print(f"  {'source':36s} {'n':>6s} {'acc':>6s} {'mF1':>6s} {'grass':>6s} {'loose':>6s} "
          f"{'paved':>6s} {'unlab%':>7s} | {'%grass':>7s} {'%loose':>7s} {'%paved':>7s}")
    lines = []
    for _, r in po_df.iterrows():
        _is_t = r['source'] == TRUTH
        _ag = ('      --     --     --     --     --      --' if _is_t else
               f"{r['accuracy']:6.3f} {r['macro_f1']:6.3f} {r['f1_grass']:6.3f} "
               f"{r['f1_loose']:6.3f} {r['f1_paved']:6.3f} {r['unlabeled_pct']:7.1f}")
        print(f"  {label[r['source']]:36s} {int(r['n']):6d} {_ag} | "
              f"{r['share_grass']:6.2f}% {r['share_loose']:6.2f}% {r['share_paved']:6.2f}%")
        _cols = (['---'] * 5 if _is_t else
                 [fmt(r['accuracy']), fmt(r['macro_f1']), fmt(r['f1_grass']),
                  fmt(r['f1_loose']), fmt(r['f1_paved'])])
        lines.append(' & '.join([label[r['source']]] + _cols
                     + [f"{r['share_grass']:.2f}\\%", f"{r['share_loose']:.2f}\\%",
                        f"{r['share_paved']:.2f}\\%"]) + ' \\\\')
    _gisrow = po_df[po_df.source == 'GIS']
    if len(_gisrow):
        lines.append('\\midrule')
        lines.append('\\multicolumn{6}{@{}l}{GIS steps left unclassified} & '
                     '\\multicolumn{3}{c}{%.2f\\%%} \\\\'
                     % float(_gisrow.unlabeled_pct.iloc[0]))
    open(os.path.join(a.out, 'gis_table_rows.tex'), 'w').write('\n'.join(lines) + '\n')
    gap = (po_df.set_index('source').macro_f1.get('as mapped', np.nan)
           - po_df.set_index('source').macro_f1.get('leave-one-walk-out', np.nan))
    if gap == gap:
        print(f"\n'as mapped' minus 'leave-one-walk-out' macro-F1: {gap:+.3f}. That gap is what "
              "the model gains from having trained on the walk it is scored on.")
    print(f"tables -> {a.out}/gis_table_pooled.csv, gis_table_per_walk.csv, gis_table_rows.tex")


if __name__ == '__main__':
    main()
