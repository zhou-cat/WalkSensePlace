"""
predict_walks.py — per-walk step-by-step lines for the maps, time-matched to each GPX:
    <stem>_pred_3class.geojson   grass/loose/paved   (3-class model)      [categorical]
    <stem>_pred_2class.geojson   soft/hard           (2-class model)      [categorical]
    <stem>_hardness.geojson      P(firm)x100         (notebook cell-38)   [continuous 0-100]
    <stem>_evenness.geojson      evenness 0-100 (standalone unsupervised model / evenness_scaler.json)
build_all_walks.py draws these automatically (gradients for hardness/evenness).

Hardness = supervised P(firm): logistic regression on compliance features, the SAME method as
the model notebooks' hardness/evenness cell. Evenness comes from the standalone unsupervised
scoring model (focused extractor + robust median/IQR + normal-CDF calibration); it loads
evenness_scaler.json if present, else fits the same scaler on the training recordings.

Usage:
    pip install xgboost scikit-learn scipy pandas numpy gpxpy
    python predict_walks.py --notebook ground_condition_xgboost_v18_merged_3class.ipynb \
        --train-dir "C:/Users/Sam/WalkSensePlace/Individual Users" \
        --walks-dir "C:/Users/Sam/WalkSensePlace/GPS"
"""
import argparse, os, glob, json, sys
import numpy as np, pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime

META = {'foot','step_id','surface','contact_time_sec','source_user','source_recording','t_ms'}
SOFT = {'grass','soft'}
COMPL_FEATS = ['accel_jerk_ratio','accel_hf_frac','accel_spec_centroid','accel_zcr',
               'gyro_hf_frac','gu_stance_frac','stance_duration','loading_rate_norm']
EVEN_NEG_FEATS = ['orient_range_a','orient_range_b','gu_cop_wander','gu_press_jitter',
                  'orient_rest_a_roll3_std','orient_rest_b_roll3_std','gyro_x_std']

# ---- standalone unsupervised EVENNESS model (matches evenness_score_unsupervised.ipynb) ----
from scipy.signal import find_peaks
from scipy.stats import norm
EV_FS=62.5; EV_MIN,EV_MAX=15,150
EV_SENSORS=[f'pressure_{i:02d}' for i in range(1,13)]
EV_WITHIN=['orient_range_a','orient_range_b','cop_wander','press_jitter','gyro_std']
EV_STEP=['stance_dur','loading_rate','impact','tiltA','tiltB']
EV_FEATS=EV_WITHIN+[f'{c}_roll3_std' for c in EV_STEP]

def ev_find_steps(tot):
    mx=np.nanmax(tot) if np.nanmax(tot)>0 else 1.0
    pk,_=find_peaks(tot,distance=25,prominence=mx*0.15); return pk

def ev_extract_sole(df,foot):
    df=df.reset_index(drop=True)
    if len(df)<EV_MIN: return pd.DataFrame()
    P=df[EV_SENSORS].apply(pd.to_numeric,errors='coerce').fillna(0).values; tot=P.sum(1)
    ax=pd.to_numeric(df['accel_x'],errors='coerce').fillna(0).values
    ay=pd.to_numeric(df['accel_y'],errors='coerce').fillna(0).values
    az=pd.to_numeric(df['accel_z'],errors='coerce').fillna(0).values
    G=df[['gyro_x','gyro_y','gyro_z']].apply(pd.to_numeric,errors='coerce').fillna(0).values; Gc=G-np.median(G,0)
    amag=np.sqrt(ax**2+ay**2+az**2); gmag=np.sqrt((Gc**2).sum(1))
    tA=np.degrees(np.arctan2(ax,np.hypot(ay,az))); tB=np.degrees(np.arctan2(az,np.hypot(ax,ay)))
    ts=pd.to_numeric(df['timestamp'],errors='coerce').values if 'timestamp' in df.columns else np.arange(len(df))
    pk=ev_find_steps(tot); rows=[]; o=0
    for i in range(len(pk)-1):
        a,b=pk[i],pk[i+1]; n=b-a
        if n<EV_MIN or n>EV_MAX: continue
        Pw=P[a:b]; totw=tot[a:b]; sc=totw.mean()+1e-9; wam=amag[a:b]; wgm=gmag[a:b]
        Qn=Pw/(Pw.sum(1,keepdims=True)+1e-9); t=int(np.argmin(wgm))
        rows.append(dict(foot=foot,order=o,t_ms=float(ts[(a+b)//2]),
            orient_range_a=float(tA[a:b].max()-tA[a:b].min()), orient_range_b=float(tB[a:b].max()-tB[a:b].min()),
            cop_wander=float(np.mean(np.abs(np.diff(Qn,axis=0)).sum(1))) if n>1 else 0.0,
            press_jitter=float(np.mean(np.std(np.diff(Pw,n=2,axis=0),axis=0))/sc) if n>2 else 0.0,
            gyro_std=float(np.mean([np.std(Gc[a:b,k]) for k in range(3)])),
            stance_dur=float((totw>0.5*sc).mean()), loading_rate=float(np.max(np.diff(totw))/sc),
            impact=float(wam[:max(1,n//3)].max()/(np.median(wam)+1e-9)), tiltA=float(tA[a:b][t]), tiltB=float(tB[a:b][t]))); o+=1
    return pd.DataFrame(rows)

def ev_extract(csv_path):
    raw=pd.read_csv(csv_path,low_memory=False)
    if 'sole_id' not in raw.columns: return pd.DataFrame()
    parts=[]
    for sole in raw['sole_id'].dropna().unique():
        foot='right' if str(sole) in ('1','1.0') else 'left' if str(sole) in ('2','2.0') else f'sole{sole}'
        st=ev_extract_sole(raw[raw['sole_id']==sole],foot)
        if len(st): parts.append(st)
    if not parts: return pd.DataFrame()
    S=pd.concat(parts,ignore_index=True).sort_values(['foot','order'])
    for c in EV_STEP:
        S[f'{c}_roll3_std']=S.groupby('foot')[c].transform(lambda x:x.rolling(3,min_periods=1).std().fillna(0))
    return S

def _robust_cs(x):
    m=float(np.median(x)); q=float(np.subtract(*np.percentile(x,[75,25])))
    return m,(q if q>0 else (float(np.std(x)) or 1.0))

def fit_even_scaler(df):
    cs={c:list(_robust_cs(df[c].values)) for c in EV_FEATS if c in df.columns}
    Z=np.column_stack([np.clip((df[c].values-cs[c][0])/(cs[c][1]+1e-9),-4,4) for c in cs]); U=Z.mean(1)
    um,us=_robust_cs(U); return {'features':list(cs.keys()),'feat_center_scale':cs,'U_center':um,'U_scale':us}

def apply_even_scaler(df,scaler):
    feats=scaler['features']; cs=scaler['feat_center_scale']
    for c in feats:
        if c not in df: df[c]=0.0
    Z=np.column_stack([np.clip((df[c].values-cs[c][0])/(cs[c][1]+1e-9),-4,4) for c in feats]); U=Z.mean(1)
    return np.clip(100*norm.cdf(-(U-scaler['U_center'])/(scaler['U_scale']+1e-9)),0,100)

def find_even_scaler(explicit, train_dir, walks_dir):
    cands=[explicit] if explicit else []
    for d in [train_dir, walks_dir]:
        cands += [os.path.join(d,'..','Charts_Graphs','evenness_scaler.json'),
                  os.path.join(d,'Charts_Graphs','evenness_scaler.json'),
                  os.path.join(d,'evenness_scaler.json')]
    for d in [train_dir, walks_dir]:
        cands += glob.glob(os.path.join(d,'..','**','evenness_scaler.json'), recursive=True)
    for c in cands:
        if c and os.path.exists(c):
            try: return json.load(open(c, encoding='utf-8')), c
            except Exception: pass
    return None, None

def load_extractor(nb_path):
    nb = json.load(open(nb_path, encoding='utf-8')); S = lambda c: ''.join(c['source'])
    codes = [S(c) for c in nb['cells'] if c['cell_type'] == 'code']
    cfg     = next(s for s in codes if 'DROP_FEATURES =' in s)
    helpers = next(s for s in codes if 'def label_surface_segments' in s)
    extr    = next(s for s in codes if 'def extract_step_features' in s)
    extr = extr.replace("        rows.append(row)",
        "        try:\n            row['t_ms'] = float(pd.to_numeric(df['timestamp'], errors='coerce').iloc[(p1+p2)//2])\n"
        "        except Exception:\n            row['t_ms'] = float('nan')\n        rows.append(row)", 1)
    ns = {'__name__': 'wsp_extract'}
    for cell, tag in [(cfg,'cfg'), (helpers,'helpers'), (extr,'extr')]:
        exec(compile(cell, tag, 'exec'), ns)
    ns.setdefault('compute_session_stats', lambda raw: None)
    return ns

def find_recordings(root):
    files = glob.glob(os.path.join(root, '**', 'REC_*.csv'), recursive=True) or \
            glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)
    return sorted(f for f in files if '_GPS' not in os.path.basename(f))

def build_training(ns, train_dir):
    lss, pp, extract = ns['label_surface_segments'], ns['preprocess'], ns['extract_step_features']
    css = ns.get('compute_session_stats', lambda r: None); frames=[]
    for f in find_recordings(train_dir):
        try: raw = pd.read_csv(f, low_memory=False)
        except Exception: continue
        if 'sole_id' not in raw.columns: continue
        ss = css(raw)
        for sole, foot in [(1,'right'),(2,'left')]:
            sub = raw[raw['sole_id']==sole].copy(); lab = lss(sub)
            if lab is None or len(lab)==0: continue
            fe = extract(pp(lab), foot_label=foot, session_stats=ss)
            if len(fe): fe['source_recording']=os.path.basename(f); frames.append(fe)
    if not frames: sys.exit(f"No labelled steps from {train_dir}.")
    full = pd.concat(frames, ignore_index=True).dropna(subset=['surface'])
    return full[full['surface']!='unknown']

def make_model():
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import Pipeline
        return Pipeline([('s',StandardScaler()),('x',XGBClassifier(n_estimators=350,learning_rate=0.05,max_depth=5,
                subsample=0.8,colsample_bytree=0.8,min_child_weight=3,eval_metric='mlogloss',random_state=42,n_jobs=-1,verbosity=0))])
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=5,learning_rate=0.05,max_iter=350,random_state=42)

def train_surface(full, drop, remap2=False):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_sample_weight
    df=full.copy()
    if remap2: df['surface']=np.where(df['surface']=='grass','soft','hard')
    feats=[c for c in df.columns if c not in META and c not in set(drop)]
    X=df[feats].fillna(0).values; le=LabelEncoder(); y=le.fit_transform(df['surface'].values); m=make_model()
    try: m.fit(X,y,**({f'{m.steps[-1][0]}__sample_weight':compute_sample_weight("balanced",y)} if hasattr(m,'steps') else {'sample_weight':compute_sample_weight("balanced",y)}))
    except Exception: m.fit(X,y)
    return m, le, feats

def extract_features(ns, csv_path):
    lss, pp, extract = ns['label_surface_segments'], ns['preprocess'], ns['extract_step_features']
    css = ns.get('compute_session_stats', lambda r: None)
    raw = pd.read_csv(csv_path, low_memory=False); ss = css(raw); frames=[]
    if 'sole_id' not in raw.columns: return pd.DataFrame()
    for s in raw['sole_id'].dropna().unique():
        foot = 'right' if str(s) in ('1','1.0') else 'left' if str(s) in ('2','2.0') else f'sole{s}'
        sub = raw[raw['sole_id']==s].copy(); lab = lss(sub); dfp = pp(lab if lab is not None else sub)
        fe = extract(dfp, foot_label=foot, session_stats=ss)
        if len(fe): frames.append(fe)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def parse_gpx(path):
    root=ET.parse(path).getroot(); strip=lambda t:t.split('}')[-1]; pts=[]
    for pt in root.iter():
        if strip(pt.tag)!='trkpt': continue
        lat=float(pt.attrib['lat']); lon=float(pt.attrib['lon']); t=None
        for c in pt:
            if strip(c.tag)=='time': t=(c.text or '').strip()
        if not t: continue
        pts.append({'lon':lon,'lat':lat,'tms':int(datetime.fromisoformat(t.replace('Z','+00:00')).timestamp()*1000)})
    return pts

def match_to_track(pts, step_vals, max_gap_ms=1500, fill='unknown'):
    if not step_vals: return [fill]*len(pts)
    st=np.array([s[0] for s in step_vals],float); v=[s[1] for s in step_vals]
    o=np.argsort(st); st=st[o]; v=[v[i] for i in o]; out=[]
    for p in pts:
        j=int(np.searchsorted(st,p['tms'])); cand=[k for k in (j-1,j) if 0<=k<len(st)]
        if not cand: out.append(fill); continue
        k=min(cand,key=lambda k:abs(st[k]-p['tms']))
        out.append(v[k] if abs(st[k]-p['tms'])<=max_gap_ms else fill)
    return out

def seg_fc(pts, vals, key):
    feats=[]
    for i in range(len(pts)-1):
        val=vals[i]
        if isinstance(val,float) and np.isnan(val): val=None
        feats.append({'type':'Feature','properties':{key:val},
            'geometry':{'type':'LineString','coordinates':[[pts[i]['lon'],pts[i]['lat']],[pts[i+1]['lon'],pts[i+1]['lat']]]}})
    return {'type':'FeatureCollection','features':feats}

def paired_gps_csv(gpx_pts, walks_dir):
    if not gpx_pts: return None
    g0,g1=gpx_pts[0]['tms'],gpx_pts[-1]['tms']; best,ov=None,0
    for f in glob.glob(os.path.join(walks_dir,'*.csv')):
        try: ts=pd.to_numeric(pd.read_csv(f,usecols=['timestamp'])['timestamp'],errors='coerce').dropna()
        except Exception: continue
        if not len(ts): continue
        o=max(0,min(g1,int(ts.max()))-max(g0,int(ts.min())))
        if o>ov: ov,best=o,f
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--notebook',required=True); ap.add_argument('--train-dir',required=True)
    ap.add_argument('--walks-dir',default='.'); ap.add_argument('--pred-dir',default=None); ap.add_argument('--out',default=None)
    ap.add_argument('--evenness-scaler',default=None,help='path to evenness_scaler.json (else auto-find, else fit on training)')
    a=ap.parse_args(); a.out=a.out or os.path.join(a.walks_dir,'walk_maps'); os.makedirs(a.out,exist_ok=True)

    print("loading extractor ..."); ns=load_extractor(a.notebook)
    print("building training features ..."); full=build_training(ns,a.train_dir)
    print(f"  {len(full)} labelled steps {dict(full.surface.value_counts())}")
    drop=ns.get('DROP_FEATURES',[])
    print("training surface models (3-class, 2-class) ...")
    m3,le3,f3=train_surface(full,drop,False); m2,le2,f2=train_surface(full,drop,True)

    # --- notebook cell-38 hardness = P(firm) + evenness = EVEN_NEG composite ---
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler; from sklearn.pipeline import Pipeline
    COMPL=[c for c in COMPL_FEATS if c in full.columns]
    hard_y=(~full['surface'].isin(SOFT)).astype(int).values
    hard_pipe=Pipeline([('s',StandardScaler()),('lr',LogisticRegression(max_iter=1000,class_weight='balanced'))]).fit(full[COMPL].fillna(0).values,hard_y)
    print("training hardness P(firm) [notebook method] ...")

    # evenness = standalone unsupervised model: load evenness_scaler.json, else fit the same scaler on training
    even_scaler,src=find_even_scaler(a.evenness_scaler,a.train_dir,a.walks_dir)
    if even_scaler:
        print(f"evenness: using scaler from {src}")
    else:
        print("evenness: no scaler found -> fitting unsupervised scaler on training recordings ...")
        ev_tr=[ev_extract(f) for f in find_recordings(a.train_dir)]
        ev_tr=pd.concat([d for d in ev_tr if len(d)],ignore_index=True) if any(len(d) for d in ev_tr) else pd.DataFrame()
        if not len(ev_tr): sys.exit("could not build evenness training features")
        even_scaler=fit_even_scaler(ev_tr)

    def lab(pdf,m,le,feats):
        for c in feats:
            if c not in pdf: pdf[c]=0.0
        return le.inverse_transform(m.predict(pdf[feats].fillna(0).values))
    def hard(pdf):
        for c in COMPL:
            if c not in pdf: pdf[c]=0.0
        return hard_pipe.predict_proba(pdf[COMPL].fillna(0).values)[:,1]*100

    for gpx in sorted(glob.glob(os.path.join(a.walks_dir,'*.gpx'))):
        stem=os.path.splitext(os.path.basename(gpx))[0]; pts=parse_gpx(gpx)
        if len(pts)<2: print(f"[skip] {stem}: <2 pts"); continue
        gps=paired_gps_csv(pts,a.walks_dir)
        base=os.path.basename(gps).replace('_GPS','') if gps else None
        search=([os.path.join(os.path.dirname(gps),base)] if gps else [])
        for d in [a.pred_dir, a.walks_dir]:          # walk recordings live in GPS, not Individual Users
            if d and base: search+=glob.glob(os.path.join(d,'**',base),recursive=True)
        rec=next((c for c in search if c and os.path.exists(c)), gps)
        if not rec: print(f"[warn] {stem}: no recording found"); continue
        if rec==gps: print(f"[note] {stem}: predicting on {os.path.basename(gps)} (raw not found)")
        pdf=extract_features(ns,rec)
        if not len(pdf): print(f"[warn] {stem}: 0 steps from {os.path.basename(rec)}"); continue
        t=pd.to_numeric(pdf['t_ms'],errors='coerce').values
        def write(vals,tag,key,fill):
            mv=match_to_track(pts,list(zip(t,vals)),fill=fill)
            json.dump(seg_fc(pts,mv,key),open(os.path.join(a.out,f"{stem}_{tag}.geojson"),'w'))
        write(lab(pdf,m3,le3,f3),'pred_3class','surface','unknown')
        write(lab(pdf,m2,le2,f2),'pred_2class','surface','unknown')
        write(hard(pdf),'hardness','value',float('nan'))
        # evenness: standalone unsupervised model on this walk's recording (own step segmentation)
        evdf=ev_extract(rec)
        if len(evdf):
            ev_vals=apply_even_scaler(evdf,even_scaler); ev_t=pd.to_numeric(evdf['t_ms'],errors='coerce').values
            evm=match_to_track(pts,list(zip(ev_t,ev_vals)),fill=float('nan'))
            json.dump(seg_fc(pts,evm,'value'),open(os.path.join(a.out,f"{stem}_evenness.geojson"),'w'))
        else:
            print(f"[warn] {stem}: 0 evenness steps")
        print(f"[ok] {stem}: {len(pdf)} steps -> pred_3class, pred_2class, hardness, evenness")
    print(f"\nDone -> geojsons in '{a.out}'. Run build_all_walks.py to draw the lines.")

if __name__=='__main__':
    main()
