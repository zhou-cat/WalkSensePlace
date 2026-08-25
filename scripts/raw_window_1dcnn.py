"""
Ground-up 1D-CNN on RAW step windows (no engineered features).
Builds 5-channel step windows straight from the raw recordings and trains a
1D convolutional net, evaluated BOTH recording-grouped and user-grouped.

Run in an env with torch (e.g. your `opensim` conda env after `pip install torch`,
or any torch env):   python raw_window_1dcnn.py  /path/to/"Individual Users"

Expectation (from every prior test): decent recording-grouped, collapse on loose
user-grouped. This script exists so you can confirm the wall with a real CNN.
"""
import sys, glob, os, numpy as np, pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = sys.argv[1] if len(sys.argv) > 1 else "Individual Users"
FILES = sorted(glob.glob(os.path.join(ROOT, "*", "*.csv")))
ALLS = [f'pressure_{i:02d}' for i in range(1, 13)]
BASE = set(['sole_id','timestamp','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z',
            'magn_x','magn_y','magn_z','corrupt'] + ALLS)
SMAP = {'grass':'grass','dirt':'loose','compact ground':'loose',
        'concrete':'paved','asphalt':'paved','brick':'paved','exposed aggregate':'paved'}
EXC = {'transition','wait'}
HEEL = ['pressure_08','pressure_11']; FORE = ['pressure_02','pressure_04','pressure_05','pressure_06','pressure_09']
L = 40; CLASSES = np.array(['grass','loose','paved'])
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def label_rows(df):
    df = df.sort_values('timestamp').reset_index(drop=True); df['surf'] = None; ev = []
    for c in df.columns:
        if c in BASE: continue
        for idx in df.index[df[c].astype(str).str.strip() == 'x']:
            ev.append((int(idx), 'EXC' if c in EXC else SMAP.get(c)))
    ev = [e for e in ev if e[1]]; ev.sort()
    for k,(i0,lab) in enumerate(ev):
        i1 = ev[k+1][0] if k+1 < len(ev) else len(df); df.loc[i0:i1-1,'surf'] = lab
    return df

def rs(sig): xs = np.linspace(0,1,len(sig)); return np.interp(np.linspace(0,1,L), xs, sig)

X, y, usr, rec = [], [], [], []
for f in FILES:
    df = label_rows(pd.read_csv(f, low_memory=False)); user = os.path.basename(os.path.dirname(f)); rc = os.path.basename(f)
    for sole in [1,2]:
        s = df[df['sole_id'] == sole].reset_index(drop=True)
        if not len(s): continue
        P = s[ALLS].apply(pd.to_numeric, errors='coerce').fillna(0).values; tot = P.sum(1)
        heel = s[HEEL].apply(pd.to_numeric, errors='coerce').sum(1).values
        fore = s[FORE].apply(pd.to_numeric, errors='coerce').sum(1).values
        A = s[['accel_x','accel_y','accel_z']].apply(pd.to_numeric, errors='coerce').fillna(0).values
        G = s[['gyro_x','gyro_y','gyro_z']].apply(pd.to_numeric, errors='coerce').fillna(0).values
        amag = np.sqrt((A**2).sum(1)); gmag = np.sqrt((G**2).sum(1)); surf = s['surf'].values
        mx = np.nanmax(tot) if np.nanmax(tot) > 0 else 1
        pk,_ = find_peaks(tot, distance=25, prominence=mx*0.15)
        for i in range(len(pk)-1):
            a,b = pk[i], pk[i+1]
            if b-a < 15 or b-a > 120: continue
            u = pd.unique(surf[a:b][pd.notna(surf[a:b])])
            if len(u) != 1 or u[0] not in SMAP.values(): continue
            sc = tot[a:b].mean()+1e-9
            X.append(np.stack([rs(tot[a:b])/sc, rs(heel[a:b])/sc, rs(fore[a:b])/sc,
                               rs(amag[a:b])/(np.median(amag[a:b])+1e-9),
                               rs(gmag[a:b])/(np.median(gmag[a:b])+1e-9)]))
            y.append(u[0]); usr.append(user); rec.append(rc)
X = np.array(X, np.float32); y = np.array(y); usr = np.array(usr); rec = np.array(rec)
print(f"{len(X)} steps, shape {X.shape} (5 channels x {L})  users={sorted(set(usr))}")

class CNN(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(5,32,5,padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32,64,5,padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,64,3,padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64,nc))
    def forward(self,x): return self.head(self.net(x))

def run(y_, by):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(y_); yi = le.transform(y_); labs = le.classes_
    g = rec if by == 'rec' else usr; k = 5 if by == 'rec' else 3
    oof = np.empty(len(y_), int)
    for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=42).split(X, yi, g):
        mu, sd = X[tr].mean((0,2), keepdims=True), X[tr].std((0,2), keepdims=True)+1e-6
        Xtr, Xte = (X[tr]-mu)/sd, (X[te]-mu)/sd
        cw = torch.tensor([len(yi[tr])/(len(labs)*max((yi[tr]==c).sum(),1)) for c in range(len(labs))], dtype=torch.float32, device=dev)
        net = CNN(len(labs)).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=cw)
        dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(yi[tr])), batch_size=128, shuffle=True)
        net.train()
        for ep in range(40):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            oof[te] = net(torch.tensor(Xte).to(dev)).argmax(1).cpu().numpy()
    oofs = labs[oof]
    return accuracy_score(y_, oofs), oofs

print("\n=== 1D-CNN on raw windows ===")
for by in ['rec','user']:
    acc, oofs = run(y, by)
    f = f1_score(y, oofs, labels=CLASSES, average=None)*100
    print(f"3-class {by:5s}: acc {acc*100:4.0f}%  grass {f[0]:4.0f}%  loose {f[1]:4.0f}%  paved {f[2]:4.0f}%  macro {f.mean():4.0f}%")
yb = np.where(y=='grass','soft','hard')
for by in ['rec','user']:
    acc, oofs = run(yb, by)
    print(f"grass-vs-hard {by:5s}: acc {acc*100:4.0f}%  soft-F1 {f1_score(yb,oofs,pos_label='soft')*100:4.0f}%")
