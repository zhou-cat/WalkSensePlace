#!/usr/bin/env python3
"""
generate_physicalization.py
Generate 3D-printable foot heightmaps (STL) + previews (PNG) for a PAIR of insole
recordings — one even, one uneven — for the GroundWalks tactile physicalization.

Two walks are required so the texture (roughness) scale is set RELATIVE to the pair:
the more-even walk prints smoother, the less-even prints rougher, on a shared scale.
This self-calibrates the contrast (no fixed evenness bounds needed).

ENCODING per walk
  * BASE height  = mean pressure footprint at each clean step's FLAT-FOOT instant
                   (peak total pressure), pooled across BOTH feet (right mirrored to the
                   left frame via the real sensor coords) -> reads as "a step".
  * TEXTURE roughness = per-zone step-to-step variability of the flat-foot maps, scaled
                   by the walk's evenness irregularity RELATIVE to the pair (floor..1.0).
  * TEXTURE fineness (bump spacing) = per-region two-point tactile threshold
                   (toes fine ~9mm -> heel coarse ~18mm).
STEP GATE (matches classification/evenness pipelines): transition/wait, surface-straddle,
  and <80%-labelled steps dropped; unlabelled recordings kept whole.

USAGE (paths are explicit; run from anywhere)
  python generate_physicalization.py \
      --even  "../Individual Users/sam/REC_even.csv"  --even-size 43 \
      --uneven "../Individual Users/sam/REC_uneven.csv" --uneven-size 44
Options:
  --floor FLOAT   texture scale for the SMOOTHER walk (default 0.15; keeps it non-flat)
  --base-mm FLOAT max base relief mm (default 12)   --tex-mm FLOAT max texture relief mm (default 4)
  --outline PATH  SVG outline override (else embedded traced outline)
  --outdir DIR    output directory (default: current dir)

HONEST LIMITS: only 12 pads -> sub-~3cm detail is interpolated (roughness MAGNITUDE is real);
  sensor coords/outline are proportional not surveyed; STL watertightness NOT verified here
  (check in your slicer); thin-data walks (<30 clean steps) print a warning.
"""
import argparse, csv, os, sys, numpy as np
from scipy.interpolate import Rbf
from scipy.signal import find_peaks

# real sensor coords from cop_coords (index k = pressure_(k+1)); left frame is the template
COORDS_L=np.array([[0.7438,0.8713],[0.7979,0.7165],[0.6264,0.5308],[0.4839,0.7074],[0.6441,0.7103],
 [0.3293,0.6891],[0.2920,0.4243],[0.3119,0.1321],[0.1844,0.6659],[0.5604,0.2574],[0.5396,0.1170],[0.4226,0.8585]])
TEMPLATE=COORDS_L
SOLE={36:(228.9,96.2),37:(236.9,96.2),38:(248.6,96.2),39:(248.7,98.7),40:(256.7,98.7),41:(270.0,98.7),
      42:(271.3,106.3),43:(279.4,106.3),44:(288.6,106.3),45:(290.1,109.7),46:(298.2,109.7),47:(307.2,109.7)}
OUTLINE_EMBED=np.array([(0.5935,0.0285),(0.6630,0.0493),(0.7500,0.0812),(0.8150,0.1120),(0.8676,0.1440),(0.9150,0.1764),(0.9315,0.1905),(0.9671,0.2221),(0.9959,0.2650),(0.9945,0.3119),(0.9998,0.3590),(1.0000,0.3764),(0.9907,0.4146),(0.9706,0.4489),(0.9329,0.4964),(0.8688,0.5643),(0.8560,0.5850),(0.8300,0.6217),(0.8104,0.6539),(0.7901,0.7008),(0.7841,0.7576),(0.7660,0.8146),(0.7551,0.8406),(0.7248,0.8804),(0.6913,0.9075),(0.6387,0.9472),(0.6087,0.9558),(0.5750,0.9655),(0.5216,0.9783),(0.4242,0.9946),(0.3694,1.0000),(0.3301,0.9994),(0.2876,0.9978),(0.2412,0.9923),(0.1863,0.9861),(0.1367,0.9794),(0.1072,0.9686),(0.0888,0.9607),(0.0525,0.9439),(0.0254,0.9248),(0.0085,0.9058),(0.0027,0.8783),(0.0000,0.8576),(0.0052,0.8514),(0.0174,0.8270),(0.0392,0.8049),(0.0606,0.7723),(0.0851,0.7408),(0.1013,0.7201),(0.1248,0.6801),(0.1270,0.6486),(0.1357,0.6182),(0.1509,0.5833),(0.1635,0.5694),(0.1864,0.5373),(0.1746,0.5006),(0.1458,0.4635),(0.1441,0.4455),(0.1328,0.4310),(0.0992,0.4047),(0.0801,0.3815),(0.0426,0.3505),(0.0102,0.3259),(0.0060,0.3047),(0.0000,0.2897),(0.0003,0.2586),(0.0117,0.2073),(0.0375,0.1531),(0.0536,0.1209),(0.0684,0.1054),(0.0997,0.0799),(0.1447,0.0548),(0.1984,0.0293),(0.2435,0.0158),(0.2657,0.0101),(0.3019,0.0037),(0.3452,0.0022),(0.4211,0.0011),(0.4952,0.0121),(0.5454,0.0224)])
ALL=[f'pressure_{k:02d}' for k in range(1,13)]
BASECOLS=set(['sole_id','timestamp','accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','magn_x','magn_y','magn_z','corrupt']+ALL)
SMAP={'grass':'grass','dirt':'loose','compact ground':'loose','compacted ground':'loose','gravel':'loose','sand':'loose',
      'concrete':'continuous_paved','asphalt':'continuous_paved','exposed aggregate':'continuous_paved','brick':'continuous_paved',
      'cobblestone':'modular_paved','uneven brick':'modular_paved','uneven modular paved':'modular_paved','modular paved':'modular_paved'}
EXCLUDE={'transition','wait'}; TMIN=0.80
def _f(r,c):
    try: return float(r[c])
    except: return 0.0
def flatfoot_and_feats(rows, sole):
    sub=[r for r in rows if r.get('sole_id')==sole]
    if not sub: return np.empty((0,12)), []
    P=np.array([[_f(r,c) for c in ALL] for r in sub]); tot=P.sum(1)
    gyr=np.array([[_f(r,'gyro_x'),_f(r,'gyro_y'),_f(r,'gyro_z')] for r in sub])
    acc=np.array([[_f(r,'accel_x'),_f(r,'accel_y'),_f(r,'accel_z')] for r in sub])
    cols=list(sub[0].keys()); surf=np.array([None]*len(sub),dtype=object); events=[]
    for c in cols:
        cl=c.strip().lower()
        if cl in BASECOLS: continue
        lab='__excl__' if cl in EXCLUDE else SMAP.get(cl)
        if lab is None: continue
        for idx in [k for k,r in enumerate(sub) if str(r[c]).strip()=='x']: events.append((idx,lab))
    if events:
        events.sort()
        for k,(i0,lab) in enumerate(events):
            i1=events[k+1][0] if k+1<len(events) else len(sub); surf[i0:i1]=lab
    def kept(a,b):
        ws=surf[a:b]
        if any(v=='__excl__' for v in ws): return False
        if events:
            lab=[v for v in ws if v is not None]
            if len(set(lab))>1 or len(lab)<TMIN*(b-a): return False
        return True
    pk,_=find_peaks(tot,distance=25,prominence=max(tot.max()*0.15,1))
    ff=[]; feats=[]
    for i in range(len(pk)-1):
        a,b=pk[i],pk[i+1]
        if b-a<5 or not kept(a,b): continue
        fi=a+int(np.argmax(tot[a:b])); ff.append(P[fi])
        tA=np.degrees(np.arctan2(acc[fi,0],acc[fi,2]+1e-9)); tB=np.degrees(np.arctan2(acc[fi,1],acc[fi,2]+1e-9))
        feats.append((tA,tB,gyr[a:b,0].std(),b-a))
    return np.array(ff), feats
def irregularity(feats):
    if len(feats)<3: return np.nan
    A=np.array(feats)
    def r3(x): return np.mean([np.std(x[i:i+3]) for i in range(len(x)-2)]) if len(x)>=3 else np.std(x)
    return float(np.mean([r3(A[:,0]),r3(A[:,1]),r3(A[:,2])/10,r3(A[:,3])]))
def pooled(path):
    rows=list(csv.DictReader(open(path)))
    ffL,fL=flatfoot_and_feats(rows,'2'); ffR,fR=flatfoot_and_feats(rows,'1')
    pool=np.vstack([x for x in (ffL,ffR) if len(x)]) if (len(ffL)+len(ffR)) else np.empty((0,12))
    return pool, irregularity(fL+fR), len(ffL), len(ffR)
def load_outline(path):
    if not path: return OUTLINE_EMBED.copy()
    import re, cv2
    raw=open(path).read(); d=re.findall(r'<path[^>]*\bd="([^"]+)"',raw)[0]
    trs=re.findall(r'transform="translate\(([-\d.]+),([-\d.]+)\)"',raw); tx,ty=(float(trs[0][0]),float(trs[0][1])) if trs else (0,0)
    NUM=r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'; toks=re.findall(r'[a-zA-Z]|'+NUM,d)
    pts=[]; cur=np.array([0.0,0.0]); cmd=None; i=0
    def num(i): return float(toks[i]),i+1
    while i<len(toks):
        t=toks[i]
        if t.isalpha(): cmd=t; i+=1
        if i>=len(toks): break
        if cmd in('m','M'): x,i=num(i);y,i=num(i);cur=(cur+[x,y]) if cmd=='m' else np.array([x,y]);pts.append(cur.copy());cmd='l' if cmd=='m' else 'L'
        elif cmd in('l','L'): x,i=num(i);y,i=num(i);cur=(cur+[x,y]) if cmd=='l' else np.array([x,y]);pts.append(cur.copy())
        elif cmd in('v','V'): y,i=num(i);cur=cur+[0,y] if cmd=='v' else np.array([cur[0],y]);pts.append(cur.copy())
        elif cmd in('h','H'): x,i=num(i);cur=cur+[x,0] if cmd=='h' else np.array([x,cur[1]]);pts.append(cur.copy())
        elif cmd in('c','C','s','S','q','Q','t','T'):
            n={'c':6,'s':4,'q':4,'t':2}[cmd.lower()]
            while i<len(toks) and not toks[i].isalpha():
                vals=[]
                for _ in range(n):
                    if i>=len(toks) or toks[i].isalpha(): break
                    vv,i=num(i);vals.append(vv)
                if len(vals)<n: break
                end=np.array(vals[-2:]);cur=(cur+end) if cmd.islower() else end;pts.append(cur.copy())
        elif cmd in('z','Z'): i+=1
        else: i+=1
    pts=np.array(pts); rect=cv2.minAreaRect(pts.astype(np.float32)); (cx,cy),(rw,rh),ang=rect
    th=np.deg2rad(ang); R=np.array([[np.cos(-th),-np.sin(-th)],[np.sin(-th),np.cos(-th)]]); q=(pts-[cx,cy])@R.T
    if np.ptp(q[:,0])>np.ptp(q[:,1]): q=q[:,::-1]
    q-=q.min(0); q[:,0]/=np.ptp(q[:,0]); q[:,1]/=np.ptp(q[:,1])
    if np.ptp(q[q[:,1]<0.15][:,0])<np.ptp(q[q[:,1]>0.85][:,0]): q[:,1]=1-q[:,1]
    return q
def fine(yf): return float(np.interp(yf,[0,.35,.60,.78,1],[18,15,12.5,10.5,9]))
def build(mean12,var12,eg,outline,W,L,BASE_MM,TEX_MM,res=1.0):
    import cv2
    nx=int(W/res); ny=int(L/res); gx,gy=np.meshgrid(np.linspace(0,W,nx),np.linspace(0,L,ny))
    px=TEMPLATE[:,0]*W; py=TEMPLATE[:,1]*L
    base=np.clip(Rbf(px,py,mean12,function='multiquadric',smooth=3)(gx,gy),0,None); base/=(base.max() or 1)
    vf=np.clip(Rbf(px,py,var12,function='multiquadric',smooth=3)(gx,gy),0,None); vf/=(vf.max() or 1)
    tex=np.zeros_like(base)
    for j in range(ny): k=2*np.pi/fine(gy[j,0]/L); tex[j,:]=np.sin(gx[j,:]*k)*np.sin(gy[j,:]*k)
    H=base*BASE_MM+tex*(vf*eg)*TEX_MM
    poly=(outline*[W,L]).reshape(-1,1,2).astype(np.float32); mask=np.zeros((ny,nx),bool)
    for j in range(ny):
        for ii in range(nx):
            if cv2.pointPolygonTest(poly,(float(gx[j,ii]),float(gy[j,ii])),False)>=0: mask[j,ii]=True
    dist=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3); H*=np.clip(dist/6,0,1); H[~mask]=0
    return gx,gy,H
def to_stl(gx,gy,H,path,floor=2.0):
    ny,nx=H.shape; Z=H+floor
    def v(i,j,t): return f"{gx[i,j]:.3f} {gy[i,j]:.3f} {(Z[i,j] if t else 0.0):.3f}"
    with open(path,'w') as f:
        f.write("solid foot\n")
        def tri(a,b,c): f.write(f"facet normal 0 0 0\n outer loop\n  vertex {a}\n  vertex {b}\n  vertex {c}\n endloop\nendfacet\n")
        for i in range(ny-1):
            for j in range(nx-1):
                tri(v(i,j,1),v(i,j+1,1),v(i+1,j+1,1)); tri(v(i,j,1),v(i+1,j+1,1),v(i+1,j,1))
                tri(v(i,j,0),v(i+1,j+1,0),v(i,j+1,0)); tri(v(i,j,0),v(i+1,j,0),v(i+1,j+1,0))
        f.write("endsolid foot\n")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--even',required=True); ap.add_argument('--uneven',required=True)
    ap.add_argument('--even-size',type=int,required=True); ap.add_argument('--uneven-size',type=int,required=True)
    ap.add_argument('--floor',type=float,default=0.15); ap.add_argument('--base-mm',type=float,default=12.0)
    ap.add_argument('--tex-mm',type=float,default=4.0); ap.add_argument('--outline',default=None)
    ap.add_argument('--outdir',default='.')
    a=ap.parse_args()
    os.makedirs(a.outdir,exist_ok=True); outline=load_outline(a.outline)
    walks=[('even',a.even,a.even_size),('uneven',a.uneven,a.uneven_size)]
    data=[]
    for label,path,size in walks:
        if size not in SOLE: sys.exit(f"unknown size {size}; known {sorted(SOLE)}")
        if not os.path.exists(path): sys.exit(f"file not found: {path}")
        pool,irr,nL,nR=pooled(path); n=len(pool)
        if n==0: sys.exit(f"{label}: no clean steps after gating -- cannot build.")
        print(f"{label}: {n} clean pooled steps (L={nL},R={nR}), irregularity {irr:.0f}"
              +("  [WARN <30 steps: noisy]" if n<30 else ""))
        data.append(dict(label=label,path=path,size=size,pool=pool,irr=irr,n=n))
    # RELATIVE scale within the pair: more-even -> floor, less-even -> 1.0
    irrs=[d['irr'] if d['irr']==d['irr'] else 0.0 for d in data]
    lo,hi=min(irrs),max(irrs); span=hi-lo if hi>lo else 1.0
    for d in data:
        d['eg']=a.floor + (1.0-a.floor)*((d['irr']-lo)/span)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for d in data:
        W,L=SOLE[d['size']][1],SOLE[d['size']][0]
        gx,gy,H=build(d['pool'].mean(0),d['pool'].std(0),d['eg'],outline,W,L,a.base_mm,a.tex_mm)
        stem=os.path.join(a.outdir,os.path.splitext(os.path.basename(d['path']))[0])
        np.save(stem+'_heightmap.npy',H); to_stl(gx,gy,H,stem+'.stl')
        plt.figure(figsize=(3,7)); plt.imshow(H,origin='lower',cmap='viridis',aspect='equal'); plt.axis('off')
        plt.title(f"{d['label']}  ({os.path.basename(d['path'])})\n{W:.0f}x{L:.0f}mm · texture {d['eg']:.2f} · n={d['n']}",fontsize=8)
        plt.colorbar(label='height (mm)',shrink=0.6); plt.savefig(stem+'.png',dpi=140,bbox_inches='tight'); plt.close()
        print(f"  {d['label']}: texture {d['eg']:.2f}, peak {H.max():.1f}mm -> {stem}.stl/.png/_heightmap.npy")
    print("NOTE: verify STLs are watertight/manifold in your slicer before printing.")
if __name__=='__main__': main()
