#!/usr/bin/env python3
"""
generate_physicalization.py
Generate 3D-printable foot heightmaps (STL) + previews (PNG) for a PAIR of insole
recordings — one even, one uneven — for the GroundWalks tactile physicalization.

Two walks are required so the unevenness scale is set RELATIVE to the pair: the more-even
walk gets gentle terrain, the less-even walk gets rugged terrain, on a shared scale. This
self-calibrates the contrast (no fixed evenness bounds needed).

ENCODING per walk
  * BASE height = mean pressure footprint at each clean step's FLAT-FOOT instant (peak total
                  pressure), pooled across BOTH feet (right mirrored to the left frame via the
                  real sensor coords) -> the print reads as "a step".
  * TERRAIN     = a few sparse, meandering, ROUNDED RIDGES ("mountain ranges") laid over the
                  footprint, evoking stepping on a root / a bump in grass — NOT a uniform field
                  of pebbles or a bed of nails. Unevenness is encoded two ways, BOTH scaled by
                  the walk's evenness irregularity (relative to the pair):
                    (1) OVERALL RELIEF  — even walk -> low gentle ridges (~even_ridge_mm);
                        uneven walk -> tall ridges (~max_ridge_mm).
                    (2) ALONG-CREST VARIATION — even walk -> near-constant crest height
                        (a smooth speed-bump); uneven walk -> crests that rise and fall
                        dramatically along their length (peaks & saddles, a jagged range).
                  Ridges are kept SPARSE and are spaced apart (min_ridge_spacing_mm), so the
                  surface can never fill into a uniform washboard (which would read as EVEN).
                  Ridge start points are data-biased toward high-load / high-variability zones,
                  so placement still reflects the specific walk.
  DESIGN RATIONALE: regularity reads to the foot as EVEN, so unevenness must be IRREGULARITY —
                  differing heights between adjacent high points, with controlled randomness —
                  not merely more or taller features. Encoding via relief + along-crest variance
                  (not feature COUNT) also avoids the "too many peaks -> flat again" failure.

STEP GATE (matches classification/evenness pipelines): transition/wait, surface-straddle,
  and <80%-labelled steps dropped; unlabelled recordings kept whole.

USAGE (paths are explicit; run from anywhere)
  python generate_physicalization.py \
      --even  "../Individual Users/sam/REC_even.csv"  --even-size 43 \
      --uneven "../Individual Users/sam/REC_uneven.csv" --uneven-size 44
Options:
  --floor FLOAT         unevenness scale for the SMOOTHER walk (default 0.15; keeps it non-flat)
  --base-mm FLOAT       footprint base relief mm (default 12)
  --max-ridge-mm FLOAT  tallest crest height mm for the uneven walk (override terrain default)
  --even-ridge-mm FLOAT crest height mm for the even walk (override; smaller = flatter even print)
  --n-ridges INT        number of ridges (override; 2-4 typical)
  --outline PATH        SVG outline override (else embedded traced outline)
  --outdir DIR          output directory (default: current dir)

GEOMETRY (--geometry)
  hollow  (default) top shell follows the terrain; the space beneath it is closed off by a
          FLAT bottom plate, trimmed to the foot outline. Material = shell_mm top skin +
          floor_mm base plate + wall_mm sides; the interior is an enclosed void, so it
          prints with zero infill. Where the terrain is too low for a void to fit, that
          part is simply solid.
  solid   top terrain + flat bottom with no void: one closed volume, let the slicer choose
          the infill. Simplest and most robust, but the slicer decides the internals.
  shell   the original constant-thickness offset 'shrinkwrap': the underside FOLLOWS the
          terrain. Thin and flexible - a 2mm skin deflects under finger pressure, which
          makes crest heights feel softer than they are.
  (finer terrain knobs — spacing, widths, meander, crest control points — live in TERRAIN_DEFAULT.)

TUNING: the terrain numbers are STARTING POINTS. This is tactile and only a test print can
  settle the feel; every terrain parameter is an exposed knob. First dials to try: even_ridge_mm
  (how gentle the even print is), max_ridge_mm (how dramatic the uneven print is), crest_var_max
  (how jagged the uneven crests are), and min_ridge_spacing_mm (how spread out the ridges are).

HONEST LIMITS: only 12 pressure pads -> spatial detail below the sensor spacing is interpolated
  (the base footprint's coarse loading pattern is real data; the fine terrain within it is an
  interpretive rendering, not measured micro-topography). Sensor coords/outline are proportional,
  not surveyed. Every writer now runs an edge-manifold + signed-volume self-check and prints
  the result, but still check in your slicer before printing. Thin-data walks
  (<30 clean steps) print a warning. If the min-spacing can't fit n_ridges on a small sole it is
  auto-relaxed and a [note] is printed.
"""
import argparse, csv, os, sys, numpy as np
from scipy.interpolate import Rbf
from scipy.signal import find_peaks

# real sensor coords from cop_coords (index k = pressure_(k+1)); left frame is the template
COORDS_L=np.array([[0.7438,0.8713],[0.7979,0.7165],[0.6264,0.5308],[0.4839,0.7074],[0.6441,0.7103],
 [0.3293,0.6891],[0.2920,0.4243],[0.3119,0.1321],[0.1844,0.6659],[0.5604,0.2574],[0.5396,0.1170],[0.4226,0.8585]])
TEMPLATE=COORDS_L
# ---- terrain (mountain-range ridge) knobs -- tune against test prints; only your feet judge the feel ----
TERRAIN_DEFAULT=dict(
    n_ridges=4,           # sparse meandering ridges ('mountain ranges'). ~2-4; more -> washboard.
    min_ridge_spacing_mm=43.0, # min distance between ridge SEED points (prevents bunching)
    max_ridge_mm=14.0,    # tallest crest height (mm) for the MOST-UNEVEN walk
    even_ridge_mm=3.0,    # crest height (mm) for the MOST-EVEN walk -> overall relief scales with evenness
    crest_mean_frac=0.55, # a 'typical' crest sits at this fraction of the (evenness-scaled) max
    crest_var_min=0.02,   # ALONG-CREST height variation when EVEN (near-constant, smooth speed-bump)
    crest_var_max=0.55,   # ALONG-CREST height variation when UNEVEN (jagged range) -> unevenness dial
    ridge_w_min=7.0,      # ridge half-width sigma (mm): narrow=root ...
    ridge_w_max=15.0,     # ... wide=broad ridge
    ridge_wiggle=0.18,    # how much each ridge path meanders (fraction of sole length)
    ridge_len_min=0.55,   # ridge length as a fraction of sole length: min ...
    ridge_len_max=0.90,   # ... max. Small values keep a ridge inside one foot region.
    region_profile=None,  # None = uniform. Else [(f,gain),...] with f=0 heel .. 1 toe:
                          # a multiplier on BOTH ridge seeding and ridge relief along the
                          # foot. Lets a prototype concentrate texture in one region while
                          # leaving another as bare pressure contour. NOT data-driven --
                          # anything other than None makes the print a design mock-up.
    crest_bumps=3,        # along-crest height control points (fewer -> fewer inflections on even ridges)
    ground_octaves=3,     # smooth undulation layers
    ground_mm=1.5,        # smooth-ground relief
)
# Prototype presets. These OVERRIDE the data-driven look on purpose -- use them to test
# how a texture feels, never to represent a walk.
PRESETS={
    'arch-peaks': dict(          # dense irregular peaks in the arch/midfoot, calm heel
        region_profile=[(0.00,0.10),(0.26,0.12),(0.38,0.85),(0.50,1.00),
                        (0.62,0.85),(0.76,0.25),(1.00,0.20)],
        n_ridges=9, min_ridge_spacing_mm=18.0,
        ridge_len_min=0.14, ridge_len_max=0.32,     # short, so they stay in the arch
        ridge_w_min=5.0, ridge_w_max=10.0,          # narrow -> peaks rather than swells
        crest_bumps=9, crest_var_max=0.70,          # many along-crest peaks and saddles
        ridge_wiggle=0.06, ground_mm=1.0,
    ),
}
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
def build(mean12, var12, eg, outline, W, L, BASE_MM, TERRAIN, seed=0, res=1.0):
    """Footprint base (from pressure) + sparse meandering RIDGES ('mountain ranges').
    Unevenness (eg in [0,1]) is encoded as ALONG-CREST height variation: an even walk gives
    smooth ridges of near-constant crest height; an uneven walk gives few ridges whose crests
    rise and fall dramatically along their length. Ridges are sparse (never a washboard) and
    have a rounded Gaussian cross-section (not sharp). TERRAIN holds the tunable knobs."""
    import cv2
    nx=int(W/res); ny=int(L/res); gx,gy=np.meshgrid(np.linspace(0,W,nx),np.linspace(0,L,ny))
    px=TEMPLATE[:,0]*W; py=TEMPLATE[:,1]*L
    base=np.clip(Rbf(px,py,mean12,function='multiquadric',smooth=3)(gx,gy),0,None); base/=(base.max() or 1)
    vf =np.clip(Rbf(px,py,var12,function='multiquadric',smooth=3)(gx,gy),0,None); vf/=(vf.max() or 1)

    poly=(outline*[W,L]).reshape(-1,1,2).astype(np.float32); mask=np.zeros((ny,nx),bool)
    for j in range(ny):
        for ii in range(nx):
            if cv2.pointPolygonTest(poly,(float(gx[j,ii]),float(gy[j,ii])),False)>=0: mask[j,ii]=True
    dist=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3); feather=np.clip(dist/6,0,1)

    rng=np.random.default_rng(seed)
    # gentle smooth ground
    ug=np.zeros((ny,nx))
    for _ in range(TERRAIN['ground_octaves']):
        cxf=rng.uniform(0.2,0.8)*W; cyf=rng.uniform(0.2,0.8)*L
        kx=2*np.pi/rng.uniform(0.5*L,1.0*L); ky=2*np.pi/rng.uniform(0.5*L,1.0*L)
        ug+=np.sin((gx-cxf)*kx)*np.sin((gy-cyf)*ky)
    ug=(ug-ug.min())/(np.ptp(ug) or 1)

    # along-foot region gain (1.0 everywhere unless region_profile is set)
    prof=TERRAIN.get('region_profile')
    if prof:
        fs=[p[0] for p in prof]; gs=[p[1] for p in prof]
        rgain=np.interp(gy/L, fs, gs)
    else:
        rgain=np.ones_like(gy)

    # data-biased ridge SEED points (start each ridge near high load/variability),
    # scaled by the region gain so seeds land where the prototype wants texture
    weight=(base*0.6+vf*0.4)*rgain*mask; flat=np.clip(weight.flatten(),0,None)
    if flat.sum()<=0: flat=mask.flatten().astype(float)
    prob=flat/flat.sum()

    def ridge_path(seed_idx):
        """A meandering diagonal polyline across the sole starting near seed_idx."""
        sy,sx = np.unravel_index(seed_idx,(ny,nx))
        # random overall direction (favor diagonal/cross-foot), length ~ sole size
        ang=rng.uniform(0,np.pi)                    # 0..pi -> any orientation, diagonal likely
        length=rng.uniform(TERRAIN['ridge_len_min'],TERRAIN['ridge_len_max'])*L
        npts=8
        t=np.linspace(0,1,npts)
        cx=gx[0,sx]+np.cos(ang)*length*(t-0.5)
        cy=gy[sy,0]+np.sin(ang)*length*(t-0.5)
        # meander: add smooth perpendicular wiggle
        perp=np.array([-np.sin(ang),np.cos(ang)])
        wig=np.cumsum(rng.normal(0,1,npts)); wig=(wig-wig.mean())
        amp=TERRAIN['ridge_wiggle']*L
        cx+=perp[0]*amp*wig/ (np.abs(wig).max() or 1)
        cy+=perp[1]*amp*wig/ (np.abs(wig).max() or 1)
        return np.column_stack([cx,cy])

    # along-crest height SPREAD scales with unevenness (even = near-constant crest)
    crest_var = TERRAIN['crest_var_min'] + eg*(TERRAIN['crest_var_max']-TERRAIN['crest_var_min'])
    # OVERALL relief ALSO scales with unevenness: even walk -> low gentle ridges, uneven -> tall
    ridge_mm  = TERRAIN['even_ridge_mm'] + eg*(TERRAIN['max_ridge_mm']-TERRAIN['even_ridge_mm'])

    field=np.zeros((ny,nx))
    # --- data-biased seeds WITH a minimum-separation constraint (Option B) ---
    def place_seeds(n, spacing):
        placed=[]; coords=[]
        tries=0; maxtries=4000
        while len(placed)<n and tries<maxtries:
            tries+=1
            cand=rng.choice(len(prob), p=prob)
            cy,cx=np.unravel_index(cand,(ny,nx)); ptmm=(gx[0,cx],gy[cy,0])
            if all(np.hypot(ptmm[0]-q[0], ptmm[1]-q[1])>=spacing for q in coords):
                placed.append(cand); coords.append(ptmm)
        return placed
    sp=TERRAIN['min_ridge_spacing_mm']; seeds=place_seeds(TERRAIN['n_ridges'], sp)
    # relax spacing if the sole/data can't fit n ridges at full distance (rather than bunch/fail)
    while len(seeds)<TERRAIN['n_ridges'] and sp>10:
        sp*=0.8; seeds=place_seeds(TERRAIN['n_ridges'], sp)
    if sp<TERRAIN['min_ridge_spacing_mm']:
        print(f'    [note] ridge spacing relaxed to {sp:.0f}mm to fit {TERRAIN["n_ridges"]} ridges on this sole')
    for s_idx in seeds:
        path=ridge_path(s_idx)
        w=rng.uniform(TERRAIN['ridge_w_min'],TERRAIN['ridge_w_max'])
        # crest height along path: interpolate random control points (peaks & saddles)
        cp=np.clip(rng.normal(TERRAIN['crest_mean_frac'],crest_var,TERRAIN['crest_bumps']),0.1,1.0)
        cp_t=np.linspace(0,1,TERRAIN['crest_bumps'])
        # densify path and its crest height
        dt=np.linspace(0,1,60)
        pxs=np.interp(dt,np.linspace(0,1,len(path)),path[:,0])
        pys=np.interp(dt,np.linspace(0,1,len(path)),path[:,1])
        pch=np.interp(dt,cp_t,cp)
        # stamp a Gaussian ridge: for each grid point, distance to nearest path sample
        for k in range(len(dt)):
            g=np.exp(-(((gx-pxs[k])**2+(gy-pys[k])**2)/(2*w**2)))
            field=np.maximum(field, pch[k]*g)   # max-blend keeps a continuous crest, no plateau

    # region gain applies to the RIDGE term only: the pressure base (and its smooth
    # ground undulation) still reads as an anatomical contour wherever ridges are damped
    H = base*BASE_MM + ug*TERRAIN['ground_mm'] + field*ridge_mm*rgain
    H*=feather; H[~mask]=0
    return gx,gy,H,mask
def to_stl(gx,gy,H,path,shell_mm=2.0,mask=None,**_):
    """Constant-thickness OFFSET SHELL ('shrinkwrap'): the top terrain surface plus an inner
    surface offset DOWNWARD by shell_mm that FOLLOWS the terrain (not a flat floor), joined by a
    rim at the foot outline. Result is a watertight skin of ~shell_mm thickness, hollow inside."""
    ny,nx=H.shape
    Ztop=H.copy()                # TOP surface = terrain, UNCHANGED (matches original heights)
    Zin = H - shell_mm           # INNER surface offset straight DOWN by the shell thickness
    # lift the whole shell so its LOWEST inner point rests at BASE_LIFT (nothing below z=0)
    BASE_LIFT=2.0
    _lo=float(np.nanmin(np.where(mask if mask is not None else True, Zin, np.inf))) if mask is not None else float(Zin.min())
    _shift=BASE_LIFT-_lo
    Ztop=Ztop+_shift; Zin=Zin+_shift
    if mask is None: mask=np.ones((ny,nx),bool)
    def T(i,j): return (gx[i,j],gy[i,j],float(Ztop[i,j]))   # outer/top
    def I(i,j): return (gx[i,j],gy[i,j],float(Zin[i,j]))    # inner (offset down by shell_mm)
    top_tris=[]
    for i in range(ny-1):
        for j in range(nx-1):
            if mask[i,j] and mask[i,j+1] and mask[i+1,j] and mask[i+1,j+1]:
                top_tris.append(((i,j),(i,j+1),(i+1,j+1)))
                top_tris.append(((i,j),(i+1,j+1),(i+1,j)))
    # open boundary edges of the surface (guaranteed closure)
    eset=set()
    for (a,b,c) in top_tris:
        for e in [(a,b),(b,c),(c,a)]: eset.add(e)
    boundary=[e for e in eset if (e[1],e[0]) not in eset]
    tris=[]
    for (a,b,c) in top_tris:
        tris.append((T(*a),T(*b),T(*c)))       # outer skin (up)
        tris.append((I(*a),I(*c),I(*b)))       # inner skin (down, reversed) -> hollow underside follows terrain
    for (a,b) in boundary:                      # rim joining outer edge to inner edge
        at,bt=T(*a),T(*b); ai,bi=I(*a),I(*b)
        # winding must be the REVERSE of the inner skin's boundary edge (bi->ai), or every
        # rim edge is emitted twice in the same direction and the mesh is not watertight
        tris.append((at,ai,bi)); tris.append((at,bi,bt))
    def norm(t):
        (ax,ay,az),(bx,by,bz),(cx,cy,cz)=t
        ux,uy,uz=bx-ax,by-ay,bz-az; vx,vy,vz=cx-ax,cy-ay,cz-az
        nx_,ny_,nz_=uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx
        m=(nx_*nx_+ny_*ny_+nz_*nz_)**0.5 or 1.0
        return nx_/m,ny_/m,nz_/m
    with open(path,'w') as f:
        f.write("solid foot\n")
        for t in tris:
            n=norm(t); f.write(f"facet normal {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n outer loop\n")
            for vx in t: f.write(f"  vertex {vx[0]:.3f} {vx[1]:.3f} {vx[2]:.3f}\n")
            f.write(" endloop\nendfacet\n")
        f.write("endsolid foot\n")


def _closed_prism(Zup, z_lo, mask, tag):
    """Faces of the closed volume between height field Zup and the plane z_lo, over
    mask. Returns keyed faces (tag, kind, i, j) with outward normals."""
    ny, nx = Zup.shape
    quads = [(i, j) for i in range(ny - 1) for j in range(nx - 1)
             if mask[i, j] and mask[i, j + 1] and mask[i + 1, j] and mask[i + 1, j + 1]]
    if not quads:
        return [], []
    top = []
    for (i, j) in quads:
        top.append(((i, j), (i, j + 1), (i + 1, j + 1)))
        top.append(((i, j), (i + 1, j + 1), (i + 1, j)))
    eset = set()
    for (a, b, c) in top:
        eset.update([(a, b), (b, c), (c, a)])
    boundary = [e for e in eset if (e[1], e[0]) not in eset]
    faces = []
    for (a, b, c) in top:
        faces.append(((tag, 'U', a), (tag, 'U', b), (tag, 'U', c)))     # upper, +z
        faces.append(((tag, 'L', a), (tag, 'L', c), (tag, 'L', b)))     # lower, -z
    for (a, b) in boundary:
        faces.append(((tag, 'U', a), (tag, 'L', a), (tag, 'L', b)))     # side, outward
        faces.append(((tag, 'U', a), (tag, 'L', b), (tag, 'U', b)))
    return faces, quads


def _write_stl(path, faces, xyz):
    def norm(tri):
        p, q, r = (np.array(x) for x in tri)
        n = np.cross(q - p, r - p)
        return n / (float(np.linalg.norm(n)) or 1.0)
    with open(path, 'w') as f:
        f.write('solid foot\n')
        for face in faces:
            tri = [xyz(v) for v in face]
            n = norm(tri)
            f.write(f'facet normal {n[0]:.4f} {n[1]:.4f} {n[2]:.4f}\n outer loop\n')
            for v in tri:
                f.write(f'  vertex {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n')
            f.write(' endloop\nendfacet\n')
        f.write('endsolid foot\n')


def _manifold_check(faces, xyz):
    """Directed-edge pairing + signed volume. Every edge must appear exactly once in
    each direction; positive volume means the normals face outward."""
    ec = {}
    for f in faces:
        for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            ec[e] = ec.get(e, 0) + 1
    bad = [e for e, n in ec.items() if n != 1 or ec.get((e[1], e[0]), 0) != 1]
    vol = 0.0
    for f in faces:
        p, q, r = (np.array(xyz(v)) for v in f)
        vol += float(np.dot(p, np.cross(q, r))) / 6.0
    return len(bad), vol


def to_stl_solid(gx, gy, H, path, floor_mm=2.0, mask=None, verify=True, **_):
    """CLOSED SOLID: the terrain top surface, a FLAT bottom at z=0, and vertical side
    walls joining the two at the foot outline. Watertight and filled (no void), so the
    slicer decides the internal infill. The terrain is shifted so its LOWEST point
    inside the outline sits floor_mm above the bottom."""
    ny, nx = H.shape
    if mask is None:
        mask = np.ones((ny, nx), bool)
    Ztop = H.astype(float).copy()
    lo = float(np.nanmin(np.where(mask, Ztop, np.inf)))
    Ztop = Ztop - lo + float(floor_mm)

    faces, quads = _closed_prism(Ztop, 0.0, mask, 'o')
    if not faces:
        raise ValueError('to_stl_solid: outline mask contains no complete grid cell')

    def xyz(v):
        tag, kind, (i, j) = v
        return (float(gx[i, j]), float(gy[i, j]),
                float(Ztop[i, j]) if kind == 'U' else 0.0)

    if verify:
        bad, vol = _manifold_check(faces, xyz)
        print(f'    [solid] {len(faces)} facets | edge-manifold: '
              f'{"OK" if not bad else f"{bad} bad edges"} | volume {vol/1000:.1f} cm^3')
        if bad or vol <= 0:
            print('    [solid] WARNING not a clean closed solid -- check in the slicer')
    _write_stl(path, faces, xyz)


def to_stl_hollow(gx, gy, H, path, shell_mm=2.0, floor_mm=2.0, wall_mm=2.5,
                  mask=None, verify=True, **_):
    """HOLLOW SOLID: the terrain-following top shell (thickness shell_mm), closed
    underneath by a FLAT bottom instead of a terrain-following underside, and trimmed
    to the foot outline. Material is the shell, a floor_mm base plate and wall_mm side
    walls; the space between the shell's underside and the base plate is an enclosed
    void, so it prints with zero infill. Where the terrain is too low for a void to
    fit, the object is simply solid there."""
    import cv2
    ny, nx = H.shape
    if mask is None:
        mask = np.ones((ny, nx), bool)
    dx = float(abs(gx[0, 1] - gx[0, 0])) if nx > 1 else 1.0
    dy = float(abs(gy[1, 0] - gy[0, 0])) if ny > 1 else 1.0
    res = 0.5 * (dx + dy)

    # sit the lowest terrain point exactly floor_mm + shell_mm above a flat z=0 bottom,
    # so the void closes itself off where the terrain is lowest
    Ztop = H.astype(float).copy()
    lo = float(np.nanmin(np.where(mask, Ztop, np.inf)))
    Ztop = Ztop - lo + float(floor_mm) + float(shell_mm)
    Zin = Ztop - float(shell_mm)                      # underside of the top shell

    # OUTER boundary: terrain top, vertical rim at the outline, flat bottom at z=0
    outer, oquads = _closed_prism(Ztop, 0.0, mask, 'o')
    if not outer:
        raise ValueError('to_stl_hollow: outline mask contains no complete grid cell')

    # VOID: inset from the outline by wall_mm, and only where the shell underside
    # clears the base plate
    k = max(1, int(round(2 * float(wall_mm) / res)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inset = cv2.erode(mask.astype(np.uint8), ker).astype(bool)
    void_mask = inset & (Zin > float(floor_mm) + 0.05)
    void, vquads = _closed_prism(Zin, float(floor_mm), void_mask, 'v')
    faces = list(outer) + [(f[0], f[2], f[1]) for f in void]   # void normals reversed

    def xyz(v):
        tag, kind, (i, j) = v
        if tag == 'o':
            z = Ztop[i, j] if kind == 'U' else 0.0
        else:
            z = Zin[i, j] if kind == 'U' else float(floor_mm)
        return (float(gx[i, j]), float(gy[i, j]), float(z))

    if verify:
        bad, vol = _manifold_check(faces, xyz)
        _, v_out = _manifold_check(outer, xyz)
        v_void = v_out - vol
        pct = 100 * v_void / v_out if v_out else 0.0
        print(f'    [hollow] {len(faces)} facets | edge-manifold: '
              f'{"OK" if not bad else f"{bad} bad edges"} | '
              f'material {vol/1000:.1f} cm^3 '
              f'(void {v_void/1000:.1f} cm^3 = {pct:.0f}% of the envelope)')
        if not vquads:
            print('    [hollow] note: no void fits at these thicknesses -- solid throughout')
        if bad or vol <= 0:
            print('    [hollow] WARNING not a clean closed solid -- check in the slicer')
    _write_stl(path, faces, xyz)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--even',required=True); ap.add_argument('--uneven',required=True)
    ap.add_argument('--even-size',type=int,required=True); ap.add_argument('--uneven-size',type=int,required=True)
    ap.add_argument('--floor',type=float,default=0.15); ap.add_argument('--base-mm',type=float,default=12.0)
    ap.add_argument('--shell-mm',type=float,default=2.0,help='top shell thickness mm')
    ap.add_argument('--geometry',choices=['hollow','solid','shell'],default='hollow',
                    help="hollow = terrain shell closed by a flat bottom (default); "
                         "solid = terrain + flat bottom, slicer-filled; "
                         "shell = original terrain-following offset shrinkwrap")
    ap.add_argument('--floor-mm',type=float,default=2.0,help='base-plate thickness mm (hollow/solid)')
    ap.add_argument('--wall-mm',type=float,default=2.5,help='side-wall thickness mm (hollow)')
    ap.add_argument('--max-ridge-mm',type=float,default=None,help='tallest crest height mm (override)')
    ap.add_argument('--n-ridges',type=int,default=None,help='number of ridges (override)')
    ap.add_argument('--preset',choices=sorted(PRESETS),default=None,
                    help='prototype terrain preset (overrides the data-driven look)')
    ap.add_argument('--even-ridge-mm',type=float,default=None,help='crest height mm for the even walk (override)'); ap.add_argument('--outline',default=None)
    ap.add_argument('--outdir',default='.')
    a=ap.parse_args()
    if a.preset:
        TERRAIN_DEFAULT.update(PRESETS[a.preset])
        print(f"PRESET '{a.preset}' applied -- terrain is rigged for prototyping, "
              "not a faithful rendering of these walks.")
    if a.max_ridge_mm is not None: TERRAIN_DEFAULT['max_ridge_mm']=a.max_ridge_mm
    if a.n_ridges is not None: TERRAIN_DEFAULT['n_ridges']=a.n_ridges
    if a.even_ridge_mm is not None: TERRAIN_DEFAULT['even_ridge_mm']=a.even_ridge_mm
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
        gx,gy,H,fmask=build(d['pool'].mean(0),d['pool'].std(0),d['eg'],outline,W,L,a.base_mm,TERRAIN_DEFAULT,seed=abs(hash(d['label']))%(2**31))
        stem=os.path.join(a.outdir,os.path.splitext(os.path.basename(d['path']))[0])
        np.save(stem+'_heightmap.npy',H)
        if a.geometry=='hollow':
            to_stl_hollow(gx,gy,H,stem+'.stl',shell_mm=a.shell_mm,floor_mm=a.floor_mm,
                          wall_mm=a.wall_mm,mask=fmask)
        elif a.geometry=='solid':
            to_stl_solid(gx,gy,H,stem+'.stl',floor_mm=a.floor_mm,mask=fmask)
        else:
            to_stl(gx,gy,H,stem+'.stl',shell_mm=a.shell_mm,mask=fmask)
        plt.figure(figsize=(3,7)); plt.imshow(H,origin='lower',cmap='viridis',aspect='equal'); plt.axis('off')
        plt.title(f"{d['label']}  ({os.path.basename(d['path'])})\n{W:.0f}x{L:.0f}mm · texture {d['eg']:.2f} · n={d['n']}",fontsize=8)
        plt.colorbar(label='height (mm)',shrink=0.6); plt.savefig(stem+'.png',dpi=140,bbox_inches='tight'); plt.close()
        print(f"  {d['label']}: texture {d['eg']:.2f}, peak {H.max():.1f}mm -> {stem}.stl/.png/_heightmap.npy")
    print(f"NOTE: geometry={a.geometry}. Self-checks above report edge-manifoldness and "
          "volume; still confirm in your slicer before printing.")
if __name__=='__main__': main()
