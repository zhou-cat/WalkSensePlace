"""Model / pipeline overview figure for the GroundWalks paper.

A two-stage block diagram: intra-modality per-step segmentation and feature extraction
on the left, cross-modality modelling and scores on the right. Vector output, sized for
acmsmall's text width so \\includegraphics[width=\\linewidth] needs no scaling — scaling
is what makes figure text disagree with body text.

    python figure_pipeline.py                 # -> figures/fig_pipeline.{pdf,svg,png}
    python figure_pipeline.py --width 7.0     # wider, e.g. for a slide
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── palette: the project's surface colours, plus a hue per sensing modality ──
PRESS = '#2A6DB0'      # plantar pressure stream
IMU = '#C0552B'        # inertial stream
INK = '#1B2A30'
GREY = '#6B7772'
OUT_FC = '#D8DEDC'     # output boxes
PANEL = '#F3F6F5'
GRASS, LOOSE, PAVED = '#4CAF50', '#8D6E63', '#607D8B'
BYPASS = '#C9D2CF'

FS = dict(tiny=4.6, small=5.2, body=5.8, head=6.4, cap=6.0)


def cmap_of(hexcol, lo=0.28, hi=1.0):
    """White-to-colour ramp for the gradient feature-extraction blocks."""
    c = matplotlib.colors.to_rgb(hexcol)
    a = tuple(1 - lo * (1 - x) for x in c)
    b = tuple(1 - hi * (1 - x) for x in c)
    return LinearSegmentedColormap.from_list('g', [a, b])


def box(ax, x, y, w, h, label='', fc='white', ec=INK, lw=0.6, fs=FS['body'],
        tc=INK, r=1.2, dashed=False, bold=False, zorder=3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0,rounding_size=%g' % r,
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
                       linestyle=(0, (2.4, 1.6)) if dashed else '-')
    ax.add_patch(p)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fs,
                color=tc, zorder=zorder + 1, linespacing=1.5,
                fontweight='bold' if bold else 'normal')
    return p


def grad_box(ax, x, y, w, h, hexcol, label, fs=FS['small']):
    """Vertical gradient block, clipped to a rounded rectangle."""
    clip = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                          linewidth=0.6, edgecolor=INK, facecolor='none', zorder=5)
    ax.add_patch(clip)
    im = ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), cmap=cmap_of(hexcol),
                   extent=[x, x + w, y, y + h], aspect='auto', zorder=4,
                   interpolation='bilinear', origin='lower')
    im.set_clip_path(clip)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', rotation=90,
            fontsize=fs, color='white', fontweight='bold', zorder=6)


def arrow(ax, p0, p1, color=INK, lw=0.7, rad=0.0, ms=5.0, zorder=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms,
                                 linewidth=lw, color=color, zorder=zorder,
                                 shrinkA=0.5, shrinkB=0.5,
                                 connectionstyle='arc3,rad=%g' % rad))


def elbow(ax, p0, p1, color=INK, lw=0.7, ms=5.0, first='h', zorder=6):
    """Right-angled connector: horizontal then vertical, or the reverse."""
    mid = (p1[0], p0[1]) if first == 'h' else (p0[0], p1[1])
    ax.plot([p0[0], mid[0]], [p0[1], mid[1]], color=color, lw=lw, zorder=zorder,
            solid_capstyle='round')
    arrow(ax, mid, p1, color=color, lw=lw, ms=ms, zorder=zorder)


def sparkline(ax, x, y, w, h, color, seed=7):
    """A stance-like trace with heel-strike peaks, for the signal icon."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 260)
    s = np.zeros_like(t)
    for c in (0.14, 0.38, 0.62, 0.86):
        s += np.exp(-((t - c) / 0.055) ** 2) * rng.uniform(0.8, 1.0)
    s += 0.05 * rng.standard_normal(t.size)
    s = (s - s.min()) / (np.ptp(s) or 1)
    ax.plot(x + 0.06 * w + t * 0.88 * w, y + 0.14 * h + s * 0.72 * h,
            color=color, lw=0.55, zorder=6)


def insole_icon(ax, x, y, w, h, png):
    """The project's own insole artwork if present, else a plain sole outline."""
    if png and Path(png).exists():
        a = mpimg.imread(png)
        al = a[..., 3] if a.shape[-1] == 4 else np.ones(a.shape[:2])
        rgba = np.zeros(a.shape[:2] + (4,))
        rgba[..., :3] = matplotlib.colors.to_rgb(INK)
        rgba[..., 3] = al
        ax.imshow(rgba, extent=[x, x + w, y, y + h], aspect='auto', zorder=6,
                  interpolation='bilinear')
        return
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(x + w / 2 + 0.42 * w * np.cos(th),
            y + h / 2 + 0.47 * h * np.sin(th) * (1 + 0.1 * np.cos(th)),
            color=INK, lw=0.6, zorder=6)


def build(width_in=5.478, png='insole_42_complete_380p.png'):
    fig = plt.figure(figsize=(width_in, width_in * 0.50))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis('off')

    # ── dashed stage containers ─────────────────────────────────────────────
    box(ax, 0.6, 1.4, 57.4, 43.6, fc='none', ec=GREY, lw=0.65, dashed=True,
        r=1.6, zorder=1)
    box(ax, 58.6, 1.4, 40.8, 43.6, fc='none', ec=GREY, lw=0.65, dashed=True,
        r=1.6, zorder=1)
    ax.text(2.4, 2.6, 'Intra-modality\nPer-step segmentation & feature extraction',
            fontsize=FS['cap'], color=INK, style='italic', va='bottom',
            linespacing=1.5, zorder=2)
    ax.text(60.4, 2.6, 'Cross-modality\nSurface & compliance models',
            fontsize=FS['cap'], color=INK, style='italic', va='bottom',
            linespacing=1.5, zorder=2)

    # ── inputs ──────────────────────────────────────────────────────────────
    insole_icon(ax, 3.4, 30.0, 5.6, 13.2, png)
    ax.text(6.4, 28.5, 'pressure $\\times$ 12', ha='center', fontsize=FS['tiny'],
            color=PRESS, fontweight='bold')
    box(ax, 3.0, 20.8, 6.8, 5.4, fc='none', ec=IMU, lw=0.5, r=0.8)
    sparkline(ax, 3.0, 20.8, 6.8, 5.4, IMU)
    ax.text(6.4, 19.3, '6-axis IMU', ha='center', fontsize=FS['tiny'],
            color=IMU, fontweight='bold')
    ax.text(2.4, 13.4, 'Insoles\n62.5 Hz · both feet\n'
                       'contact events · tags · GPX\n4 walkers · 78 recordings',
            ha='left', va='center', fontsize=FS['tiny'], color=GREY,
            linespacing=1.7)

    # ── segmentation + gate ─────────────────────────────────────────────────
    box(ax, 11.6, 18.6, 18.8, 24.6, fc=PANEL, ec=INK, lw=0.6)
    ax.text(21.0, 41.0, 'Step segmentation', ha='center', fontsize=FS['body'],
            fontweight='bold', color=INK, zorder=4)
    ax.text(21.0, 33.8,
            'vendor contact events\nheel strike $\\rightarrow$ heel strike\n'
            '(one gait cycle, same foot)\ntoe-off $\\rightarrow$ measured stance\n'
            '$\\geq$ 5 samples',
            ha='center', va='center', fontsize=FS['small'], color=INK,
            linespacing=1.8, zorder=4)
    box(ax, 12.8, 19.6, 16.4, 7.4, fc='white', ec=GREY, lw=0.5, r=0.9)
    ax.text(21.0, 25.5, 'Step gate', ha='center', fontsize=FS['small'],
            fontweight='bold', color=INK, zorder=4)
    ax.text(21.0, 22.4, 'drop transition / wait\ndrop multi-surface\n'
                        '$\\geq$ 80% one surface',
            ha='center', va='center', fontsize=FS['tiny'], color=GREY,
            linespacing=1.7, zorder=4)
    arrow(ax, (9.4, 36.6), (11.6, 36.6), color=PRESS, lw=0.9, ms=6)
    arrow(ax, (9.9, 23.5), (11.6, 23.5), color=IMU, lw=0.9, ms=6)

    # ── per-modality feature extraction ─────────────────────────────────────
    grad_box(ax, 31.6, 31.4, 4.2, 11.8, PRESS, 'Feature\nExtraction')
    grad_box(ax, 31.6, 17.6, 4.2, 11.8, IMU, 'Feature\nExtraction')
    arrow(ax, (30.4, 37.3), (31.6, 37.3), color=PRESS, lw=0.9, ms=6)
    arrow(ax, (30.4, 23.5), (31.6, 23.5), color=IMU, lw=0.9, ms=6)

    press_feats = ['zone means at flat foot', 'heel-curve shape & impulse',
                   'loading rate, midstance', 'CoP wander, spatial entropy']
    imu_feats = ['impact peak at heel strike', 'spectral centroid & entropy',
                 'ZUPT-gated foot tilt', 'step-to-step variability']
    for i, lab in enumerate(press_feats):
        yy = 41.6 - i * 3.0
        box(ax, 37.2, yy - 1.2, 19.4, 2.4, lab, fc='white', ec=PRESS, lw=0.5,
            fs=FS['small'], r=0.7)
        arrow(ax, (35.8, 37.3), (37.2, yy), color=PRESS, lw=0.5, ms=4)
    for i, lab in enumerate(imu_feats):
        yy = 27.8 - i * 3.0
        box(ax, 37.2, yy - 1.2, 19.4, 2.4, lab, fc='white', ec=IMU, lw=0.5,
            fs=FS['small'], r=0.7)
        arrow(ax, (35.8, 23.5), (37.2, yy), color=IMU, lw=0.5, ms=4)

    # ── feature vector ──────────────────────────────────────────────────────
    box(ax, 60.2, 33.4, 18.4, 9.8, fc=PANEL, ec=INK, lw=0.6)
    ax.text(69.4, 41.2, 'Feature vector', ha='center', fontsize=FS['body'],
            fontweight='bold', color=INK, zorder=4)
    ax.text(69.4, 36.9, '71 per-step features\n$-$ 21 person confounds\n'
                        '$=$ 50 modelled',
            ha='center', va='center', fontsize=FS['small'], color=INK,
            linespacing=1.8, zorder=4)
    elbow(ax, (56.6, 37.3), (60.2, 39.4), color=PRESS, lw=0.8, ms=5.5, first='h')
    elbow(ax, (56.6, 23.5), (60.2, 35.6), color=IMU, lw=0.8, ms=5.5, first='h')

    # ── raw-window bypass: the thick shortcut past feature extraction ───────
    ax.add_patch(FancyArrowPatch((20.6, 18.6), (66.6, 13.0), arrowstyle='-|>',
                                 mutation_scale=12, linewidth=4.6, color=BYPASS,
                                 zorder=2, shrinkA=1, shrinkB=1,
                                 connectionstyle='arc3,rad=0.30'))
    ax.text(41.5, 6.4, 'raw step windows, per-channel z-scored', ha='center',
            fontsize=FS['tiny'], color=GREY, style='italic', zorder=3)

    # ── models ──────────────────────────────────────────────────────────────
    box(ax, 66.6, 27.4, 12.6, 3.6, 'XGBoost', fc='white', ec=INK, lw=0.6,
        fs=FS['body'], bold=True)
    box(ax, 66.6, 22.6, 12.6, 3.6, 'SVM, k-NN', fc='white', ec=INK, lw=0.6,
        fs=FS['body'], bold=True)
    box(ax, 66.6, 11.2, 12.6, 3.6, '1D-CNN', fc='white', ec=INK, lw=0.6,
        fs=FS['body'], bold=True)
    # down the left of the feature vector, then into the LEFT side of each model box,
    # so no connector crosses a model's label
    elbow(ax, (63.2, 33.4), (66.6, 29.2), color=INK, lw=0.7, first='v')
    elbow(ax, (63.2, 33.4), (66.6, 24.4), color=INK, lw=0.7, first='v')
    ax.text(72.9, 17.8, 'per-classifier backward\nelimination $\\cdot$ grouped\nleave-one-user-out CV',
            ha='center', fontsize=FS['tiny'], color=GREY, style='italic', linespacing=1.6)

    # ── outputs ─────────────────────────────────────────────────────────────
    outs = [(39.4, 'Surface class\ngrass / loose / paved', PAVED),
            (31.8, 'Compliance\nsoft / hard', GRASS),
            (19.2, 'Evenness score\n0-100', LOOSE),
            (11.6, 'Hardness score\nP(firm)', PAVED)]
    for yy, lab, tag in outs:
        box(ax, 82.0, yy - 3.1, 16.4, 6.2, fc=OUT_FC, ec=INK, lw=0.6, r=1.0)
        ax.add_patch(FancyBboxPatch((82.0, yy - 3.1), 0.9, 6.2,
                                    boxstyle='round,pad=0,rounding_size=0.5',
                                    facecolor=tag, edgecolor='none', zorder=4))
        ax.text(90.6, yy, lab, ha='center', va='center', fontsize=FS['small'],
                color=INK, linespacing=1.6, zorder=5)
    arrow(ax, (79.2, 29.4), (82.0, 37.4), color=INK, lw=0.7, rad=-0.12)
    arrow(ax, (79.2, 25.2), (82.0, 31.8), color=INK, lw=0.7, rad=-0.12)
    arrow(ax, (79.2, 13.8), (82.0, 19.2), color=INK, lw=0.7, rad=-0.12)
    arrow(ax, (79.2, 12.4), (82.0, 11.6), color=INK, lw=0.7, rad=0.10)
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=float, default=5.478,
                    help='figure width in inches (default: acmsmall text width)')
    ap.add_argument('--insole', default='insole_42_complete_380p.png')
    ap.add_argument('--outdir', default='figures')
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    fig = build(a.width, a.insole)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out / ('fig_pipeline.' + ext), dpi=400,
                    bbox_inches='tight', pad_inches=0.02)
    print('wrote %s/fig_pipeline.{pdf,svg,png} at %.3f in wide' % (out, a.width))
