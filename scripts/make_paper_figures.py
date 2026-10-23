"""Rebuild the paper's data charts as vector PDFs, from the same numbers as the originals.

    python scripts/make_paper_figures.py --out figures/ [--comp-dir GPS/]

Produces, each sized for the width it is printed at, with type at final size:
    step_by_surface.pdf        step counts per surface
    user_surface_mix.pdf       each participant's surface mix
    feat_sel_improvement.pdf   macro-F1, full vs selected features, both tasks
    comp_4walk.pdf             per-walk surface composition by source

Every figure also gets a .png at 600 dpi for slides. Colours come from figstyle.py.
Numbers are declared in this file (or read from map_prop_long_*.csv for the composition
chart), so updating a result means editing one line here, not re-exporting from Tableau.
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import (SURFACE, SOURCE, FEATSET, TEXTWIDTH_IN, use_paper_style,
                      despine)                                        # noqa: E402

TW = TEXTWIDTH_IN

# ----------------------------------------------------------------- the numbers
STEPS = {'grass': 5175, 'loose': 2851, 'paved': 13187}
ORDER = ['grass', 'loose', 'paved']

# participant -> (grass, loose, paved); labels are pseudonyms, as in the paper
USERS = {'User 1': (1753, 890, 3230), 'User 2': (1642, 1051, 4787),
         'User 3': (1594, 853, 4313), 'User 4': (186, 57, 857)}

# macro-F1, leave-one-user-out: (full-50, selected). CNN has raw windows only.
FEATSEL = {
    '3-class': {'XGBoost': (0.528, 0.575), 'SVM-linear': (0.502, 0.583),
                'SVM-poly': (0.482, 0.534), 'k-NN': (0.401, 0.524),
                '1D-CNN': (None, 0.510)},     # NOTE: the paper now reports 0.549 - see README
    '2-class': {'XGBoost': (0.783, 0.822), 'SVM-linear': (0.627, 0.734),
                'SVM-poly': (0.722, 0.760), 'k-NN': (0.610, 0.781),
                '1D-CNN': (None, 0.693)},
}
MODEL_ORDER = ['XGBoost', 'SVM-linear', 'SVM-poly', 'k-NN', '1D-CNN']


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name + '.pdf'))
    fig.savefig(os.path.join(out, name + '.png'), dpi=600)
    plt.close(fig)
    print(f'  {name}.pdf  ({fig.get_size_inches()[0]:.2f} x {fig.get_size_inches()[1]:.2f} in)')


# ----------------------------------------------------------------- 1. step counts
def fig_step_counts(out):
    # printed at 0.40\linewidth in the paper, so make it that wide and scale type up
    use_paper_style(scale=2.3)
    w = 0.40 * TW
    fig, ax = plt.subplots(figsize=(w * 2.3, w * 2.3 * 0.95))
    tot = sum(STEPS.values())
    for i, k in enumerate(ORDER):
        ax.bar(i, STEPS[k], width=0.68, color=SURFACE[k], zorder=2)
        ax.annotate(f'{STEPS[k]:,}\n{100*STEPS[k]/tot:.1f}%', (i, STEPS[k]),
                    textcoords='offset points', xytext=(0, 4), ha='center', va='bottom')
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Steps'); ax.set_ylim(0, 15500)
    ax.set_yticks(range(0, 15001, 5000))
    ax.set_yticklabels([f'{v//1000}K' if v else '0' for v in range(0, 15001, 5000)])
    ax.yaxis.grid(True, zorder=0); despine(ax)
    fig.tight_layout(pad=0.2)
    save(fig, out, 'step_by_surface')


# ----------------------------------------------------------------- 2. per-user mix
def fig_user_mix(out):
    use_paper_style(scale=1.9)
    w = 0.56 * TW
    rows = [('All participants', tuple(STEPS[k] for k in ORDER))]
    rows += sorted(USERS.items(), key=lambda kv: -sum(kv[1]))
    fig, ax = plt.subplots(figsize=(w * 1.9, (0.42 * len(rows) + 0.75) * 1.9))
    ypos = np.arange(len(rows))[::-1]
    for y, (name, counts) in zip(ypos, rows):
        tot = sum(counts); left = 0.0
        for k, c in zip(ORDER, counts):
            pct = 100 * c / tot
            ax.barh(y, pct, left=left, height=0.62, color=SURFACE[k],
                    edgecolor='white', linewidth=0.8, zorder=2)
            if pct >= 8:                       # fits inside the segment
                ax.text(left + pct / 2, y, f'{pct:.0f}%', ha='center', va='center',
                        color='white' if k in ('paved', 'grass') else '#111111',
                        fontweight='medium')
            elif pct >= 1:                     # too narrow: label above, with a leader
                ax.annotate(f'{pct:.0f}%', (left + pct / 2, y + 0.33),
                            ha='center', va='bottom', color='#111111',
                            fontsize=plt.rcParams['font.size'] * 0.92)
            left += pct
    ax.set_yticks(ypos)
    ax.set_yticklabels([f'{n}\n(n={sum(c):,})' for n, c in rows])
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 100); ax.set_xlabel('Share of steps (%)')
    ax.set_xticks(range(0, 101, 25)); despine(ax, left=True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SURFACE[k]) for k in ORDER]
    ax.legend(handles, ORDER, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              ncol=3, handlelength=1.1, columnspacing=1.4)
    fig.tight_layout(pad=0.2)
    save(fig, out, 'user_surface_mix')


# ------------------------------------------------- 3. feature-selection improvement
def fig_featsel(out):
    use_paper_style(scale=1.35)
    fig, axes = plt.subplots(1, 2, figsize=(TW * 1.35, TW * 1.35 * 0.42), sharey=True)
    for ax, task in zip(axes, ('3-class', '2-class')):
        d = FEATSEL[task]
        for i, m in enumerate(MODEL_ORDER):
            full, sel = d[m]
            if full is None:                       # CNN: one bar, raw windows
                ax.bar(i, sel, width=0.52, color=FEATSET['Raw windows'], zorder=2)
                ax.annotate(f'{sel:.3f}', (i, sel), textcoords='offset points',
                            xytext=(0, 3), ha='center', va='bottom')
            else:
                ax.bar(i - 0.19, full, width=0.36, color=FEATSET['Full-50'], zorder=2)
                ax.bar(i + 0.19, sel, width=0.36, color=FEATSET['Selected'], zorder=2)
                ax.annotate(f'$+${sel-full:.3f}', (i + 0.19, sel),
                            textcoords='offset points', xytext=(0, 3), ha='center',
                            va='bottom')
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(MODEL_ORDER, rotation=20, ha='right')
        ax.tick_params(axis='x', length=0)
        ax.set_title(task, pad=4)
        ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.yaxis.grid(True, zorder=0); despine(ax)
    axes[0].set_ylabel('Macro-F1 (leave-one-user-out)')
    handles = [plt.Rectangle((0, 0), 1, 1, color=FEATSET[k])
               for k in ('Full-50', 'Selected', 'Raw windows')]
    fig.legend(handles, ['All 50 features', 'Selected subset', 'Raw windows (CNN)'],
               loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
               handlelength=1.1, columnspacing=1.6)
    fig.tight_layout(pad=0.2)
    save(fig, out, 'feat_sel_improvement')


# ----------------------------------------------------------------- 4. composition
def fig_composition(out, comp_dir):
    files = sorted(glob.glob(os.path.join(comp_dir, 'map_prop_long_*.csv')))
    if not files:
        print(f'  [skip] comp_4walk: no map_prop_long_*.csv in {comp_dir}')
        return
    use_paper_style(scale=1.35)
    walks = {}
    for p in files:
        m = re.search(r'(\d{6})\.csv$', p)
        d = {}
        for r in csv.DictReader(open(p, encoding='utf-8-sig')):
            v = str(r['Share']).strip().rstrip('%')
            d[(r['Surface'].strip().lower(), r['Source'].strip())] = \
                float(v) / (100 if str(r['Share']).strip().endswith('%') else 1)
        walks[m.group(1) if m else os.path.basename(p)] = d
    srcs = ['Ground truth', 'GIS', 'Prediction model']
    surf = [s for s in ('paved', 'grass', 'loose', 'unknown')
            if any(d.get((s, k), 0) > 0 for d in walks.values() for k in srcs)]
    for w, d in walks.items():
        for k in srcs:
            t = sum(d.get((s, k), 0) for s in surf)
            if abs(t - 1) > 0.01:
                print(f'  [check] walk {w}, {k}: shares sum to {t:.3f}')
    fig, axes = plt.subplots(1, len(walks), figsize=(TW * 1.35, TW * 1.35 * 0.40),
                             sharey=True, gridspec_kw={'wspace': 0.10})
    bw = 0.26
    for ax, (w, d) in zip(np.atleast_1d(axes), walks.items()):
        for j, k in enumerate(srcs):
            ax.bar([i + (j - 1) * bw for i in range(len(surf))],
                   [d.get((s, k), 0) for s in surf], width=bw * 0.9,
                   color=SOURCE[k], zorder=2)
        ax.set_xticks(range(len(surf)))
        ax.set_xticklabels(surf, rotation=25, ha='right')
        ax.tick_params(axis='x', length=0)
        ax.set_title(w, pad=4)
        ax.set_ylim(0, 1); ax.set_yticks(np.arange(0, 1.01, 0.25))
        ax.yaxis.grid(True, zorder=0); despine(ax)
        ax.set_xlim(-0.55, len(surf) - 0.45)
    np.atleast_1d(axes)[0].set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    np.atleast_1d(axes)[0].set_ylabel('Share of steps')
    handles = [plt.Rectangle((0, 0), 1, 1, color=SOURCE[k]) for k in srcs]
    fig.legend(handles, srcs, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3,
               handlelength=1.1, columnspacing=1.6)
    fig.tight_layout(pad=0.2)
    save(fig, out, 'comp_4walk')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures')
    ap.add_argument('--comp-dir', default='.', help='folder with map_prop_long_*.csv')
    a = ap.parse_args()
    print('writing figures:')
    fig_step_counts(a.out)
    fig_user_mix(a.out)
    fig_featsel(a.out)
    fig_composition(a.out, a.comp_dir)


if __name__ == '__main__':
    main()
