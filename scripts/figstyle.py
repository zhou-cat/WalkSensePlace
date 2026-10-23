"""Shared style for every figure in the paper: one palette, one set of font sizes.

Import it and call use_paper_style() before plotting; call fig.savefig(...) as usual.

PALETTE. The colours are the Okabe-Ito set, designed to stay distinguishable under
deuteranopia, protanopia and tritanopia (the three common forms of colour-vision
deficiency). It replaces the green / brown / blue-grey surface palette, whose green and
brown differ almost entirely in a hue direction that deuteranopes and protanopes cannot
see - those two classes looked nearly identical to roughly 8% of male readers. The
substitutes also differ in lightness, so they survive greyscale printing.

Checked with a Vienot CVD simulation. Minimum pairwise CIELAB distance within each
colour group, where 10 is roughly the threshold for "clearly different":

    group                normal  deuter  protan  tritan
    surface (3)            78.9    78.3    76.8    81.9
    surface + unknown      48.3    52.0    48.3    57.3
    two-class              120.2   130.8   121.7    84.2
    sources                 63.2    63.2    63.2    63.2
    feature sets            37.4    54.6    47.3    38.0

For comparison, the previous green / brown / blue-grey surface palette scored 47.2 under
normal vision but only 24.3, 16.0 and 5.2 under the three deficiencies - the three classes
converged for exactly the readers who need them separated.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Okabe-Ito ------------------------------------------------------------------
BLACK = '#000000'
ORANGE = '#E69F00'
SKY = '#56B4E9'
GREEN = '#009E73'
YELLOW = '#F0E442'
BLUE = '#0072B2'
VERMILLION = '#D55E00'
PURPLE = '#CC79A7'
GREY = '#999999'
LIGHT_GREY = '#BFBFBF'

# ---- role assignments: use these names, not raw hex, so every figure agrees -------
# Chosen by exhaustive search over the palette for the largest worst-case separation under
# normal vision AND all three deficiencies (see the module docstring for the numbers).
SURFACE = {'paved': SKY, 'grass': BLACK, 'loose': VERMILLION, 'unknown': PURPLE}
CLASS2 = {'hard': BLUE, 'soft': ORANGE}
SOURCE = {'Ground truth': BLACK, 'GIS': GREY, 'Prediction model': VERMILLION}
FEATSET = {'Full-50': GREY, 'Selected': BLUE, 'Raw windows': VERMILLION}
# continuous ramps: blue->orange is the standard CVD-safe diverging pair
RAMP_HARD = [BLUE, '#F2F2F2', VERMILLION]      # soft -> hard
RAMP_EVEN = [VERMILLION, '#F2F2F2', BLUE]      # rough -> smooth

# ---- type sizes, in points of the FINAL figure ------------------------------------
# A figure made at its printed width needs no rescaling, so these are the sizes a reader
# sees. 8 pt axis labels and 7 pt ticks are the smallest comfortably legible in print.
SIZES = dict(base=8, tick=7, label=8, title=9, legend=7.5, annot=7.5)


def use_paper_style(scale=1.0):
    """Set rcParams. scale>1 enlarges all type, for a figure printed smaller than made."""
    s = SIZES
    mpl.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 600,              # for the PNG copies; the PDFs are vector anyway
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42,              # embed real glyphs, not outlines
        'ps.fonttype': 42,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': s['base'] * scale,
        'axes.labelsize': s['label'] * scale,
        'axes.titlesize': s['title'] * scale,
        'xtick.labelsize': s['tick'] * scale,
        'ytick.labelsize': s['tick'] * scale,
        'legend.fontsize': s['legend'] * scale,
        'axes.linewidth': 0.6,
        'axes.edgecolor': '#444444',
        'axes.labelcolor': '#111111',
        'text.color': '#111111',
        'xtick.color': '#444444',
        'ytick.color': '#444444',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'axes.grid': False,
        'grid.color': '#DDDDDD',
        'grid.linewidth': 0.5,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def despine(ax, left=False):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if left:
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0)


def hatch_for(key):
    """Optional hatching, for figures that must also work in pure greyscale."""
    return {'paved': '', 'grass': '///', 'loose': '...', 'unknown': 'xxx',
            'hard': '', 'soft': '///',
            'Full-50': '', 'Selected': '///', 'Raw windows': '...'}.get(key, '')


# ACM acmsmall text width, in inches
TEXTWIDTH_IN = 5.478
