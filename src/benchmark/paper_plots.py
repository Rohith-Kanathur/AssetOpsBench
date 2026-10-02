"""Paper-style vertical grouped bars with explicit sample-SD error bars."""
from __future__ import annotations

import math

PAPER_COLORS = ['#4684b2', '#cfb5d5', '#ac98d1', '#288db4', '#adbfd9']
CRITERION_LABELS = ['Task Completion ↑', 'Data Retrieval\nAccuracy ↑', 'Result Verification ↑']
CRITERION_KEYS = ['task_completion', 'data_retrieval_accuracy', 'generalized_result_verification']


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'axes.edgecolor': '#303030',
                         'text.color': '#222222', 'axes.labelcolor': '#222222',
                         'xtick.color': '#222222', 'ytick.color': '#222222',
                         'svg.hashsalt': 'assetopsbench-paper-plots'})
    return plt


def style_axis(ax, *, ymax=110, ylabel='Rate (%)'):
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(.7)
    ax.tick_params(axis='both', length=3, width=.7)
    ax.grid(False)


def grouped_bars(ax, labels, model_names, values, errors, colors):
    """Values/errors are [model][group]; unavailable means are not drawn."""
    import numpy as np
    width = .8 / len(model_names)
    highest = 100.0
    for means, sds in zip(values, errors):
        highest = max([highest, *(value + (sd or 0) for value, sd in zip(means, sds) if value is not None)])
    ymax = max(110, math.ceil((highest + 10) / 10) * 10)
    for index, (name, color) in enumerate(zip(model_names, colors)):
        for group, (value, sd) in enumerate(zip(values[index], errors[index])):
            if value is None:
                continue
            x = group + (index - (len(model_names) - 1) / 2) * width
            ax.bar(x, value, width=width * .93, color=color, edgecolor='#303030', linewidth=.65,
                   label=name if group == 0 else None, zorder=2)
            if sd is not None:
                ax.errorbar(x, value, yerr=sd, fmt='none', ecolor='#303030', elinewidth=.85,
                            capsize=2.5, capthick=.85, zorder=3)
            ax.text(x, value + (sd or 0) + ymax * .018, f'{value:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_xlim(-.55, len(labels) - .45)
    style_axis(ax, ymax=ymax)


def shared_legend(fig, model_names, colors, *, repeated=True, caption=None):
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=color, edgecolor='#303030', linewidth=.65) for color in colors],
               model_names, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .99), columnspacing=2.0, handlelength=1.8)
    description = 'Bars: mean rate · whiskers: sample SD across execution repetitions' if repeated else 'Bars: observed criterion rate'
    fig.text(.5, .025, ((caption + '\n') if caption else '') + description, ha='center', fontsize=9, color='#444444')


def criterion_figure(panels, model_names, colors=None, *, repeated=True, caption=None):
    """One panel per cohort, each grouping the three paper criteria on x."""
    plt = plotting()
    colors = colors or PAPER_COLORS[:len(model_names)]
    fig, axes = plt.subplots(1, len(panels), figsize=(8.2 * len(panels), 5.4), squeeze=False)
    for ax, (title, rows) in zip(axes[0], panels):
        values = [[None if row['metrics'][key]['mean'] is None else row['metrics'][key]['mean'] * 100
                   for key in CRITERION_KEYS] for row in rows]
        errors = [[None if row['metrics'][key]['sd'] is None else row['metrics'][key]['sd'] * 100
                   for key in CRITERION_KEYS] for row in rows]
        grouped_bars(ax, CRITERION_LABELS, model_names, values, errors, colors)
        if title:
            ax.set_title(title, pad=12, fontsize=12)
    # Cohort bars must use the same vertical scale for a fair visual comparison.
    ymax = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(0, ymax)
    shared_legend(fig, model_names, colors, repeated=repeated, caption=caption)
    fig.subplots_adjust(left=.07, right=.985, bottom=.20, top=.77, wspace=.22)
    return fig


def strict_pass_figure(cohorts, model_names, colors=None, *, caption=None):
    """Each cohort is an x group; model colors match the criterion figure."""
    plt = plotting()
    colors = colors or PAPER_COLORS[:len(model_names)]
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    values = [[None if metrics[index]['mean'] is None else metrics[index]['mean'] * 100
               for _, metrics in cohorts] for index in range(len(model_names))]
    errors = [[None if metrics[index]['sd'] is None else metrics[index]['sd'] * 100
               for _, metrics in cohorts] for index in range(len(model_names))]
    grouped_bars(ax, [title for title, _ in cohorts], model_names, values, errors, colors)
    ax.set_title('Strict Pass Rate ↑', pad=12)
    shared_legend(fig, model_names, colors, caption=caption)
    fig.subplots_adjust(left=.085, right=.985, bottom=.16, top=.77)
    return fig


def save_figure(fig, dest, name):
    plt = plotting()
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / f'{name}.png', dpi=180, bbox_inches='tight')
    svg = dest / f'{name}.svg'
    fig.savefig(svg, bbox_inches='tight', metadata={'Date': None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)
