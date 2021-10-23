import glob
import re
import time

import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt

TICK_LABEL_SIZE = 18
TITLE_LABEL_SIZE = 28
AXIS_LABEL_SIZE = 26
LEGEND_FONT_SIZE = 22
FIGSIZE = (18, 3)
LINE_WIDTH = 3

CP_LABELS = {1: r"$\mu_1=0.0,\mu_2=0.0$",
             2: r"$\mu_1=0.5,\mu_2=5.0$",
             4: r"$\mu_1=1.0,\mu_2=5.0$",
             5: r"$\mu_1=5.0,\mu_2=5.0$"
             }
CP_COLORS = {1: "tab:blue",
             2: "tab:orange",
             4: "tab:green",
             5: "tab:red"
             }

parser = argparse.ArgumentParser()
parser.add_argument("--indir", type=pathlib.Path, help="where there are npz files with actions as key `actions`")
parser.add_argument("--xlabel", type=str, default="Steps")
parser.add_argument("--ylabel", type=str, default="Steering")
args = parser.parse_args()

indir = args.indir

files = [f for f in indir.glob("*npz")]

all_data = {}
for file in files:
    print(file)
    # find checkpoint to group-by it
    cp_id = int(re.findall("\d+", re.findall("checkpoint\d+", str(file))[0])[0])
    assert cp_id in CP_LABELS.keys(), f'cp labels do not cover the ids - found id={cp_id}'
    # extract data
    data = np.load(file)
    assert 'actions' in data.keys()
    actions = data['actions']
    selection = actions[:, 1]  # only steering angle
    # concatenate with prev data
    if cp_id in all_data and len(selection) > len(all_data[cp_id]):  # keep only longest episode
        all_data[cp_id] = np.concatenate([all_data[cp_id], selection])
    else:
        all_data[cp_id] = selection

plt.rc('axes', titlesize=TITLE_LABEL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=AXIS_LABEL_SIZE)  # fontsize of the x and y labels
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE)
plt.tight_layout(pad=3.5)

sort_ids = sorted(list(all_data.keys()))
bins = np.linspace(-1, +1, 10)
min_x = np.Inf
labels = set()
for i, cp_id in enumerate(sort_ids):
    min_x = min(min_x, len(all_data[cp_id]))
    ax.plot(range(len(all_data[cp_id])), all_data[cp_id], label=CP_LABELS[cp_id], color=CP_COLORS[cp_id],
            linewidth=LINE_WIDTH)
    labels.add(CP_LABELS[cp_id])
    if len(all_data[cp_id]) < 250:
        if 'Collision' not in labels:
            ax.scatter(len(all_data[cp_id]) - 1, all_data[cp_id][-1], marker='*', s=200, c=CP_COLORS[cp_id],
                       label='Collision', zorder=3)
            labels.add('Collision')
        else:
            ax.scatter(len(all_data[cp_id]) - 1, all_data[cp_id][-1], marker='*', s=200, c='r')
    # ax.set_ylim(0, 3)
    ax.set_yticks([-1, 0, +1])
    # ax.set_xticks([])
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    # ax.set_title(CP_LABELS[cp_id])
    ax.tick_params(axis='x', labelsize=TICK_LABEL_SIZE)
for spine in ['right', 'top']:
    ax.spines[spine].set_visible(False)
ax.set_xlim(0, 250)
ax.legend(loc="upper center", ncol=5, fontsize=LEGEND_FONT_SIZE, bbox_to_anchor=(0.5, 1.4))
logdir = file.parents[0]
plt.savefig(f"{logdir}/plot_continuous_actions_{time.time()}.pdf")
plt.show()
