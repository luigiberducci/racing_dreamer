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
FIGSIZE=(16, 4)

CP_LABELS = {1: "La=0.0,Lt=0.0",
             2: "La=0.5,Lt=5.0",
             4: "La=1.0,Lt=5.0",
             5: "La=5.0,Lt=5.0"}

parser = argparse.ArgumentParser()
parser.add_argument("--indir", type=pathlib.Path, help="where there are npz files with actions as key `actions`")
parser.add_argument("--xlabel", type=str, default="Steering")
parser.add_argument("--ylabel", type=str, default="Normalized Count")
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
    selection = actions[:, 1]   # only steering angle
    # concatenate with prev data
    if cp_id in all_data:
        all_data[cp_id] = np.concatenate([all_data[cp_id], selection])
    else:
        all_data[cp_id] = selection

plt.rc('axes', titlesize=TITLE_LABEL_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=AXIS_LABEL_SIZE)    # fontsize of the x and y labels
fig, axes = plt.subplots(nrows=1, ncols=len(all_data.keys()), figsize=FIGSIZE)
plt.tight_layout(pad=3.5)

sort_ids = sorted(list(all_data.keys()))
bins = np.linspace(-1, +1, 10)
for i, (cp_id, ax) in enumerate(zip(sort_ids, axes)):
    ax.hist(all_data[cp_id], bins=bins, label=CP_LABELS[cp_id],
            density=True, edgecolor='w')
    ax.set_ylim(0, 3)
    ax.set_xticks([-1, 0, +1])
    ax.set_yticks([])
    ax.set_xlabel(args.xlabel)
    ax.set_title(CP_LABELS[cp_id])
    ax.tick_params(axis='x', labelsize=TICK_LABEL_SIZE)
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)
    if i == 0:
        ax.set_ylabel(args.ylabel)

logdir = file.parents[0]
plt.savefig(f"{logdir}/action_distribution_{time.time()}.pdf")
plt.show()