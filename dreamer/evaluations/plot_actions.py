import glob

import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=pathlib.Path, help="numpy file with actions stored as key `actions`", default=None)
parser.add_argument("--regex", type=str, help="numpy file with actions stored as key `actions`", default=None)
parser.add_argument("--steps", type=int, help="zoom in the first steps", required=False, default=None)
args = parser.parse_args()
assert args.file is not None or args.regex is not None

file = args.file
regex = args.regex
steps = args.steps

if regex is not None:
    files = [pathlib.Path(file) for file in glob.glob(regex)]
else:
    files = [file]

all_data = []
for file in files:
    print(file)
    assert file.exists()

    data = np.load(file)
    assert 'actions' in data.keys()

    actions = data['actions']
    selection = actions[:steps, 1]
    all_data = np.concatenate([all_data, selection])


plt.figure()
plt.xticks([-1, 0, +1])
plt.yticks([])
plt.xlabel("Steering Command")
plt.ylabel("Normalized Count")

bins = np.linspace(-1, +1, 25)
plt.hist(all_data, bins=bins, label="steering", density=True)

logdir = file.parents[0]
#plt.savefig(f"{logdir}/{file.stem}_Steps{selection.shape[0]}_distr.pdf")
plt.show()