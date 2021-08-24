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

for file in files:
    print(file)
    assert file.exists()

    data = np.load(file)
    assert 'actions' in data.keys()

    actions = data['actions']
    selection = actions[:steps, :]

    plt.clf()
    plt.subplot(2, 1, 1)
    x = np.arange(selection.shape[0])
    plt.plot(x, selection[:, 0], label="motor")
    plt.title("motor")

    plt.subplot(2, 1, 2)
    plt.plot(x, selection[:, 1], label="steering")
    plt.title("steering")

    logdir = file.parents[0]
    plt.savefig(f"{logdir}/{file.stem}_Steps{selection.shape[0]}.pdf")

    plt.clf()
    bins = np.linspace(-1, +1, 25)
    plt.subplot(2, 1, 1)
    plt.hist(selection[:, 0], bins=bins, label="motor")
    plt.title("motor")

    plt.subplot(2, 1, 2)
    plt.hist(selection[:, 1], bins=bins, label="steering")
    plt.title("steering")

    logdir = file.parents[0]
    plt.savefig(f"{logdir}/{file.stem}_Steps{selection.shape[0]}_distr.pdf")