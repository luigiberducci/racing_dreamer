import numpy as np
import argparse
import pathlib
import time


parser = argparse.ArgumentParser()
parser.add_argument("--file", type=pathlib.Path, help="numpy file with actions stored as key `actions`", required=True)
parser.add_argument("--steps", type=int, help="zoom in the first steps", required=False, default=None)
args = parser.parse_args()

file = args.file
steps = args.steps
assert file.exists()

data = np.load(file)
assert 'actions' in data.keys()

actions = data['actions']
selection = actions[:steps, :]

import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
x = np.arange(selection.shape[0])
plt.plot(x, selection[:, 0], label="motor")
plt.title("motor")

plt.subplot(2, 1, 2)
plt.plot(x, selection[:, 1], label="steering")
plt.title("steering")

logdir = file.parents[0]
plt.savefig(f"{logdir}/{file.stem}_Steps{selection.shape[0]}.pdf")