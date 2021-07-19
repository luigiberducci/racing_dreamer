import argparse
import pathlib
import pandas as pd
from bagpy import bagreader
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=pathlib.Path, help="rosbag file", required=True)
args = parser.parse_args()

# read file
if not args.file.exists():
    raise FileNotFoundError(f"bag {args.file} not exist")
    exit(-1)
b = bagreader(str(args.file))

# extract topics
scan_topic = "/scan"
act_topic = "/vesc/high_level/ackermann_cmd_mux/input/nav_0"
brake_topic = "/brake_bool"
obss = b.message_by_topic(scan_topic)
actions = b.message_by_topic(act_topic)
brakes = b.message_by_topic(brake_topic)

# filter fields, prepare df
n_ranges = 1081
scan_cols = ['Time'] + [f'ranges_{i}' for i in range(n_ranges)]
act_cols = ['Time', 'drive.speed', 'drive.steering_angle']
brake_cols = ['Time', 'data']
df_obss = pd.read_csv(obss)[scan_cols]
df_actions = pd.read_csv(actions)[act_cols]
df_brakes = pd.read_csv(brakes)[brake_cols]

# consider data from 1st action till 1st aeb
init_time = df_actions['Time'].min()
stop_time = df_brakes[df_brakes.data]['Time'].min()

df_actions = df_actions[(init_time <= df_actions.Time) & (df_actions.Time < stop_time)]
stop_time = min(stop_time, df_actions.Time.max())

df_obss = df_obss[(init_time <= df_obss.Time) & (df_obss.Time < stop_time)]
df_brakes = df_brakes[(init_time <= df_brakes.Time) & (df_brakes.Time < stop_time)]

# save to numpy
np_file = str(args.file.parent / '../np' / args.file.name).replace('.bag', '')
np.savez(f"{np_file}", actions=df_actions.to_numpy(),
                       observations=df_obss.to_numpy(),
                       brakes=df_brakes.to_numpy())
print(f"saved {len(df_obss)} scans, {len(df_actions)} actions, {len(df_brakes)} brakes,\ninto {np_file}")





