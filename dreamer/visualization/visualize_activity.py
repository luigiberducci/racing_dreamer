import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib import colors, cm
import math
import scipy.stats as stats
from scipy.special import erf

def p(x, mu, s):
    return (1.0 / np.sqrt(2 * np.pi * (s ** 2))) * np.exp(-1.0 / (2 * (s ** 2)) * ((x - mu) ** 2))

def c(x, mu, s):
    return 1 / 2 * (1 + erf((x - mu) / (s * np.sqrt(2))))

def scan_to_xy(scan):
    # note: assume ranges have been normalized in +-0.5
    scan = scan + 0.5
    angles = np.linspace(math.pi / 2 - math.radians(270.0 / 2),
                         math.pi / 2 + math.radians(270.0 / 2),
                         scan.shape[-1])[::-1]
    x = scan * np.cos(angles)
    y = scan * np.sin(angles)
    return x, y


timestamp = "1620555767"
evaldir = "eval_dreamer_austria_lidar_1620555640.9984643"
filenames = {
    'actor': f"logs/evaluations/{evaldir}/neural_activity/actor_0_{timestamp}.pkl",
    'decoder': f"logs/evaluations/{evaldir}/neural_activity/decoder_0_{timestamp}.pkl",
    'dynamics': f"logs/evaluations/{evaldir}/neural_activity/dynamics_0_{timestamp}.pkl",
    'encoder': f"logs/evaluations/{evaldir}/neural_activity/encoder_0_{timestamp}.pkl",
    'reward': f"logs/evaluations/{evaldir}/neural_activity/reward_0_{timestamp}.pkl",
    'value': f"logs/evaluations/{evaldir}/neural_activity/value_0_{timestamp}.pkl",
    'pcont': f"logs/evaluations/{evaldir}/neural_activity/pcont_0_{timestamp}.pkl"
}

tmpdir = pathlib.Path(".tmp")
tmpdir.mkdir(exist_ok=True)

data = {}
for name, file in filenames.items():
    with open(file, "rb") as f:
        data[name] = pickle.load(f)

# 1. visualize `input` and output `features` of dynamics model
# note 1: data[actor][k]['input'] contains [prev_state, prev_action, embed]
# show `embed` and output `feat`
embeds = np.array([np.squeeze(data['dynamics'][k]['input'][-1]) for k in range(len(data['dynamics']))])
feats = np.array([np.squeeze(data['dynamics'][k]['output_feat']) for k in range(len(data['dynamics']))])
deters, stochs = feats[:, :200], feats[:, 200:]

# 2. visualize value models (reward, value, pr_cont)
rewards = np.array([np.squeeze(data['reward'][k]['output']) for k in range(len(data['reward']))])
values = np.array([np.squeeze(data['value'][k]['output']) for k in range(len(data['value']))])
pconts = np.array([np.squeeze(data['pcont'][k]['output']) for k in range(len(data['pcont']))])

# 3. visualize action model
action_means = np.array([np.squeeze(data['actor'][k]['out_mean']) for k in range(len(data['actor']))])
action_stds = np.array([np.squeeze(data['actor'][k]['out_std']) for k in range(len(data['actor']))])

deter_min, deter_max = np.min(deters), np.max(deters)
stoch_min, stoch_max = np.min(stochs), np.max(stochs)
print(f"{deter_min} {deter_max}")
print(f"{stoch_min} {stoch_max}")

plt.figure(1, (20, 10))
for t in range(embeds.shape[0]):
    embed = embeds[t]
    deter = deters[t]
    stoch = stochs[t]
    reward = rewards[t]
    value = values[t]
    pcont = pconts[t]
    action_mean = action_means[t]
    action_std = action_stds[t]

    plt.clf()

    plt.subplot(1, 4, 1)
    plt.title("embed (1080 to 2d)")
    x, y = scan_to_xy(embed)
    plt.plot(x, y)
    plt.xlim(-1, +1)
    plt.ylim(-1, +1)

    plt.subplot(2, 4, 2)
    plt.title(f"deter (200 as 20x10, norm in [{deter_min:.1f}, {deter_max:.1f}])")
    plt.axis(False)
    plt.imshow(np.reshape(deter, (20, 10)))
    plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(deter_min, deter_max), cmap=None))

    plt.subplot(2, 4, 6)
    plt.title(f"stoch (30 as 6x5, norm in [{stoch_min:.1f}, {stoch_max:.1f}])")
    plt.axis(False)
    plt.imshow(np.reshape(stoch, (6, 5)))
    plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(stoch_min, stoch_max), cmap=None))

    plt.subplot(3, 4, 3)
    plt.title("reward ~ Normal(r, 1.0)")
    mu = reward
    sigma = 1.0
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))

    plt.subplot(3, 4, 7)
    plt.title("value ~ Normal(v, 1.0)")
    mu = value
    sigma = 1.0
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))

    plt.subplot(3, 4, 11)
    plt.title("pcont ~ Bernoulli(p)")
    mu = pcont
    x = np.linspace(0, 1, 100)
    plt.plot(x, stats.bernoulli.pmf(x, mu))

    plt.subplot(2, 4, 4)
    plt.title("motor ~ Tanh(Normal(m,s))")
    mu, std = action_mean[0], action_std[0]
    x = np.linspace(-1, 1, 1000)
    y = np.diff(c(np.arctanh(x), mu, std)) / (x[1]-x[0])
    plt.plot(x, y)
    plt.xlim(-1.1, 1.1)

    plt.subplot(2, 4, 8)
    plt.title("steer ~ Tanh(Normal(m,s))")
    mu, std = action_mean[1], action_std[1]
    x = np.linspace(-1, 1, 1000)
    y = np.diff(c(np.arctanh(x), mu, std)) / (x[1] - x[0])
    plt.plot(x, y)
    plt.xlim(-1.1, 1.1)
    # plt.ylim(0.0, 1.2)

    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5, wspace=0.5)
    plt.savefig(tmpdir / f"{timestamp}_{t}.png")
