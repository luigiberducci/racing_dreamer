import argparse
import pathlib
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tools
from evaluations.racing_agent import RacingAgent
from tools import lidar_to_image, preprocess

tf.config.run_functions_eagerly(run_eagerly=True)   # we need it to resume a model without need of same batchlen

for gpu in tf.config.experimental.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(gpu, True)

def make_log_dir(args):
  out_dir = args.outdir / f'reconstructions_dreamer_FlipObs{args.flip_obs}_{time.time()}'
  out_dir.mkdir(parents=True, exist_ok=True)
  with open(out_dir / "cmd.txt", "w+") as f:
      f.write(args.cmd)
  return out_dir


def main(args):
    rendering = False
    basedir = make_log_dir(args)
    # load offline dataset
    for file in args.indir.glob("*npz"):
        episode = np.load(file)
        true_obss, true_actions = episode['observations'].astype(np.float32), episode['actions'].astype(np.float32)
        # start reconstructing episode
        print()
        init = time.time()
        i_max = min(len(true_obss), len(true_actions))

        # extract data
        actions = true_actions[:i_max, 1:]
        lidars = true_obss[:i_max, 1:]

        # preprocess
        # lidar: ensure no NaN, clip values to max 15 meters, select the first 1080 rays
        lidars = np.nan_to_num(lidars, 30.0)[:, :1080]   # real-car 1081 rays, sim 1080
        lidars = np.clip(lidars, 0.0, 15.0) / 15.0 - 0.5
        lidars = lidars.astype(np.float32)
        if args.flip_obs:
            lidars = np.flip(lidars, axis=-1)
        # actions: revert the processing on real-hw
        # steer: clip +-0.42, scale to +-1
        # speed: clip to 0-5. m/s, scale to +-1 according to increase/decrease of speed w.r.t the prev step
        actions = np.nan_to_num(actions, 0.0)
        actions = np.clip(actions, [0.0, -0.42], [5.0, 0.42])
        speed, steer = actions[:, 0], actions[:, 1]
        speed = -1.0 + 2.0 * np.greater_equal(speed, np.concatenate([[0.0], speed[:-1]]))  # real-hw: if motor>.5, inc speed
        steer = steer / 0.42    # in this way, from -1 to +1
        actions = np.stack([speed, steer], -1)
        actions = actions.astype(np.float32)
        # prepare data
        data = {'lidar': lidars[None], 'action': actions[None]}
        print(f"[Info] Data are ready. Time: {time.time()-init:.3f} sec")

        plt.clf()
        plt.title(f"File: {file.stem}")
        plt.xlabel("Time")
        plt.ylabel("Reconstruction Error (ie, MSE(true obs, rec obs)")
        for cp in args.checkpoints:
            agent = RacingAgent("dreamer", cp, obs_type=args.obs_type, action_dist='tanh_normal')
            dreamer = agent._agent
            # reconstruction
            init = time.time()
            embed = dreamer._encode(data)
            post, prior = dreamer._dynamics.observe(embed, data['action'])    # note: we are ignoring the state
            feat = dreamer._dynamics.get_feat(post)
            image_pred = dreamer._decode(feat)
            print(f"[Info] Reconstruction completed. Time: {time.time()-init:.3f} sec")

            # iterate over each sample, plt lidar onto 2d plan, create gif
            init = time.time()
            truth = data['lidar'] + 0.5
            recon = image_pred.mode() + 0.5
            # compute reconstruction error
            mses = np.mean((truth - recon) ** 2, axis=-1)  # format BxT
            truth_imgs = lidar_to_image(truth, min_v=-0.5, max_v=0.5)
            recon_imgs = lidar_to_image(recon, min_v=-0.5, max_v=0.5, text=mses)
            print(f"[Info] Created frames. Time: {time.time()-init:.3f} sec")

            # compare each reconstruction at time t with the real lidar at time t
            init = time.time()
            frames_s2s = tf.concat([truth_imgs, recon_imgs], 3)[0]  # remove batch size
            frames_s2s = frames_s2s.numpy()

            from subprocess import Popen, PIPE
            outfile = basedir / f"{file.stem}_{cp.stem}"
            fps = 20
            h, w, c = frames_s2s[0].shape
            pxfmt = 'rgb24'
            cmd = ' '.join([
                f'ffmpeg -y -f rawvideo -vcodec rawvideo',
                f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
                f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
                f'-r {fps:.02f} -f gif - {outfile}.gif'])
            proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
            for image in frames_s2s:
                proc.stdin.write(image.tostring())
            out, err = proc.communicate()
            if proc.returncode:
                raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
            del proc
            print(f"[Info] GIF produced. Time: {time.time()-init:.3f} sec")

            plt.plot(range(mses.shape[1]), mses[0], label=f"cp: {cp.stem}")
        plt.legend()
        outfile = basedir / file.stem
        plt.savefig(str(outfile) + '.pdf')
        # todo: try preprocessing observation and see if it helps the reconstruction (e.g. median filter)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_type', type=str, choices=["lidar", "lidar_occupancy"], required=True)
    parser.add_argument('--checkpoints', nargs='+', type=pathlib.Path, required=True)
    parser.add_argument("--indir", type=pathlib.Path, help="dir containing numpy file of experimental data",
                        required=True)
    parser.add_argument('--outdir', type=pathlib.Path, required=True)
    parser.add_argument('-flip_obs', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    import sys
    t0 = time.time()
    args = parse()
    args.cmd = ' '.join(sys.argv)
    main(args)
    print(f"\n[Info] Elapsed Time: {time.time() - t0:.3f} seconds")
