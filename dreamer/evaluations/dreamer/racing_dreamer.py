import pathlib
import shutil
import time

import callbacks
from evaluations.make_env import make_single_track_env
from evaluations.racing_agent import Agent
from dream import define_config, Dreamer
import tools
import wrappers
import tensorflow as tf


def init_config(obs_type, action_dist):
    config = define_config()
    config.log_scalars = True  # note: important log_scalars True for build_models
    config.log_images = False
    config.training = False
    config.batch_length = 5
    config.batch_size = 5
    config.horizon = 5
    config.obs_type = obs_type
    config.action_dist = action_dist
    return config


def make_initialization_env(config):
    env = make_single_track_env('columbia', action_repeat=config.action_repeat, rendering=False)
    datadir = pathlib.Path('.tmp')
    writer = tf.summary.create_file_writer(str(datadir), max_queue=1000, flush_millis=20000)
    callbacks_list = []
    callbacks_list.append(lambda episodes: callbacks.save_episodes(datadir, episodes))
    env = wrappers.Collect(env, callbacks_list, config.precision)
    return writer, datadir, env


class RacingDreamer(Dreamer, Agent):

    def __init__(self, checkpoint_path, obs_type="lidar", action_dist="tanh_normal"):
        config = init_config(obs_type, action_dist)
        writer, datadir, env = make_initialization_env(config)
        random_agent = lambda o, d, s: ([env.action_space['A'].sample()], None)
        tools.simulate([random_agent for _ in range(env.n_agents)], env, config, datadir, writer,
                       prefix='prefill', episodes=10)
        # initialize model
        actspace, obspace = env.action_space, env.observation_space
        super().__init__(config, datadir, actspace, obspace, writer=None)
        try:
            shutil.rmtree(datadir)  # remove tmp directory
        except:
            pass
        self.load(checkpoint_path)
        print(f"[Info] Agent Variables: {len(self.variables)}")

    def action(self, obs, reset, state, log=False):
        if state is None:
            latent = self._dynamics.initial(len(obs[self._c.obs_type]))
            action = tf.zeros((len(obs[self._c.obs_type]), self._actdim), self._float)
        else:
            latent, action = state
        embed = self._encode(tools.preprocess(obs, self._c))
        latent, _ = self._dynamics.obs_step(latent, action, embed, log=log)
        feat = self._dynamics.get_feat(latent, log=log)
        if log:
            exp_feat = tf.expand_dims(feat, 0)
            image_pred = self._decode(exp_feat, log=log)
            reward_pred = self._reward(exp_feat, log=log)
            pcont = self._pcont(exp_feat, log=log)
            value = self._value(exp_feat, log=log)
        action = self._actor(feat, log=log).mode()
        state = (latent, action)
        return action[0], state  # because it returns action of size (1,2)

    def load(self, checkpoint):
        super(RacingDreamer, self).load(checkpoint)

    def write_activity(self, outdir: pathlib.Path, episode_id: int):
        outdir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        for name, model in zip(["encoder", "dynamics", "decoder", "reward", "actor", "value", "pcont"],
                               [self._encode, self._dynamics, self._decode, self._reward, self._actor, self._value,
                                self._pcont]):
            model.log_write(outdir / f"{name}_{episode_id}_{timestamp}.pkl")
