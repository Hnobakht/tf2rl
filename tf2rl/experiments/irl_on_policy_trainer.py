import os
import time

import numpy as np
import tensorflow as tf
import random

from cpprb import ReplayBuffer

from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.utils import save_path, frames_to_gif
from tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from tf2rl.misc.discount_cumsum import discount_cumsum
from tf2rl.envs.utils import is_discrete


class IRLOnPolicyTrainer(Trainer):
    def __init__(self,
            policy,
            env,
            args,
            irl,
            expert_obs,
            expert_next_obs,
            expert_act,
            test_env=None,
            is_perb_rew = False,
            is_episodic = False,
            is_short_memory = False,
            discriminator_noise=0,
            **kwargs):
        self._irl = irl
        args.dir_suffix = self._irl.policy_name + args.dir_suffix
        super().__init__(policy, env, args, test_env, is_episodic=is_episodic, **kwargs)
        # TODO: Add assertion to check dimention of expert demos and current policy, env is the same
        self._expert_obs = expert_obs
        self._expert_next_obs = expert_next_obs
        self._expert_act = expert_act
        # Minus one to get next obs
        self._random_range = range(expert_obs.shape[0])
        self._is_perb_rew = is_perb_rew
        self.disc_noise = discriminator_noise
        self._irl_chechpoint()

    def _irl_chechpoint(self):
        self._checkpoint_irl = tf.train.Checkpoint(policy=self._irl)
        self.checkpoint_manager_irl = tf.train.CheckpointManager(
            self._checkpoint_irl, directory=self._output_dir + "/irl", max_to_keep=10)

    def __call__(self):
        # Prepare buffer
        self.replay_buffer = get_replay_buffer(
            self._policy, self._env, self._irl)
        kwargs_local_buf = get_default_rb_dict(
            size=self._policy.horizon, env=self._env)
        kwargs_local_buf["env_dict"]["logp"] = {}
        kwargs_local_buf["env_dict"]["val"] = {}
        if is_discrete(self._env.action_space):
            kwargs_local_buf["env_dict"]["act"]["dtype"] = np.int32
        self.local_buffer = ReplayBuffer(**kwargs_local_buf)

        episode_steps = 0
        episode_return = 0
        episode_start_time = time.time()
        total_steps = np.array(0, dtype=np.int32)
        n_epoisode = 0
        obs = self._env.reset()

        tf.summary.experimental.set_step(total_steps)
        while total_steps < self._max_steps:
            # Collect samples
            for _ in range(self._policy.horizon):
                if self._normalize_obs:
                    obs = self._obs_normalizer(obs, update=False)
                act, logp, val = self._policy.get_action_and_val(obs)
                if not is_discrete(self._env.action_space):
                    env_act = np.clip(act, self._env.action_space.low, self._env.action_space.high)
                else:
                    env_act = act
                next_obs, reward, done, _ = self._env.step(env_act)
                if self._show_progress:
                    self._env.render()

                episode_steps += 1
                total_steps += 1
                episode_return += reward

                done_flag = done
                if (hasattr(self._env, "_max_episode_steps") and
                    episode_steps == self._env._max_episode_steps):
                    done_flag = False
                self.local_buffer.add(
                    obs=obs, act=act, next_obs=next_obs,
                    rew=reward, done=done_flag, logp=logp, val=val)
                obs = next_obs

                if done or episode_steps == self._episode_max_steps:
                    tf.summary.experimental.set_step(total_steps)
                    self.finish_horizon()
                    obs = self._env.reset()
                    n_epoisode += 1
                    fps = episode_steps / (time.time() - episode_start_time)
                    self.logger.info(
                        "Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                            n_epoisode, int(total_steps), episode_steps, episode_return, fps))
                    tf.summary.scalar(name="Common/training_return", data=episode_return)
                    tf.summary.scalar(name="Common/training_episode_length", data=episode_steps)
                    tf.summary.scalar(name="Common/fps", data=fps)
                    episode_steps = 0
                    episode_return = 0
                    episode_start_time = time.time()

                if total_steps % self._test_interval == 0:
                    avg_test_return, avg_test_steps = self.evaluate_policy(total_steps)
                    self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                        total_steps, avg_test_return, self._test_episodes))
                    tf.summary.scalar(
                        name="Common/average_test_return", data=avg_test_return)
                    tf.summary.scalar(
                        name="Common/average_test_episode_length", data=avg_test_steps)
                    self.writer.flush()

                if total_steps % self._save_model_interval == 0:
                    self.checkpoint_manager.save()

            self.finish_horizon(last_val=val)
            print("[DEBUG] Time to grad...")

            tf.summary.experimental.set_step(total_steps)

            # Train actor critic
            if self._policy.normalize_adv:
                samples = self.replay_buffer.get_all_transitions()
                mean_adv = np.mean(samples["adv"])
                std_adv = np.std(samples["adv"])
                # Update normalizer
                if self._normalize_obs:
                    self._obs_normalizer.experience(samples["obs"])
            with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                br = False
                for idx_epoch in range(self._policy.n_epoch):
                    samples = self.replay_buffer._encode_sample(
                        np.random.permutation(self._policy.horizon))
                    if self._normalize_obs:
                        samples["obs"] = self._obs_normalizer(samples["obs"], update=False)
                    if self._policy.normalize_adv:
                        adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)
                    else:
                        adv = samples["adv"]
                    if not br:
                        for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                            target = slice(idx * self._policy.batch_size,
                                           (idx + 1) * self._policy.batch_size)
                            _, _, kl = self._policy.train(
                            # _, kl = self._policy.train_actor(
                                            states=samples["obs"][target],
                                            actions=samples["act"][target],
                                            advantages=adv[target],
                                            logp_olds=samples["logp"][target],
                                            returns=samples["ret"][target])
                            if (kl > self._policy.target_kl).any():
                                print("[DEBUG] The KL ({}) has reached the limit at batch: {} and  epoch: {}."
                                      .format(kl.mean(), idx, idx_epoch))
                                # br = True
                                break
                    # for idx_critic in range(int(self._policy.horizon / self._policy.batch_size)):
                    #     target = slice(idx_critic * self._policy.batch_size,
                    #                    (idx_critic + 1) * self._policy.batch_size)
                    #     self._policy.train_critic(
                    #                     states=samples["obs"][target],
                    #                     returns=samples["ret"][target])
                    # TODO: Should i continue with this epoch after the kl pass the target?
                    if (kl > self._policy.target_kl).any():
                        break
                for _ in range(self._irl.n_training):
                    samples = self.replay_buffer._encode_sample(np.random.permutation(self._policy.horizon))
                    noise_size = int(self._irl.batch_size * self.disc_noise)
                    noise = self.replay_buffer.sample(noise_size);

                    for idx in range(int(self._policy.horizon / self._irl.batch_size)):
                        target = slice(idx * self._irl.batch_size, (idx + 1) * self._irl.batch_size)

                        indices = random.sample(self._random_range, self._irl.batch_size - noise_size)
                        indices = np.array(indices)
                        assert indices.shape[0] + noise_size == self._irl.batch_size, \
                            "The size of expert batch does not match the IRL batch size."
                        if "EQL" in self._irl.policy_name:
                            self._irl.train(
                                agent_states=samples["obs"][target],
                                agent_acts=samples["act"][target],
                                agent_next_states=samples["next_obs"][target],
                                expert_states=np.concatenate([self._expert_obs[indices], noise["obs"]]),
                                expert_acts=np.concatenate([self._expert_act[indices], noise["act"]]),
                                expert_next_states=np.concatenate([self._expert_next_obs[indices], noise["next_obs"]]),
                                itr=total_steps)
                        else:
                            self._irl.train(
                                agent_states=samples["obs"][target],
                                agent_acts=samples["act"][target],
                                agent_next_states = ["next_obs"][target],
                                expert_states=np.concatenate([self._expert_obs[indices], noise["obs"]]),
                                expert_acts=np.concatenate([self._expert_act[indices], noise["act"]]),
                                expert_next_states=np.concatenate([self._expert_next_obs[indices], noise["next_obs"]])
                            )

        tf.summary.flush()

    def finish_horizon(self, last_val=0):
        self.local_buffer.on_episode_end()
        samples = self.local_buffer._encode_sample(
            np.arange(self.local_buffer.get_stored_size()))
        # rews = np.append(samples["rew"], last_val)
        # TODO: Check this to see of negetive this is ok
        rews  = np.append(self._irl.inference(samples["obs"], samples["act"], samples["next_obs"]), last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]
        if self._policy.enable_gae:
            advs = discount_cumsum(deltas, self._policy.discount * self._policy.lam)
        else:
            advs = deltas

        # Rewards-to-go, to be targets for the value function
        rets = discount_cumsum(rews, self._policy.discount)[:-1]
        self.replay_buffer.add(
            obs=samples["obs"], act=samples["act"], next_obs=samples["next_obs"],
            done=samples["done"], ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))

        # TODO: It may be redundant. or even wrong
        self.replay_buffer.on_episode_end()
        self.local_buffer.clear()

    def evaluate_policy(self, total_steps):
        avg_test_return = 0.
        avg_test_steps = 0
        if self._save_test_path:
            replay_buffer = get_replay_buffer(
                self._policy, self._test_env, size=self._episode_max_steps)
        for i in range(self._test_episodes):
            episode_return = 0.
            frames = []
            obs = self._test_env.reset()
            avg_test_steps += 1
            for _ in range(self._episode_max_steps):
                if self._normalize_obs:
                    obs = self._obs_normalizer(obs, update=False)
                act, _ = self._policy.get_action(obs, test=True)
                act = (act if is_discrete(self._env.action_space) else
                       np.clip(act, self._env.action_space.low, self._env.action_space.high))
                next_obs, reward, done, _ = self._test_env.step(act)
                avg_test_steps += 1
                if self._save_test_path:
                    replay_buffer.add(
                        obs=obs, act=act, next_obs=next_obs,
                        rew=reward, done=done)

                if self._save_test_movie:
                    frames.append(self._test_env.render(mode='rgb_array'))
                elif self._show_test_progress:
                    self._test_env.render()
                episode_return += reward
                obs = next_obs
                if done:
                    break
            prefix = "step_{0:08d}_epi_{1:02d}_return_{2:010.4f}".format(
                total_steps, i, episode_return)
            if self._save_test_path:
                save_path(replay_buffer.sample(self._episode_max_steps),
                          os.path.join(self._output_dir, prefix + ".pkl"))
                replay_buffer.clear()
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            avg_test_return += episode_return
        if self._show_test_images:
            images = tf.cast(
                tf.expand_dims(np.array(obs).transpose(2, 0, 1), axis=3),
                tf.uint8)
            tf.summary.image('train/input_img', images, )
        return avg_test_return / self._test_episodes, avg_test_steps / self._test_episodes
