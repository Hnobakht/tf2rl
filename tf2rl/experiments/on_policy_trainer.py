import time

import numpy as np
import tensorflow as tf

from cpprb import ReplayBuffer

from tf2rl.experiments.trainer import Trainer, with_summary_writer
from tf2rl.misc.get_replay_buffer import get_replay_buffer, get_default_rb_dict
from tf2rl.misc.discount_cumsum import discount_cumsum
from tf2rl.envs.utils import is_discrete


class OnPolicyTrainer(Trainer):
    """Collect fixed-length rollouts and update an on-policy agent."""
    def __init__(self, *args, **kwargs):
        """Configure training, replay storage, evaluation, and checkpoint handling."""
        super().__init__(*args, **kwargs)

    @with_summary_writer
    def __call__(self):
        # Prepare buffer
        """Run the training loop, including data collection, policy updates, and evaluation."""
        self.replay_buffer = get_replay_buffer(
            self._policy, self._env)
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
        obs, _ = self._env.reset()

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
                next_obs, reward, terminated, truncated, _ = self._env.step(env_act)
                done = terminated or truncated
                if self._show_progress:
                    self._env.render()

                episode_steps += 1
                total_steps += 1
                episode_return += reward

                done_flag = terminated
                self.local_buffer.add(
                    obs=obs, act=act, next_obs=next_obs,
                    rew=reward, done=done_flag, logp=logp, val=val)
                obs = next_obs

                if done or episode_steps == self._episode_max_steps:
                    tf.summary.experimental.set_step(total_steps)
                    bootstrap = 0. if terminated else self._policy.get_action_and_val(next_obs)[2]
                    self.finish_horizon(last_val=bootstrap)
                    obs, _ = self._env.reset()
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

            self.finish_horizon(last_val=self._policy.get_action_and_val(obs)[2])
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
                    samples = self.replay_buffer.get_all_transitions()
                    order = np.random.permutation(len(samples["obs"]))
                    samples = {key: value[order] for key, value in samples.items()}
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
                                            states=samples["obs"][target],
                                            actions=samples["act"][target],
                                            advantages=adv[target],
                                            logp_olds=samples["logp"][target],
                                            returns=samples["ret"][target])
                            if (kl > self._policy.target_kl).any():
                                print("[DEBUG] The KL ({}) has reached the limit at batch: {} and  epoch: {}."
                                      .format(kl.mean(), idx, idx_epoch))
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

        tf.summary.flush()

    def finish_horizon(self, last_val=0):
        """Finish the rollout by computing return targets and policy advantages."""
        self.local_buffer.on_episode_end()
        samples = self.local_buffer.get_all_transitions()
        rews = np.append(samples["rew"], last_val)
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
            obs=samples["obs"], act=samples["act"], done=samples["done"],
            ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))
        self.local_buffer.clear()

    def evaluate_policy(self, total_steps):
        """Evaluate the policy in the test environment and return its performance."""
        return super().evaluate_policy(total_steps)
