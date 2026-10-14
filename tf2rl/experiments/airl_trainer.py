import time

import numpy as np
import random
import tensorflow as tf

from tf2rl.misc.get_replay_buffer import get_replay_buffer
# from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.algos.airl import AIRL
from tf2rl.algos.sac import SAC
from tf2rl.algos.airl_sac import AIRL_SAC

class AIRLTrainer(IRLTrainer):
    def __init__(
            self,
            policy,
            env,
            args,
            irl,
            expert_obs,
            expert_next_obs,
            expert_act,
            expert_logp=None,
            test_env=None,
            **kwargs):
        assert "airl" in irl.name, "[ERROR] This trainer is only for the AIRL algorithms."
        assert expert_logp is not None
        self._irl = irl
        args.dir_suffix = self._irl.policy_name + args.dir_suffix
        super().__init__(policy, env, args, irl, expert_obs, expert_next_obs, expert_act, test_env, **kwargs)
        assert isinstance(self._policy, SAC), "Only works with stoastic policies"
        # TODO: Add assertion to check dimention of expert demos and current policy, env is the same
        self._expert_obs = expert_obs
        self._expert_next_obs = expert_next_obs
        self._expert_act = expert_act
        self._expert_logp = expert_logp
        # Minus one to get next obs
        self._random_range = range(expert_obs.shape[0])
        self._noise_size = int(self._irl.batch_size * self.disc_noise)
        self._is_sac = isinstance(self._policy, SAC) or isinstance(self._policy, AIRL_SAC)
        self._is_airl_sac = isinstance(self._policy, AIRL_SAC)

    def __call__(self):
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_return_D = 0
        episode_start_time = time.perf_counter()
        n_episode = 0

        replay_buffer = get_replay_buffer(
            policy=self._policy, env=self._env, irl=self._irl,
            use_prioritized_rb=self._use_prioritized_rb,
            use_nstep_rb=self._use_nstep_rb, n_step=self._n_step)

        obs = np.float32(self._env.reset())

        while total_steps < self._max_steps:
            if total_steps < self._policy.n_warmup:
                action = self._env.action_space.sample()
            else:
                if self._is_sac:
                    action, logp = self._policy.get_action(obs, test=True)
                else:
                    action = self._policy.get_action(obs, test=True)

            next_obs, reward, done, _ = self._env.step(action)
            done = False if self._is_episodic else done
            next_obs, reward = np.float32(next_obs), np.float32(reward)
            if self._show_progress and total_steps % self._show_progress_interval:
                self._env.render()
            episode_steps += 1
            episode_return += reward

            if not total_steps < self._policy.n_warmup: # To not to ove trace the @tf
                if "EQL" in self._irl.policy_name and self._irl.atol_starting_step:
                    raise NotImplemented
                    episode_return_D += self._irl.inference(obs, action, next_obs,
                                                            total_steps >  self._irl.atol_starting_step)[0][0]
                else:
                    episode_return_D += self._irl.inference(obs, action, next_obs, logp if self._is_sac else None)[0][0]

            done_flag = done
            if (hasattr(self._env, "_max_episode_steps") and
                    episode_steps == self._env._max_episode_steps):
                done_flag = False

            if total_steps < self._policy.n_warmup:
                logp_dic = self._policy.get_logp(obs)
            else:
                logp_dic = logp

            data = {"obs": obs, "act": action, "next_obs": next_obs,
                    "rew": reward, "done": done_flag, "logp": logp_dic}
            replay_buffer.add(**data)
            obs = next_obs

            total_steps += 1
            tf.summary.experimental.set_step(total_steps)

            if done or episode_steps == self._episode_max_steps:
                replay_buffer.on_episode_end()
                obs = np.float32(self._env.reset())

                n_episode += 1
                fps = episode_steps / (time.perf_counter() - episode_start_time)
                self.logger.info(
                    "Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                        n_episode, int(total_steps), episode_steps, episode_return, fps))
                tf.summary.scalar(
                    name="Common/training_return", data=episode_return)
                tf.summary.scalar(
                    name="Common/training_adverseral_return", data=episode_return_D / episode_steps)

                episode_steps = 0
                episode_return = 0
                episode_return_D = 0
                episode_start_time = time.perf_counter()

            if total_steps < self._policy.n_warmup:
                continue

            if total_steps % self._policy.update_interval == 0:
                samples = replay_buffer.sample(self._policy.batch_size)
                if "EQL" in self._irl.policy_name and self._irl.atol_starting_step:
                    raise NotImplemented("No ATOL for AIRL!")
                    rew = self._irl.inference(
                        samples["obs"],
                        samples["act"],
                        samples["next_obs"],
                        total_steps > self._irl.atol_starting_step)
                else:
                    if not self._is_airl_sac:
                        rew = self._irl.inference(
                            samples["obs"],
                            samples["act"],
                            samples["next_obs"],
                            samples["logp"])
                    else:
                        raise NotImplemented("AIRL_SAC generator is not ready.")
                        rew = self._irl.inference_bellman(
                            samples["obs"],
                            samples["act"],
                            samples["next_obs"],
                            np.expand_dims(self._policy.get_logp(samples["next_obs"]), 1))
                with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                    self._policy.train(
                        samples["obs"], samples["act"], samples["next_obs"],
                        rew, samples["done"],
                        None if not self._use_prioritized_rb else samples["weights"])
                    if self._use_prioritized_rb:
                        td_error = self._policy.compute_td_error(
                            samples["obs"], samples["act"], samples["next_obs"],
                            rew, samples["done"])
                        replay_buffer.update_priorities(
                            samples["indexes"], np.abs(td_error) + 1e-6)

                    # Train AIRL
                    for _ in range(self._irl.n_training):
                        samples = replay_buffer.sample(self._irl.batch_size)
                        noise = replay_buffer.sample(self._noise_size);
                        self._irl.train(
                            **self._get_train_kwargs(
                                samples,
                                noise,
                                itr=total_steps))


            if total_steps % self._test_interval == 0:
                avg_test_return, avg_test_steps = self.evaluate_policy(total_steps)
                self.logger.info("Evaluation Total Steps: {0: 7} Average Reward {1: 5.4f} over {2: 2} episodes".format(
                    total_steps, avg_test_return, self._test_episodes))
                tf.summary.scalar(
                    name="Common/average_test_return", data=avg_test_return)
                tf.summary.scalar(
                    name="Common/average_test_episode_length", data=avg_test_steps)
                tf.summary.scalar(
                    name="Common/fps", data=fps)
                self.writer.flush()

            if total_steps % self._save_model_interval == 0:
                self.checkpoint_manager.save()
                self.checkpoint_manager_irl.save()

        tf.summary.flush()

    def _get_train_kwargs(self, samples, noise, itr):
        # Do not allow duplication!!!
        # indices = np.random.choice(self._random_range, self._irl.batch_size, replace=False)
        indices = random.sample(self._random_range, self._irl.batch_size - self._noise_size)
        indices = np.array(indices)
        kwargs = {
            "agent_states": samples["obs"],
            "agent_acts": samples["act"],
            "agent_next_states": samples["next_obs"],
            "agent_logps": samples["logp"],
            "expert_states": np.concatenate([self._expert_obs[indices], noise["obs"]]),
            "expert_acts": np.concatenate([self._expert_act[indices], noise["act"]]),
            "expert_next_states": np.concatenate([self._expert_next_obs[indices], noise["next_obs"]]),
            "expert_logps": np.concatenate([self._expert_logp[indices], noise["logp"]]),
            "itr": itr
        }
        return kwargs


    @staticmethod
    def get_argument(parser=None):
        parser = IRLTrainer.get_argument(parser)
        return parser
