import time
import random

import numpy as np
import tensorflow as tf

from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.experiments.trainer import Trainer

# tf.random.set_seed(1234)
# random.seed(1234)
# np.random.seed(1234)

class IRLTrainer(Trainer):
    def __init__(
            self,
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
            discriminator_noise=0):
        self._irl = irl
        args.dir_suffix = self._irl.policy_name + args.dir_suffix
        super().__init__(policy, env, args, test_env, is_episodic=is_episodic)
        # TODO: Add assertion to check dimention of expert demos and current policy, env is the same
        self._expert_obs = expert_obs
        self._expert_next_obs = expert_next_obs
        self._expert_act = expert_act
        # Minus one to get next obs
        self._random_range = range(expert_obs.shape[0])
        self._is_perb_rew = is_perb_rew
        self.disc_noise = discriminator_noise
        self._irl_chechpoint()
        assert hasattr(self._env, "_max_episode_steps"), "The imitation algorithems need _max_episode_steps"

    def _irl_chechpoint(self):
        # self._checkpoint_irl = tf.train.Checkpoint(policy=self._irl)
        self._checkpoint_irl = tf.train.Checkpoint(irl_weights=self._irl.weights)
        self.checkpoint_manager_irl = tf.train.CheckpointManager(
            self._checkpoint_irl, directory=self._output_dir + "/irl", max_to_keep=10)

    def __call__(self):
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        episode_steps = 0
        episode_return = 0
        episode_return_D = 0
        episode_start_time = time.perf_counter()
        n_episode = 0

        replay_buffer = get_replay_buffer(
            self._policy, self._env, self._use_prioritized_rb,
            self._use_nstep_rb, self._n_step)

        obs = np.float32(self._env.reset())

        while total_steps < self._max_steps:
            if total_steps < self._policy.n_warmup:
                action = self._env.action_space.sample()
            else:
                if "sac" in self._policy.name:
                    action, logp = self._policy.get_action(obs)
                else:
                    action = self._policy.get_action(obs)

            # assert np.isfinite(action).all(), "The action for steping into environment is not valid. step number is {}".format(total_steps)
            next_obs, reward, done, _ = self._env.step(action)
            next_obs, reward = np.float32(next_obs), np.float32(reward)
            done = False if self._is_episodic else done
            if self._show_progress and total_steps % self._show_progress_interval:
                self._env.render()
            episode_steps += 1
            episode_return += reward

            if total_steps > self._policy.n_warmup + 1: # To not to ove trace the @tf
                if "EQL" in self._irl.policy_name and self._irl.atol_starting_step:
                    episode_return_D += self._irl.inference(obs, action, next_obs,
                                                            total_steps >  self._irl.atol_starting_step)[0][0]
                else:
                    episode_return_D += self._irl.inference(obs, action, next_obs)[0][0]
            total_steps += 1
            tf.summary.experimental.set_step(total_steps)

            done_flag = done
            if (hasattr(self._env, "_max_episode_steps") and
                episode_steps == self._env._max_episode_steps):
                done_flag = False

            replay_buffer.add(obs=obs, act=action,
                              next_obs=next_obs, rew=reward, done=done_flag)
            obs = next_obs

            if done or episode_steps == self._episode_max_steps:
                replay_buffer.on_episode_end()
                obs = np.float32(self._env.reset())

                n_episode += 1
                fps = episode_steps / (time.perf_counter() - episode_start_time)
                self.logger.info("Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f} FPS: {4:5.2f}".format(
                                 n_episode, int(total_steps), episode_steps, episode_return, fps))
                tf.summary.scalar(name="Common/training_return", data=episode_return)
                tf.summary.scalar(name="Common/training_adverseral_return", data = episode_return_D/episode_steps)

                episode_steps = 0
                episode_return = 0
                episode_return_D = 0
                episode_start_time = time.perf_counter()

            if total_steps < self._policy.n_warmup:
                continue

            if total_steps % self._policy.update_interval == 0:
                samples = replay_buffer.sample(self._policy.batch_size)
                # Train policy
                if "EQL" in self._irl.policy_name and self._irl.atol_starting_step:
                    rew = self._irl.inference(samples["obs"], samples["act"], samples["next_obs"],
                                              total_steps > self._irl.atol_starting_step)
                else:
                    rew = self._irl.inference(samples["obs"], samples["act"], samples["next_obs"])
                # print(np.mean(rew))
                # rew = tf.add(rew, tf.random.normal(rew.shape, stddev=(rew.numpy()).std())) if self._is_perb_rew else rew
                with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                    # # TODO: debug this one line
                    # tf.summary.histogram(name=self._policy.policy_name + "/D_reward", data=rew)
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

                    # Train IRL
                    for _ in range(self._irl.n_training):
                        samples = replay_buffer.sample(self._irl.batch_size)
                        noise_size = int(self._irl.batch_size * self.disc_noise)
                        noise = replay_buffer.sample(noise_size);
                        # Do not allow duplication!!!
                        # my code: changing np.rsndom.choice to random.sample for speed
                        # indices = np.random.choice(self._random_range, self._irl.batch_size, replace=False)
                        # indices = random.sample(self._random_range, self._irl.batch_size)
                        #TODO: permute the mixed expert data
                        indices = random.sample(self._random_range, self._irl.batch_size - noise_size)
                        indices = np.array(indices)
                        assert indices.shape[0] + noise_size == self._irl.batch_size, \
                            "The size of expert batch does not match the IRL batch size."
                        if "EQL" in self._irl.policy_name:
                            self._irl.train(
                                agent_states=samples["obs"],
                                agent_acts=samples["act"],
                                agent_next_states=samples["next_obs"],
                                # expert_states=self._expert_obs[indices],
                                # expert_acts=self._expert_act[indices],
                                # expert_next_states=self._expert_next_obs[indices],
                                expert_states=np.concatenate([self._expert_obs[indices], noise["obs"]]),
                                expert_acts=np.concatenate([self._expert_act[indices], noise["act"]]),
                                expert_next_states=np.concatenate([self._expert_next_obs[indices], noise["next_obs"]]),
                                itr = total_steps)
                        else:
                            self._irl.train(
                                agent_states=samples["obs"],
                                agent_acts=samples["act"],
                                agent_next_states=samples["next_obs"],
                                # expert_states=self._expert_obs[indices],
                                # expert_acts=self._expert_act[indices],
                                # expert_next_states=self._expert_next_obs[indices]
                                expert_states=np.concatenate([self._expert_obs[indices], noise["obs"]]),
                                expert_acts=np.concatenate([self._expert_act[indices], noise["act"]]),
                                expert_next_states=np.concatenate([self._expert_next_obs[indices], noise["next_obs"]])
                                )


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

            if total_steps % self._save_model_interval == 0:
                self.checkpoint_manager.save()
                # TODO: Sometimes returns error in case of EQL activaton functions
                self.checkpoint_manager_irl.save()

        tf.summary.flush()

    @staticmethod
    def get_argument(parser=None):
        parser = Trainer.get_argument(parser)
        parser.add_argument('--expert-path-dir', default=None,
                            help='Path to directory that contains expert trajectories.\n' +
                            'In case d4rl data sets: d4rl:<name of the environment>. <default: %(default)s>')
        parser.add_argument('--discriminator-noise',  type=float, default=0.15,
                            help='It used to Stabilize Generative Adversarial Training. <default: %(default)s>'),
        parser.add_argument('--absorbed-state-disable', action='store_false',help=" <default: %(default)s>")
        parser.add_argument('--n_expert_trajectories', type=int, default=20,help=" <default: %(default)s>")

        return parser
