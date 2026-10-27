"""Gymnasium off-policy training, evaluation, and expert trajectory export."""
import argparse
import copy
import logging
from contextlib import ExitStack
from functools import wraps
from pathlib import Path

import numpy as np
import tensorflow as tf

from tf2rl.envs.normalizer import EmpiricalNormalizer
from tf2rl.experiments.utils import frames_to_gif, save_path
from tf2rl.misc.dac_ops import dac_transitions
from tf2rl.misc.get_replay_buffer import get_replay_buffer
from tf2rl.misc.initialize_logger import initialize_logger
from tf2rl.misc.prepare_output_dir import prepare_output_dir


def with_summary_writer(method):
    """Keep each trainer's writer and step scoped to its own execution."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        """Call the trainer operation inside its TensorBoard summary-writer context."""
        with self.writer.as_default(step=0):
            return method(self, *args, **kwargs)
    return wrapped


class Trainer:
    """Collect environment transitions, train a policy, and manage evaluation and checkpoints."""
    def __init__(self, policy, env, args, test_env=None, is_episodic=False):
        """Configure training, replay storage, evaluation, and checkpoint handling."""
        if isinstance(args, dict):
            supplied = args
            args = policy.get_argument(self.get_argument()).parse_args([])
            for key, value in supplied.items():
                if not hasattr(args, key):
                    raise ValueError(f"{key} is invalid parameter")
                setattr(args, key, value)
        self._set_from_args(args)
        self._policy, self._env = policy, env
        if test_env is env:
            raise ValueError("Training and evaluation must use separate environment instances")
        self._test_env = copy.deepcopy(env) if test_env is None else test_env
        self._is_episodic = is_episodic
        self._absorbing = bool(getattr(policy, "_is_absorbed_state", False))
        self._mask_encoding = getattr(policy, "replay_mask_encoding", "done")
        if self._absorbing:
            if self._mask_encoding != "dac" or not env.has_wrapper_attr("get_absorbing_state"):
                raise ValueError("Absorbing-state training requires an absorbing wrapper and DAC masks")
            if self._normalize_obs or self._use_nstep_rb:
                raise ValueError("Absorbing-state training requires unnormalized one-step transitions")
        self._is_dm_env = False
        self._episode_max_steps = self._horizon(env)
        if self._normalize_obs:
            self._obs_normalizer = EmpiricalNormalizer(shape=env.observation_space.shape)
        self._output_dir = prepare_output_dir(args, self._logdir,
                                              suffix=f"{policy.policy_name}_{args.dir_suffix}")
        self.logger = initialize_logger(logging_level=logging.getLevelName(args.logging_level),
                                         output_dir=self._output_dir)
        self.writer = tf.summary.create_file_writer(self._output_dir)
        try:
            self._set_check_point(args.model_dir)
            if args.evaluate and args.model_dir is None:
                raise ValueError("Evaluation requires --model-dir")
        except BaseException:
            # Construction failed before a caller could register trainer.close().
            self.writer.close()
            raise

    def _horizon(self, env):
        """Return the configured episode limit or the environment time limit."""
        configured = self._episode_max_steps
        limit = env.spec.max_episode_steps if env.spec is not None else None
        return min(configured, limit) if limit is not None else configured

    def _set_check_point(self, model_dir):
        """Create the TensorFlow checkpoint manager and restore a checkpoint when requested."""
        self._checkpoint = tf.train.Checkpoint(policy=self._policy)
        self.checkpoint_manager = tf.train.CheckpointManager(self._checkpoint, self._output_dir,
                                                              max_to_keep=10)
        self._latest_path_ckpt = None
        if model_dir is not None:
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            if self._latest_path_ckpt is None:
                raise ValueError(f"No checkpoint in {model_dir}")
            self._checkpoint.restore(self._latest_path_ckpt)

    @staticmethod
    def _action(result):
        """Extract the action from a policy result, accepting either an action or a tuple."""
        return result if not isinstance(result, tuple) else result[0]

    def _policy_observation(self, obs, update=False):
        """Return the observation expected by the policy, optionally stripping the DAC indicator."""
        if not self._normalize_obs:
            return obs
        if update:
            self._obs_normalizer.experience(obs[None])
        return self._obs_normalizer(obs, update=False)

    @with_summary_writer
    def __call__(self):
        """Run the training loop, including data collection, policy updates, and evaluation."""
        if self._evaluate:
            return self.evaluate_policy_continuously()
        self.replay_buffer = get_replay_buffer(self._policy, self._env,
                                               use_prioritized_rb=self._use_prioritized_rb,
                                               use_nstep_rb=self._use_nstep_rb, n_step=self._n_step)
        obs, _ = self._env.reset()
        episode_steps, episode_return = 0, 0.
        for total_steps in range(1, self._max_steps + 1):
            policy_obs = self._policy_observation(np.asarray(obs, np.float32), update=True)
            action = (self._env.action_space.sample() if total_steps - 1 < self._policy.n_warmup
                      else self._action(self._policy.get_action(policy_obs)))
            next_obs, reward, terminated, truncated, _ = self._env.step(action)
            episode_steps += 1
            truncated = truncated or episode_steps >= self._episode_max_steps
            final_obs = self._policy_observation(np.asarray(next_obs, np.float32))
            if self._mask_encoding == "dac":
                rows = dac_transitions(self._env, policy_obs, action, final_obs, reward,
                                       terminated, truncated, absorbing=self._absorbing)
                for row in rows:
                    self.replay_buffer.add(**row)
            else:
                self.replay_buffer.add(obs=policy_obs, act=action, next_obs=final_obs,
                                       rew=reward, done=float(terminated))
            episode_return += reward
            obs = next_obs
            tf.summary.experimental.set_step(total_steps)
            if self._show_progress and total_steps % self._show_progress_interval == 0:
                self._env.render()
            if terminated or truncated:
                self.replay_buffer.on_episode_end()
                tf.summary.scalar("Common/training_return", episode_return)
                obs, _ = self._env.reset()
                episode_steps, episode_return = 0, 0.
            if total_steps >= self._policy.n_warmup and total_steps % self._policy.update_interval == 0:
                samples = self.replay_buffer.sample(self._policy.batch_size)
                with tf.summary.record_if(total_steps % self._save_summary_interval == 0):
                    self._policy.train(samples["obs"], samples["act"], samples["next_obs"],
                                        samples["rew"], samples["done"],
                                        samples.get("weights") if self._use_prioritized_rb else None)
                if self._use_prioritized_rb:
                    errors = self._policy.compute_td_error(samples["obs"], samples["act"],
                                                           samples["next_obs"], samples["rew"], samples["done"])
                    self.replay_buffer.update_priorities(samples["indexes"], np.abs(errors) + 1e-6)
            self._periodic(total_steps)
        self.writer.flush()
        return self.replay_buffer

    def _periodic(self, total_steps):
        """Run evaluation and save checkpoints at their configured training intervals."""
        if total_steps % self._test_interval == 0:
            reward, length = self.evaluate_policy(total_steps)
            tf.summary.scalar("Common/average_test_return", reward)
            tf.summary.scalar("Common/average_test_episode_length", length)
            self.logger.info("Step %d: evaluation return %.3f, length %.1f",
                             total_steps, reward, length)
            self.writer.flush()
        if total_steps % self._save_model_interval == 0:
            self.checkpoint_manager.save()

    def evaluate_policy_continuously(self):
        # Retain the historical one-shot implementation despite the old name.
        """Watch for new checkpoints and evaluate each restored policy."""
        return self.evaluate_policy(0)

    @with_summary_writer
    def evaluate_policy(self, total_steps):
        """Evaluate the policy in the test environment and return its performance."""
        returns, lengths = [], []
        horizon = self._horizon(self._test_env)
        for episode in range(self._test_episodes):
            obs, _ = self._test_env.reset()
            rows, frames = [], []
            episode_return = 0.
            for step in range(1, horizon + 1):
                result = self._policy.get_action(self._policy_observation(np.asarray(obs, np.float32)), test=True)
                action = self._action(result)
                next_obs, reward, terminated, truncated, _ = self._test_env.step(action)
                truncated = truncated or step >= horizon
                row = dict(obs=obs, act=action, next_obs=next_obs, rew=reward,
                           done=terminated or truncated, terminated=terminated, truncated=truncated)
                if isinstance(result, tuple):
                    row["logp"] = result[1]
                rows.append(row)
                if self._save_test_movie:
                    frames.append(self._test_env.render())
                elif self._show_test_progress:
                    self._test_env.render()
                episode_return += reward
                obs = next_obs
                if terminated or truncated:
                    break
            prefix = f"step_{total_steps:08d}_epi_{episode:02d}_return_{episode_return:010.4f}"
            if self._save_test_path:
                samples = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
                samples["format_version"] = 2
                save_path(samples, str(Path(self._output_dir) / (prefix + ".pkl")))
            if self._save_test_movie:
                frames_to_gif(frames, prefix, self._output_dir)
            returns.append(episode_return)
            lengths.append(len(rows))
        return float(np.mean(returns)), float(np.mean(lengths))

    def close(self):
        # Attempt every cleanup even if a viewer or summary writer fails to close.
        """Close the summary writer and both environments, attempting every cleanup callback."""
        with ExitStack() as resources:
            resources.callback(self.writer.close)
            resources.callback(self._env.close)
            resources.callback(self._test_env.close)

    def _set_from_args(self, args):
        """Copy trainer settings from the parsed command-line arguments."""
        self._max_steps = args.max_steps
        self._episode_max_steps = (args.episode_max_steps
                                   if args.episode_max_steps is not None
                                   else args.max_steps)
        self._n_experiments = args.n_experiments
        self._show_progress = args.show_progress
        self._show_progress_interval = args.show_progress_interval
        self._save_model_interval = args.save_model_interval
        self._save_summary_interval = args.save_summary_interval
        self._normalize_obs = args.normalize_obs
        self._logdir = args.logdir
        self._model_dir = args.model_dir
        # replay buffer
        self._use_prioritized_rb = args.use_prioritized_rb
        self._use_nstep_rb = args.use_nstep_rb
        self._n_step = args.n_step
        # test settings
        self._evaluate = args.evaluate
        self._test_interval = args.test_interval
        self._show_test_progress = args.show_test_progress
        self._test_episodes = args.test_episodes
        self._save_test_path = args.save_test_path
        self._save_test_movie = args.save_test_movie
        self._show_test_images = args.show_test_images

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
        if parser is None:
            parser = argparse.ArgumentParser(conflict_handler='resolve')
        # experiment settings
        parser.add_argument('--max-steps', type=int, default=int(1e6),
                            help='Maximum number steps to interact with env. <default: %(default)s>')
        parser.add_argument('--episode-max-steps', type=int, default=int(1e3),
                            help='Maximum steps in an episode <default: %(default)s>')
        parser.add_argument('--n-experiments', type=int, default=1,
                            help='Number of experiments. <default: %(default)s>')
        parser.add_argument('--show-progress', action='store_true',
                            help='Call `render` in training process. <default: %(default)s>')
        parser.add_argument('--show-progress-interval', type=int, default=int(1e4),
                            help='Interval to save model. <default: %(default)s>')
        parser.add_argument('--save-model-interval', type=int, default=int(5e4),
                            help='Interval to save model. <default: %(default)s>')
        parser.add_argument('--save-summary-interval', type=int, default=int(1e3),
                            help='Interval to save summary. <default: %(default)s>')
        parser.add_argument('--model-dir', type=str, default=None,
                            help='Directory to restore model. <default: %(default)s>')
        parser.add_argument('--reward-dir', type=str, default=None,
                            help='Directory to restore rewad function. <default: %(default)s>')
        parser.add_argument('--dir-suffix', type=str, default='',
                            help='Suffix for directory that contains results. <default: %(default)s>')
        parser.add_argument('--normalize-obs', action='store_true',
                            help='Normalize observation. <default: %(default)s>')
        parser.add_argument('--logdir', type=str, default='results',
                            help='Output directory. <default: %(default)s>')
        # test settings
        parser.add_argument('--evaluate', action='store_true',
                            help='Evaluate trained model. <default: %(default)s>')
        parser.add_argument('--test-interval', type=int, default=int(1e4),
                            help='Interval to evaluate trained model. <default: %(default)s>')
        parser.add_argument('--show-test-progress', action='store_true',
                            help='Call `render` in evaluation process. <default: %(default)s>')
        parser.add_argument('--test-episodes', type=int, default=5,
                            help='Number of episodes to evaluate at once. <default: %(default)s>')
        parser.add_argument('--save-test-path', action='store_true',
                            help='Save trajectories of evaluation. <default: %(default)s>')
        parser.add_argument('--show-test-images', action='store_true',
                            help='Show input images to neural networks when an episode finishes. <default: %(default)s>')
        parser.add_argument('--save-test-movie', action='store_true',
                            help='Save rendering results. <default: %(default)s>')
        # replay buffer
        parser.add_argument('--use-prioritized-rb', action='store_true',
                            help='Flag to use prioritized experience replay. <default: %(default)s>')
        parser.add_argument('--use-nstep-rb', action='store_true',
                            help='Flag to use nstep experience replay. <default: %(default)s>')
        parser.add_argument('--n-step', type=int, default=4,
                            help='Number of steps to look over. <default: %(default)s>')
        # others
        parser.add_argument('--logging-level', choices=['DEBUG', 'INFO', 'WARNING'],
                            default='INFO', help='Logging level. <default: %(default)s>')
        parser.add_argument('--is-debug', action='store_true',
                            help='Plot the weights and/or gradient of the networks (only available for EQL methods). <default: %(default)s>')
        return parser
