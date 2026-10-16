"""Unified off-policy DAC collector and discriminator trainer."""
import numpy as np
import tensorflow as tf

from tf2rl.algos.sac_irl import SAC_IRL
from tf2rl.experiments.trainer import Trainer, with_summary_writer
from tf2rl.experiments.expert_sampling import ExpertSampler
from tf2rl.misc.dac_ops import dac_transitions
from tf2rl.misc.get_replay_buffer import get_replay_buffer


class IRLTrainer(Trainer):
    def __init__(self, policy, env, args, irl, expert_obs, expert_next_obs, expert_act,
                 test_env=None, is_perb_rew=False, is_episodic=False, is_short_memory=False,
                 discriminator_noise=0, random_epoch=0, expert_logp=None):
        self._irl = irl
        if is_perb_rew or is_short_memory:
            raise NotImplementedError("Perturbed rewards and short-memory modes were never implemented")
        if not isinstance(policy, SAC_IRL):
            raise ValueError("The DAC trainer requires the research SAC_IRL generator")
        if policy._is_absorbed_state != is_episodic:
            raise ValueError("Collector absorbing mode must match SAC_IRL target mode")
        if random_epoch < 0:
            raise ValueError("random_epoch must be nonnegative")
        super().__init__(policy, env, args, test_env, is_episodic=is_episodic)
        if self._normalize_obs:
            raise ValueError("DAC observation normalization needs an explicit expert/indicator transform")
        if self._use_prioritized_rb or self._use_nstep_rb:
            raise NotImplementedError("DAC uses uniform one-step replay; prioritized/N-step masks need a new objective")
        self._expert_obs = np.asarray(expert_obs, np.float32)
        self._expert_next_obs = np.asarray(expert_next_obs, np.float32)
        self._expert_act = np.asarray(expert_act, np.float32)
        if self._expert_obs.shape[1:] != env.observation_space.shape or self._expert_next_obs.shape != self._expert_obs.shape:
            raise ValueError("Expert and environment observation shapes must match")
        if self._expert_act.shape[1:] != env.action_space.shape:
            raise ValueError("Expert and environment action shapes must match")
        data = dict(obs=self._expert_obs, next_obs=self._expert_next_obs, act=self._expert_act)
        if irl.requires_log_probs:
            if expert_logp is None:
                raise ValueError("AIRL requires stored expert log probabilities")
            self._expert_logp = np.asarray(expert_logp, np.float32).reshape(-1, 1)
            data["logp"] = self._expert_logp
        self.disc_noise = discriminator_noise
        self._random_epoch = random_epoch
        if random_epoch and policy.n_warmup < 1:
            raise ValueError("Epoch penalty requires a nonempty warmup replay")
        self._sampler = ExpertSampler(data, irl.batch_size, discriminator_noise)
        self._noise_size = self._sampler.noise_size
        self._checkpoint_irl = tf.train.Checkpoint(irl=irl)
        self.checkpoint_manager_irl = tf.train.CheckpointManager(
            self._checkpoint_irl, self._output_dir + "/irl", max_to_keep=10)

    def _get_train_kwargs(self, samples, noise=None, itr=1):
        random_replay = (self.random_replay_buffer if self._random_epoch
                         and itr % (self._random_epoch * self._episode_max_steps) == 0 else None)
        expert = self._sampler.sample(self.replay_buffer, random_replay)
        kwargs = dict(agent_states=samples["obs"], agent_acts=samples["act"],
                      agent_next_states=samples["next_obs"], expert_states=expert["obs"],
                      expert_acts=expert["act"], expert_next_states=expert["next_obs"], itr=itr)
        if self._irl.requires_log_probs:
            kwargs.update(agent_logps=samples["logp"], expert_logps=expert["logp"])
        return kwargs

    @with_summary_writer
    def __call__(self):
        if self._evaluate:
            return self.evaluate_policy_continuously()
        self.replay_buffer = get_replay_buffer(self._policy, self._env, irl=self._irl)
        self.random_replay_buffer = get_replay_buffer(self._policy, self._env, irl=self._irl,
                                                     size=max(1, self._policy.n_warmup))
        obs, _ = self._env.reset()
        episode_steps, episode_return = 0, 0.
        for step in range(1, self._max_steps + 1):
            warmup = step - 1 < self._policy.n_warmup
            obs = np.asarray(obs, np.float32)
            if warmup:
                action = self._env.action_space.sample()
                logp = self._policy.get_logp(obs) if self._irl.requires_log_probs else None
            else:
                action, policy_logp = self._policy.get_action(obs)
                logp = policy_logp if self._irl.requires_log_probs else None
            next_obs, reward, terminated, truncated, _ = self._env.step(action)
            episode_steps += 1
            truncated = truncated or episode_steps >= self._episode_max_steps
            rows = dac_transitions(self._env, obs, action, np.asarray(next_obs, np.float32),
                                    reward, terminated, truncated, absorbing=self._is_episodic, logp=logp)
            for row in rows:
                self.replay_buffer.add(**row)
            # Preserve epoch penalty: physical warmup transitions only, excluding
            # the last warmup step, as in the original IRLTrainer.
            if step < self._policy.n_warmup:
                self.random_replay_buffer.add(**rows[0])
            episode_return += reward
            obs = next_obs
            tf.summary.experimental.set_step(step)
            if self._show_progress and step % self._show_progress_interval == 0:
                self._env.render()
            if terminated or truncated:
                self.replay_buffer.on_episode_end()
                tf.summary.scalar("Common/training_return", episode_return)
                obs, _ = self._env.reset()
                episode_steps, episode_return = 0, 0.
            if step >= self._policy.n_warmup and step % self._policy.update_interval == 0:
                samples = self.replay_buffer.sample(self._policy.batch_size)
                apply_atol = (getattr(self._irl, "atol_starting_step", None) is not None
                              and step > self._irl.atol_starting_step)
                kwargs = {"apply_atol": True} if apply_atol else {}
                rewards = self._irl.inference(samples["obs"], samples["act"], samples["next_obs"], **kwargs)
                with tf.summary.record_if(step % self._save_summary_interval == 0):
                    tf.summary.scalar("DAC/absorbing_batch_fraction",
                                      np.mean(samples["done"] < 0))
                    tf.summary.scalar("DAC/terminal_batch_fraction",
                                      np.mean(samples["done"] == 0))
                    tf.summary.scalar("GAIL/learned_reward_mean", tf.reduce_mean(rewards))
                    # Match the existing update order: generator, then discriminator.
                    self._policy.train(samples["obs"], samples["act"], samples["next_obs"],
                                        rewards, samples["done"])
                    for _ in range(self._irl.n_training):
                        samples = self.replay_buffer.sample(self._irl.batch_size)
                        self._irl.train(**self._get_train_kwargs(samples, itr=step))
            self._periodic(step)
            if step % self._save_model_interval == 0:
                self.checkpoint_manager_irl.save()
        self.writer.flush()
        return self.replay_buffer

    @staticmethod
    def get_argument(parser=None):
        parser = Trainer.get_argument(parser)
        parser.add_argument('--expert-path-dir', default=None)
        parser.add_argument('--discriminator-noise', type=float, default=.15)
        parser.add_argument('--absorbed-state-disable', action='store_false')
        parser.add_argument('--n_expert_trajectories', '--n-expert-trajectories', type=int, default=20)
        parser.add_argument('--random-epoch-n', type=int, default=0)
        return parser
