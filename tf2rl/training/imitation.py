"""Off-policy imitation: learned rewards, generator update, then discriminator update."""

import numpy as np
import tensorflow as tf

from tf2rl.algos.base import OffPolicyAgent
from tf2rl.data.replay import get_replay_buffer
from tf2rl.data.sampling import ExpertSampler
from tf2rl.training.off_policy import OffPolicyTrainer


class ImitationTrainer(OffPolicyTrainer):
    """Reuse the off-policy run loop with an adversarial learning step."""

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
        is_perb_rew=False,
        is_episodic=False,
        is_short_memory=False,
        discriminator_noise=0,
        random_epoch=0,
        expert_logp=None,
    ):
        """Configure training, replay storage, evaluation, and checkpoint handling."""
        if hasattr(env, "num_envs"):
            raise ValueError("ImitationTrainer currently uses a single Gymnasium environment")
        self._irl = irl
        self._export_equation_enabled = getattr(args, "export_equation", False)
        if is_perb_rew or is_short_memory:
            raise NotImplementedError(
                "Perturbed rewards and short-memory modes were never implemented"
            )
        if not isinstance(policy, OffPolicyAgent):
            raise TypeError("The DAC trainer requires an off-policy generator")
        if bool(getattr(policy, "_is_absorbed_state", False)) != is_episodic:
            raise ValueError("Collector absorbing mode must match the policy's target mode")
        if is_episodic and policy.replay_mask_encoding != "dac":
            raise ValueError("Absorbing-state training requires DAC replay masks")
        if irl.requires_log_probs and not callable(getattr(policy, "get_logp", None)):
            raise ValueError("AIRL requires a generator that supplies action log probabilities")
        if random_epoch < 0:
            raise ValueError("random_epoch must be nonnegative")
        super().__init__(policy, env, args, test_env, is_episodic=is_episodic)
        if self._normalize_obs:
            raise ValueError(
                "DAC observation normalization needs an explicit expert/indicator transform"
            )
        if self._use_prioritized_rb or self._use_nstep_rb:
            raise NotImplementedError(
                "DAC uses uniform one-step replay; prioritized/N-step masks need a new objective"
            )
        self._record_logp = irl.requires_log_probs
        self._expert_obs = np.asarray(expert_obs, np.float32)
        self._expert_next_obs = np.asarray(expert_next_obs, np.float32)
        self._expert_act = np.asarray(expert_act, np.float32)
        if (
            self._expert_obs.shape[1:] != env.observation_space.shape
            or self._expert_next_obs.shape != self._expert_obs.shape
        ):
            raise ValueError("Expert and environment observation shapes must match")
        if self._expert_act.shape[1:] != env.action_space.shape:
            raise ValueError("Expert and environment action shapes must match")
        data = {"obs": self._expert_obs, "next_obs": self._expert_next_obs, "act": self._expert_act}
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
            self._checkpoint_irl, self._output_dir + "/irl", max_to_keep=10
        )

    def _get_train_kwargs(self, samples, noise=None, itr=1):
        """Assemble the expert and agent minibatches needed by the IRL update."""
        random_replay = (
            self.random_replay_buffer
            if self._random_epoch and itr % (self._random_epoch * self._episode_max_steps) == 0
            else None
        )
        expert = self._sampler.sample(self.replay_buffer, random_replay)
        kwargs = {
            "agent_states": samples["obs"],
            "agent_acts": samples["act"],
            "agent_next_states": samples["next_obs"],
            "expert_states": expert["obs"],
            "expert_acts": expert["act"],
            "expert_next_states": expert["next_obs"],
            "itr": itr,
        }
        if self._irl.requires_log_probs:
            # Historical off-policy AIRL uses collection-time densities, not
            # current-generator densities. Warmup logps are fresh-action samples;
            # they need not describe the uniform action executed in the world.
            kwargs.update(agent_logps=samples["logp"], expert_logps=expert["logp"])
        return kwargs

    def _prepare_single(self):
        self.replay_buffer = get_replay_buffer(self._policy, self._env, irl=self._irl)
        self.random_replay_buffer = get_replay_buffer(
            self._policy, self._env, irl=self._irl, size=max(1, self._policy.n_warmup)
        )

    def _store_single(self, rows, step):
        super()._store_single(rows, step)
        # Historical epoch-penalty replay excludes the final warmup step and synthetic loops.
        if step < self._policy.n_warmup:
            self.random_replay_buffer.add(**rows[0])

    def update(self, step=None, record_summaries=False):
        samples = self.replay_buffer.sample(self._policy.batch_size)
        apply_atol = (
            getattr(self._irl, "atol_starting_step", None) is not None
            and step > self._irl.atol_starting_step
        )
        kwargs = {"apply_atol": True} if apply_atol else {}
        rewards = self._irl.inference(samples["obs"], samples["act"], samples["next_obs"], **kwargs)
        rewards = tf.reshape(rewards, (-1, 1))
        with tf.summary.record_if(step % self._save_summary_interval == 0):
            masks = (
                samples["done"]
                if self._policy.replay_mask_encoding == "dac"
                else 1.0 - samples["done"]
            )
            tf.summary.scalar("DAC/absorbing_batch_fraction", np.mean(masks < 0))
            tf.summary.scalar("DAC/terminal_batch_fraction", np.mean(masks == 0))
            tf.summary.scalar("GAIL/learned_reward_mean", tf.reduce_mean(rewards))
            # Match the existing update order: generator, then discriminator.
            self._policy.train(
                samples["obs"], samples["act"], samples["next_obs"], rewards, samples["done"]
            )
            for _ in range(self._irl.n_training):
                samples = self.replay_buffer.sample(self._irl.batch_size)
                self._irl.train(**self._get_train_kwargs(samples, itr=step))

    def save(self, step=None):
        super().save(step)
        self.checkpoint_manager_irl.save()
        self._export_equation(step)

    def _finish_single(self):
        if self._max_steps % self._save_model_interval:
            self.checkpoint_manager.save()
            self.checkpoint_manager_irl.save()
        self._export_equation(self._max_steps)

    def _export_equation(self, step):
        """Save the current inference equation when equation export is enabled."""
        if not self._export_equation_enabled:
            return
        from tf2rl.networks.equation_export import (
            expression_statistics,
            extract_equation,
            save_equation,
        )

        model = self._irl.equation_model
        state_only = getattr(self._irl, "_state_only", False)
        state_names = [f"obs_{i}" for i in range(self._env.observation_space.shape[0] - 1)]
        if self._env.spec.id == "Pendulum-v1":
            state_names = ["cos_theta", "sin_theta", "theta_dot"]
        names = state_names + ["absorbing_indicator"]
        if not state_only:
            names += [f"action_{i}" for i in range(self._env.action_space.shape[0])]
        graph = extract_equation(
            model,
            reward_kind="airl" if self._irl.requires_log_probs else "gail",
            input_names=names,
            metadata={
                "environment": self._env.spec.id,
                "environment_step": step,
                "state_only": state_only,
            },
        )
        save_equation(graph, self._output_dir + f"/equations/step_{step:08d}")
        complexity = expression_statistics(graph)
        for key in ("operations", "nonlinear_operations", "affine_coefficients"):
            tf.summary.scalar("EQN/expression_" + key, complexity[key])
        tf.summary.scalar("EQN/expression_inputs", len(complexity["inputs"]))

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser."""
        parser = OffPolicyTrainer.get_argument(parser)
        parser.add_argument("--expert-path-dir", default=None)
        parser.add_argument("--discriminator-noise", type=float, default=0.15)
        parser.add_argument("--absorbed-state-disable", action="store_false")
        parser.add_argument(
            "--n_expert_trajectories", "--n-expert-trajectories", type=int, default=20
        )
        parser.add_argument("--random-epoch-n", type=int, default=0)
        return parser
