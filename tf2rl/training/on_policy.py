"""On-policy rollout, minibatch update, and experiment loop."""

import importlib.metadata
import json
import time

import tensorflow as tf

from tf2rl.algos.base import OnPolicyAgent
from tf2rl.training.checkpoints import policy_checkpoint
from tf2rl.training.collectors import RolloutCollector
from tf2rl.training.evaluation import evaluate_on_policy
from tf2rl.utils.logging import save_run_config
from tf2rl.utils.returns import rollout_targets


class OnPolicyTrainer:
    """Train PPO/VPG using the same collector for Gymnasium and MJWarp."""

    def __init__(self, policy, env, rollout_steps, seed=None):
        """Initialize OnPolicyTrainer and its configuration.

        Args:
            policy (OnPolicyAgent): Tensor-capable agent whose horizon equals num_envs * rollout_steps.
            env (gym.vector.VectorEnv): Tensor-backed environment whose num_envs determines the complete
                rollout batch size.
            rollout_steps (int): Number of sequential steps collected per environment in one rollout.
            seed (int or None): Random seed; None leaves initialization to the underlying random
                generator.

        Returns:
            None: Initializes the instance.

        Raises:
            TypeError: Vector on-policy collection requires an OnPolicyAgent.
            ValueError: Policy horizon must equal num_envs * rollout_steps and be divisible by
                batch_size.
        """
        if not isinstance(policy, OnPolicyAgent):
            raise TypeError("Vector on-policy collection requires an OnPolicyAgent")
        self.policy = policy
        self.env = env
        self.rollout_steps = rollout_steps
        self.count = env.num_envs * rollout_steps
        if self.count != policy.horizon or self.count % policy.batch_size:
            raise ValueError(
                "Policy horizon must equal num_envs * rollout_steps and be divisible by batch_size"
            )
        self.collector = RolloutCollector(policy, env, rollout_steps, seed)

    def collect(self):
        """Collect a fixed-length vector rollout through the stateful rollout collector.

        Returns:
            tuple[tuple[tf.Tensor, ...], dict]: Eight time-major tensors (states, actions, logps,
                values, rewards, terminated, truncated, final_obs) and collection statistics. Each
                tensor starts with (rollout_steps, environments).
        """
        return self.collector.collect()

    def __call__(self, args):
        """Execute the configured on-policy training loop.

        Args:
            args (argparse.Namespace): Parsed experiment options used to configure this operation.

        Returns:
            dict: Completed on-policy run summary returned by run(args).
        """
        return self.run(args)

    def update(self, *rollout):
        """Place the rollout on the learner device before preprocessing and updates.

        Args:
            *rollout (tuple[tf.Tensor, ...]): Eight rollout fields in collector order: states, actions,
                logps, values, rewards, terminated, truncated, final_obs.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Mean [actor_loss, critic_loss, estimated_kl] diagnostics and
                integer minibatch-update count.
        """
        with tf.device(self.policy.device):
            rollout = tuple(tf.identity(value) for value in rollout)
            return self._update(*rollout)

    @tf.function(reduce_retracing=True)
    def _update(self, states, actions, logps, values, rewards, terminated, truncated, final_obs):
        """Build boundary-aware rollout targets, shuffle minibatches, and optimize until epochs
        finish or KL stops the update.

        Args:
            states (tf.Tensor): Current observation tensor shaped (time, environments, *state_shape).
            actions (tf.Tensor): Collected actions shaped (time, environments, *action_shape).
            logps (tf.Tensor): Collection-time action log probabilities shaped (time, environments).
            values (tf.Tensor): Collection-time state values shaped (time, environments).
            rewards (tf.Tensor): Environment rewards shaped (time, environments).
            terminated (tf.Tensor): True termination flags shaped (time, environments).
            truncated (tf.Tensor): Time-limit flags shaped (time, environments).
            final_obs (tf.Tensor): Physical successor observations shaped (time, environments,
                *state_shape), before autoreset.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Mean diagnostics of shape (3,) and scalar minibatch count after
                optimization, including any KL-based early stop.
        """
        policy = self.policy
        state_shape = self.env.single_observation_space.shape
        action_shape = self.env.single_action_space.shape
        states = tf.reshape(states, (self.count, *state_shape))
        actions = tf.reshape(actions, (self.count, *action_shape))
        logps = tf.reshape(logps, (self.count,))
        next_values = tf.reshape(
            policy.get_value_tensor(tf.reshape(final_obs, (self.count, *state_shape))),
            tf.shape(values),
        )
        advantages, returns = rollout_targets(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            policy.discount,
            policy.lam,
            policy.enable_gae,
        )
        advantages = tf.reshape(advantages, (self.count,))
        returns = tf.reshape(returns, (self.count,))
        if policy.normalize_adv:
            mean, variance = tf.nn.moments(advantages, axes=[0])
            advantages = (advantages - mean) / tf.sqrt(variance + 1e-8)
        tf.debugging.assert_all_finite(states, "Nonfinite rollout observations")
        tf.debugging.assert_all_finite(returns, "Nonfinite return targets")
        diagnostics = tf.zeros(3)
        updates = tf.constant(0)
        stop = tf.constant(False)
        for _ in tf.range(policy.n_epoch):
            if not stop:
                order = tf.random.shuffle(tf.range(self.count))
                for offset in tf.range(0, self.count, policy.batch_size):
                    if not stop:
                        index = order[offset : offset + policy.batch_size]
                        result = policy.train_tensor(
                            tf.gather(states, index),
                            tf.gather(actions, index),
                            tf.gather(advantages, index),
                            tf.gather(logps, index),
                            tf.gather(returns, index),
                        )
                        tf.debugging.assert_all_finite(result, "Nonfinite policy losses")
                        diagnostics += result
                        updates += 1
                        stop = policy.should_stop_update(result)
        return diagnostics / tf.cast(updates, tf.float32), updates

    def run(self, args):
        """Collect rollouts, update minibatches, and record each measured iteration.

        Use validated options from on_policy_parser. The run writes metrics.json, TensorBoard
        events, managed checkpoints, and a final checkpoint, and closes its local summary
        writer.

        Args:
            args (argparse.Namespace): Parsed experiment options used to configure this operation.

        Returns:
            dict: Run summary with rollout/update counters, timing, checkpoint path, configuration, and
                optional evaluation statistics.
        """
        policy, trainer = self.policy, self
        count = self.count
        env = self.env
        env_kwargs = dict(args.env_kwargs)
        if args.max_episode_steps is not None:
            env_kwargs["max_episode_steps"] = args.max_episode_steps
        config = save_run_config(args)
        writer = tf.summary.create_file_writer(str(args.output / "events"))
        checkpoint, manager = policy_checkpoint(policy, args.output / "checkpoints", 5)
        self.checkpoint_manager = manager
        try:
            with writer.as_default(step=0), tf.device(policy.device):
                initial = evaluate_on_policy(
                    policy, args.env_name, args.eval_episodes, args.seed + 10000, env_kwargs
                )
                print(json.dumps({"initial_evaluation": initial}), flush=True)
                records = []
                for iteration in range(-args.warmup_iterations, args.iterations):
                    start = time.perf_counter()
                    rollout, stats = trainer.collect()
                    collected = time.perf_counter()
                    losses, updates = trainer.update(*rollout)
                    losses = losses.numpy().tolist()
                    updates = int(updates.numpy())
                    end = time.perf_counter()
                    if hasattr(env, "check_overflow"):
                        env.check_overflow()
                    record = {
                        "iteration": iteration + 1,
                        "rollout_seconds": collected - start,
                        "update_seconds": end - collected,
                        "seconds": end - start,
                        "steps_per_second": count / (end - start),
                        "minibatch_updates": updates,
                        "actor_loss": losses[0],
                        "critic_loss": losses[1],
                        "kl": losses[2],
                        "mean_step_reward": stats["mean_step_reward"],
                        "episodes": stats["episodes"],
                        "episode_return": stats["return_sum"] / stats["episodes"]
                        if stats["episodes"]
                        else None,
                    }
                    if iteration >= 0:
                        records.append(record)
                        with writer.as_default(step=(iteration + 1) * count):
                            for key, value in record.items():
                                if value is not None:
                                    tf.summary.scalar(f"{policy.policy_name}/{key}", value)
                        writer.flush()
                        if (iteration + 1) % args.checkpoint_interval == 0:
                            self.save(iteration + 1)
                    print(json.dumps({"warmup": iteration < 0, **record}), flush=True)
                final = evaluate_on_policy(
                    policy, args.env_name, args.eval_episodes, args.seed + 10000, env_kwargs
                )
                if args.iterations % args.checkpoint_interval:
                    self.save(args.iterations)
                checkpoint.write(str(args.output / "checkpoint"))
                versions = {}
                for package in (
                    "tensorflow",
                    "tensorflow-probability",
                    "tf-keras",
                    "gymnasium",
                    "numpy",
                    "mujoco",
                    "mujoco-warp",
                    "warp-lang",
                ):
                    try:
                        versions[package] = importlib.metadata.version(package)
                    except importlib.metadata.PackageNotFoundError:
                        pass
                summary = {
                    "config": config,
                    "versions": versions,
                    "initial_evaluation": initial,
                    "final_evaluation": final,
                    "training_steps_per_second": count
                    * args.iterations
                    / sum(r["seconds"] for r in records),
                    "rollout_steps_per_second": count
                    * args.iterations
                    / sum(r["rollout_seconds"] for r in records),
                    "measured_transitions": count * args.iterations,
                    "simulation_device": f"cuda:{args.env_gpu}"
                    if args.env_backend == "mjwarp"
                    else "cpu",
                    "actor_devices": sorted(
                        {
                            v.value.device
                            for v in getattr(getattr(policy, "actor", None), "variables", ())
                        }
                    ),
                    "critic_devices": sorted(
                        {
                            v.value.device
                            for v in getattr(getattr(policy, "critic", None), "variables", ())
                        }
                    ),
                    "policy_devices": sorted({v.value.device for v in policy.variables}),
                    "iterations": records,
                }
                (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(
                    json.dumps(
                        {key: value for key, value in summary.items() if key != "iterations"}
                    ),
                    flush=True,
                )

                return summary
        finally:
            writer.close()

    def evaluate(self, args):
        """Evaluate the trained policy in a fresh scalar environment when evaluation episodes
        are requested.

        Args:
            args (argparse.Namespace): Parsed experiment options used to configure this operation.

        Returns:
            dict or None: Per-episode returns and mean_return; None when evaluation is disabled with
                zero episodes.
        """
        kwargs = dict(args.env_kwargs)
        if args.max_episode_steps is not None:
            kwargs["max_episode_steps"] = args.max_episode_steps
        return evaluate_on_policy(
            self.policy, args.env_name, args.eval_episodes, args.seed + 10000, kwargs
        )

    def save(self, iteration):
        """Save policy weights using the rollout iteration as the checkpoint number.

        Args:
            iteration (int): Iteration counter used for checkpoint numbering or reporting.

        Returns:
            str: TensorFlow checkpoint prefix saved with the supplied iteration number.
        """
        return self.checkpoint_manager.save(checkpoint_number=iteration)

    def close(self):
        """Close the vector environment owned by the trainer.

        Returns:
            None.
        """
        self.env.close()
