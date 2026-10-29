"""Off-policy training loops. Put breakpoints in run(), collect(), or update()."""

import copy
import json
import logging
import time
from contextlib import ExitStack
from functools import wraps
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from tf2rl.data.replay import get_replay_buffer
from tf2rl.training.checkpoints import policy_checkpoint
from tf2rl.training.collectors import SingleStepCollector, StepCollector, numpy
from tf2rl.training.evaluation import evaluate_policy, evaluate_vector_policy
from tf2rl.utils.logging import initialize_logger, prepare_output_dir, save_run_config
from tf2rl.utils.normalization import EmpiricalNormalizer


def with_summary_writer(method):
    """Keep each trainer's writer and step scoped to its own execution.

    Args:
        method (callable): Trainer method to execute inside its TensorBoard writer context.

    Returns:
        callable: Wrapped method preserving its metadata and return value while selecting the
            trainer summary writer.
    """

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        """Call the trainer operation inside its TensorBoard summary-writer context.

        Args:
            *args (tuple): Positional arguments passed unchanged to the decorated trainer method.
            **kwargs (dict): Keyword arguments passed unchanged to the decorated trainer method.

        Returns:
            object: The decorated trainer method return value, unchanged.
        """
        with self.writer.as_default(step=0):
            return method(self, *args, **kwargs)

    return wrapped


class OffPolicyTrainer:
    """Shared off-policy execution with scalar and vector collection recipes."""

    def __init__(
        self,
        policy,
        env,
        args=None,
        test_env=None,
        is_episodic=False,
        *,
        updates_per_transition=1.0,
        absorbing=False,
        seed=0,
        replay_buffer=None,
    ):
        """Configure training, replay storage, evaluation, and checkpoint handling.

        Args:
            policy (Policy): Agent supplying action selection and learning operations.
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.
            args (argparse.Namespace or dict or None): Scalar trainer settings, or vector-run options.
                None permits direct vector collect/update calls without a configured run.
            test_env (gym.Env or None): Independent evaluation environment; must differ from the
                training instance.
            is_episodic (bool): Historical scalar episodic-mode field; absorbing semantics come from the
                agent configuration.
            updates_per_transition (float): Gradient minibatch updates budgeted per physical transition
                after random warmup.
            absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
                interpretation.
            seed (int or None): Random seed; None leaves initialization to the underlying random
                generator.
            replay_buffer (ReplayBuffer or None): Optional externally allocated replay buffer; None
                creates one for the agent.

        Returns:
            None: Initializes the instance.

        Raises:
            ValueError: Training and evaluation must use separate environment instances. Absorbing-state
                training requires an absorbing wrapper and DAC masks. Absorbing-state training requires
                unnormalized one-step transitions. Evaluation requires --model-dir.
        """
        self._vector = hasattr(env, "num_envs")
        self.args = args
        self._record_logp = False
        if self._vector:
            self.policy, self.env = policy, env
            self._policy, self._env = policy, env
            self._test_env = test_env
            self.writer = None
            self.collector = StepCollector(
                policy,
                env,
                updates_per_transition=updates_per_transition,
                absorbing=absorbing,
                seed=seed,
            )
            spaces = SimpleNamespace(
                observation_space=env.single_observation_space, action_space=env.single_action_space
            )
            self.replay_buffer = (
                get_replay_buffer(policy, spaces) if replay_buffer is None else replay_buffer
            )
            self.total_steps = self.total_updates = self.replay_rows = 0
            self.update_credit = 0.0
            self.updates_per_transition = updates_per_transition
            return
        if args is None:
            args = {}
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
                raise ValueError(
                    "Absorbing-state training requires an absorbing wrapper and DAC masks"
                )
            if self._normalize_obs or self._use_nstep_rb:
                raise ValueError(
                    "Absorbing-state training requires unnormalized one-step transitions"
                )
        self._is_dm_env = False
        self._episode_max_steps = self._horizon(env)
        if self._normalize_obs:
            self._obs_normalizer = EmpiricalNormalizer(shape=env.observation_space.shape)
        self._output_dir = prepare_output_dir(
            args, self._logdir, suffix=f"{policy.policy_name}_{args.dir_suffix}"
        )
        self.logger = initialize_logger(
            logging_level=logging.getLevelName(args.logging_level), output_dir=self._output_dir
        )
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
        """Return the configured episode limit or the environment time limit.

        Args:
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.

        Returns:
            int: Effective scalar episode horizon, capped by the environment time limit when available.
        """
        configured = self._episode_max_steps
        limit = env.spec.max_episode_steps if env.spec is not None else None
        return min(configured, limit) if limit is not None else configured

    @staticmethod
    def _action(result):
        """Extract the action from a policy result, accepting either an action or a tuple.

        Args:
            result (array_like or tuple): Action-selection result, optionally paired with action log
                probabilities.

        Returns:
            array_like: The action component, removing a tuple of auxiliary action diagnostics when
                present.
        """
        return result if not isinstance(result, tuple) else result[0]

    def _policy_observation(self, obs, update=False):
        """Optionally update scalar observation statistics and return the policy input
        observation.

        Args:
            obs (array_like): Current observation, or batch of observations with a leading batch axis.
            update (bool): Update running statistics from the supplied observations before normalizing.

        Returns:
            np.ndarray: Observation in policy coordinates, optionally updating the observation
                normalizer first.
        """
        if not self._normalize_obs:
            return obs
        if update:
            self._obs_normalizer.experience(obs[None])
        return self._obs_normalizer(obs, update=False)

    def _set_from_args(self, args):
        """Copy trainer settings from the parsed command-line arguments.

        Args:
            args (argparse.Namespace): Parsed experiment options used to configure this operation.

        Returns:
            None.
        """
        self._max_steps = args.max_steps
        self._episode_max_steps = (
            args.episode_max_steps if args.episode_max_steps is not None else args.max_steps
        )
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
        """Add shared scalar trainer arguments to an existing or new parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        from tf2rl.experiments.config import add_trainer_arguments

        return add_trainer_arguments(parser)

    def _set_check_point(self, model_dir):
        """Create the policy checkpoint manager and restore weights when model_dir is supplied.

        Args:
            model_dir (str or Path or None): Directory containing policy weights to restore; None starts
                with current initialization.

        Returns:
            None.
        """
        self._checkpoint, self.checkpoint_manager = policy_checkpoint(
            self._policy, self._output_dir
        )
        self._latest_path_ckpt = None
        if model_dir is not None:
            self._latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            if self._latest_path_ckpt is None:
                raise ValueError(f"No checkpoint in {model_dir}")
            self._checkpoint.restore(self._latest_path_ckpt)

    def __call__(self):
        """Execute the configured training or evaluation run.

        Returns:
            object: Result of run(): scalar replay buffer, scalar evaluation pair, or vector-run summary
                depending on configuration.
        """
        return self.run()

    def run(self, args=None, *, evaluation_env=None, restored=None):
        """Collect, update, evaluate, and save. Imitation overrides update().

        Scalar settings come from construction. Vector settings come from off_policy_parser and
        may be supplied here. The run advances environment state, updates network parameters,
        and writes configured checkpoints and metrics.

        Args:
            args (argparse.Namespace or None): Validated vector off-policy options; None reuses
                constructor options. Scalar runs use the constructor configuration.
            evaluation_env (gym.vector.VectorEnv or None): Independent vector environment used for
                evaluation and expert export.
            restored (str or None): Checkpoint prefix previously restored by experiment assembly,
                recorded in the run summary.

        Returns:
            cpprb.ReplayBuffer or tuple[float, float] or dict: Scalar training returns its replay
                buffer; scalar evaluation mode returns mean return/length; vector training returns a
                run-summary dictionary.
        """
        if self._vector:
            self.args = args or self.args
            self._test_env = evaluation_env
            return self._run_vector(self.args, evaluation_env, restored)
        with self.writer.as_default(step=0):
            if self._evaluate:
                return self.evaluate_policy_continuously()
            self._prepare_single()
            self.collector = SingleStepCollector(
                self._policy,
                self._env,
                self._episode_max_steps,
                self._policy_observation,
                absorbing=self._absorbing,
                mask_encoding=self._mask_encoding,
                record_logp=self._record_logp,
            )
            for step in range(1, self._max_steps + 1):
                self.collect(step)
                if step >= self._policy.n_warmup and step % self._policy.update_interval == 0:
                    self.update(step)
                self._periodic(step)
            self._finish_single()
            self.writer.flush()
            return self.replay_buffer

    def _prepare_single(self):
        """Allocate scalar replay with the configured priority and N-step storage options.

        Returns:
            None.
        """
        self.replay_buffer = get_replay_buffer(
            self._policy,
            self._env,
            use_prioritized_rb=self._use_prioritized_rb,
            use_nstep_rb=self._use_nstep_rb,
            n_step=self._n_step,
        )

    def _store_single(self, rows, step):
        """Insert the collected physical and synthetic transition rows into scalar replay.

        Args:
            rows (sequence[dict]): Physical replay rows and any synthetic absorbing self-loops to
                insert.
            step (int or None): Physical environment-step counter used for warmup, summaries, or
                checkpoint numbering. This implementation does not read this argument.

        Returns:
            None.
        """
        for row in rows:
            self.replay_buffer.add(**row)

    def collect(self, step=None):
        """Collect one scalar or vector step, store replay rows, and advance collection
        counters.

        Args:
            step (int or None): Required scalar environment-step count for warmup and scheduling; unused
                in vector mode.

        Returns:
            dict: Transition/episode statistics from one collection step; the collected rows are
                inserted into replay and learning budgets are advanced.
        """
        if self._vector:
            rows, stats = self.collector.collect()
            self.replay_buffer.add(**rows)
            self.total_steps = self.collector.total_steps
            self.replay_rows += len(rows["obs"])
            self.update_credit += stats["learning_transitions"] * self.updates_per_transition
            return stats
        rows, stats = self.collector.collect(step)
        self._store_single(rows, step)
        tf.summary.experimental.set_step(step)
        if self._show_progress and step % self._show_progress_interval == 0:
            self._env.render()
        if stats["boundary"]:
            self.replay_buffer.on_episode_end()
            tf.summary.scalar("Common/training_return", stats["episode_return"])
            self.collector.reset_episode()
        return stats

    def collect_step(self):
        """Compatibility spelling for vector collection.

        Returns:
            dict: Statistics returned by collect() after one environment batch has been stored.
        """
        return self.collect()

    def update(self, step=None, record_summaries=False):
        """Run eligible replay minibatch updates and maintain priorities, summaries, and the
        vector update budget.

        Args:
            step (int or None): Required scalar environment-step count for summary scheduling; unused in
                vector mode.
            record_summaries (bool): Enable summaries for the selected update; scalar imitation uses its
                step-based schedule.

        Returns:
            dict or None: Vector mode returns updates and mean_update_diagnostic; scalar mode updates
                the agent and summaries in place and returns None.
        """
        if self._vector:
            return self._update_vector(record_summaries)
        samples = self.replay_buffer.sample(self._policy.batch_size)
        with tf.summary.record_if(step % self._save_summary_interval == 0):
            self._policy.train(
                samples["obs"],
                samples["act"],
                samples["next_obs"],
                samples["rew"],
                samples["done"],
                samples.get("weights") if self._use_prioritized_rb else None,
            )
        if self._use_prioritized_rb:
            errors = self._policy.compute_td_error(
                samples["obs"], samples["act"], samples["next_obs"], samples["rew"], samples["done"]
            )
            self.replay_buffer.update_priorities(samples["indexes"], np.abs(errors) + 1e-6)

    def _finish_single(self):
        """Provide a no-op scalar-run finalization hook for subclasses.

        Returns:
            None.
        """
        pass

    def _periodic(self, step):
        """Run scheduled scalar evaluation and checkpoint saving at the supplied environment
        step.

        Args:
            step (int or None): Physical environment-step counter used for warmup, summaries, or
                checkpoint numbering.

        Returns:
            None.
        """
        if step % self._test_interval == 0:
            reward, length = self.evaluate_policy(step)
            tf.summary.scalar("Common/average_test_return", reward)
            tf.summary.scalar("Common/average_test_episode_length", length)
            self.logger.info("Step %d: evaluation return %.3f, length %.1f", step, reward, length)
            self.writer.flush()
        if step % self._save_model_interval == 0:
            self.save(step)

    def save(self, step=None):
        """Save policy state with the configured TensorFlow checkpoint manager, if present.

        Args:
            step (int or None): Physical environment-step counter used for warmup, summaries, or
                checkpoint numbering.

        Returns:
            str or None: Saved checkpoint prefix returned by the TensorFlow manager. A vector run must
                configure its manager before this method is called.
        """
        if self._vector:
            return self.checkpoint_manager.save(checkpoint_number=step)
        return self.checkpoint_manager.save()

    @with_summary_writer
    def evaluate_policy(self, total_steps):
        """Evaluate the scalar policy and record configured summaries, demonstrations, or
        videos.

        Args:
            total_steps (int): Physical environment-step count used to label evaluation summaries and
                trajectory files.

        Returns:
            tuple[float, float]: Mean undiscounted episode return and mean episode length.
        """
        return evaluate_policy(self, total_steps)

    def evaluate(self, total_steps=0):
        """Evaluate the configured scalar or vector policy on its independent test environment.

        Args:
            total_steps (int): Physical environment-step count used to label evaluation summaries and
                trajectory files.

        Returns:
            dict or tuple[float, float]: Vector evaluation statistics, or the scalar mean-return/mean-
                length pair.

        Raises:
            ValueError: Vector evaluation requires a separate evaluation_env.
        """
        if self._vector:
            if self._test_env is None:
                raise ValueError("Vector evaluation requires a separate evaluation_env")
            return evaluate_vector_policy(
                self.policy, self._test_env, self.args.eval_episodes, seed=self.args.seed + 10000
            )
        return self.evaluate_policy(total_steps)

    def evaluate_policy_continuously(self):
        """Perform one scalar evaluation pass using the compatibility evaluation entry point.

        Returns:
            tuple[float, float]: Mean return and length from one evaluation pass; this compatibility
                method does not loop.
        """
        return self.evaluate_policy(0)

    def close(self):
        """Close training/evaluation environments and any configured summary writer.

        Returns:
            None.
        """
        with ExitStack() as resources:
            if self.writer is not None:
                resources.callback(self.writer.close)
            resources.callback(self._env.close)
            if self._test_env is not None and self._test_env is not self._env:
                resources.callback(self._test_env.close)

    def _update_vector(self, record_summaries=False):
        """Spend accumulated update credit; return counts and a finite diagnostic.

        Args:
            record_summaries (bool): Enable summaries for the selected update; scalar imitation uses its
                step-based schedule.

        Returns:
            dict: Number of updates performed and mean_update_diagnostic, which is None when no
                diagnostic was produced.

        Raises:
            FloatingPointError: Nonfinite off-policy learner diagnostic.
        """
        if self.replay_buffer.get_stored_size() < self.policy.batch_size:
            return {"updates": 0, "mean_update_diagnostic": None}
        count = int(self.update_credit + 1e-9)
        diagnostics = []
        with tf.device(self.policy.device):
            for index in range(count):
                batch = self.replay_buffer.sample(self.policy.batch_size)
                with tf.summary.record_if(record_summaries and index == count - 1):
                    result = self.policy.train(
                        batch["obs"], batch["act"], batch["next_obs"], batch["rew"], batch["done"]
                    )
                values = numpy(result)
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite off-policy learner diagnostic")
                diagnostics.append(float(values.mean()))
        self.update_credit -= count
        self.total_updates += count
        return {
            "updates": count,
            "mean_update_diagnostic": float(np.mean(diagnostics)) if diagnostics else None,
        }

    def _run_vector(self, args, evaluation_env, restored):
        """Run the vector collect/update loop, save checkpoints, and optionally evaluate the
        trained policy.

        Args:
            args (argparse.Namespace): Parsed experiment options used to configure this operation.
            evaluation_env (gym.vector.VectorEnv or None): Independent vector environment used for
                evaluation and expert export.
            restored (str or None): Checkpoint prefix previously restored by experiment assembly,
                recorded in the run summary.

        Returns:
            dict: Run summary containing environment/learner counters, timing, configuration,
                checkpoint, and optional evaluation results.

        Raises:
            ValueError: run() requires experiment options; collect()/update() can be used directly.
        """
        if args is None:
            raise ValueError(
                "run() requires experiment options; collect()/update() can be used directly"
            )
        policy, trainer = self.policy, self
        env = self.env
        config = save_run_config(args)
        writer = self.writer = tf.summary.create_file_writer(str(args.output / "events"))
        checkpoint, manager = policy_checkpoint(policy, args.output / "checkpoints", 5)
        self.checkpoint_manager = manager
        try:
            records, evaluations = [], []
            next_save, next_eval = args.checkpoint_interval, args.eval_interval
            with writer.as_default(step=0), tf.device(policy.device):
                while not args.evaluate and trainer.total_steps < args.max_steps:
                    start = time.perf_counter()
                    collected = []
                    updates = 0
                    collection_seconds = update_seconds = 0.0
                    for index in range(args.collection_steps):
                        if trainer.total_steps >= args.max_steps:
                            break
                        before = time.perf_counter()
                        collected.append(trainer.collect())
                        collection_seconds += time.perf_counter() - before
                        before = time.perf_counter()
                        report = (
                            index == args.collection_steps - 1
                            or trainer.total_steps >= args.max_steps
                        )
                        with writer.as_default(step=trainer.total_steps):
                            updates += trainer.update(record_summaries=report)["updates"]
                        update_seconds += time.perf_counter() - before
                    if hasattr(env, "check_overflow"):
                        env.check_overflow()
                    elapsed = time.perf_counter() - start
                    transitions = sum(row["transitions"] for row in collected)
                    episodes = sum(row["episodes"] for row in collected)
                    record = {
                        "total_steps": trainer.total_steps,
                        "transitions": transitions,
                        "replay_size": trainer.replay_buffer.get_stored_size(),
                        "total_updates": trainer.total_updates,
                        "updates": updates,
                        "collection_seconds": collection_seconds,
                        "update_seconds": update_seconds,
                        "steps_per_second": transitions / elapsed,
                        "seconds": elapsed,
                        "episodes": episodes,
                        "mean_step_reward": sum(row["reward_sum"] for row in collected)
                        / transitions,
                        "episode_return": sum(row["return_sum"] for row in collected) / episodes
                        if episodes
                        else None,
                        "terminations": sum(row["terminations"] for row in collected),
                        "truncations": sum(row["truncations"] for row in collected),
                    }
                    records.append(record)
                    with writer.as_default(step=trainer.total_steps):
                        for key, value in record.items():
                            if value is not None:
                                tf.summary.scalar(f"OffPolicy/{key}", value)
                    if trainer.total_steps >= next_save:
                        self.save(trainer.total_steps)
                        next_save = (
                            trainer.total_steps // args.checkpoint_interval + 1
                        ) * args.checkpoint_interval
                    if args.eval_episodes and trainer.total_steps >= next_eval:
                        evaluation = evaluate_vector_policy(
                            policy, evaluation_env, args.eval_episodes, seed=args.seed + 10000
                        )
                        evaluations.append(dict(total_steps=trainer.total_steps, **evaluation))
                        with writer.as_default(step=trainer.total_steps):
                            tf.summary.scalar(
                                "OffPolicy/evaluation_return", evaluation["mean_return"]
                            )
                        next_eval = (
                            trainer.total_steps // args.eval_interval + 1
                        ) * args.eval_interval
                    writer.flush()
                    print(json.dumps(record), flush=True)
                steps = trainer.total_steps if not args.evaluate else 0
                if not args.evaluate:
                    self.save(steps)
                    checkpoint.write(str(args.output / "checkpoint"))
                final_evaluation = (
                    evaluate_vector_policy(
                        policy, evaluation_env, args.eval_episodes, seed=args.seed + 10000
                    )
                    if args.eval_episodes
                    else None
                )
                experts = None
                if args.save_expert:
                    experts = evaluate_vector_policy(
                        policy,
                        evaluation_env,
                        args.expert_episodes,
                        seed=args.seed + 20000,
                        output_dir=args.expert_dir or args.output / "expert",
                        total_steps=steps,
                        deterministic=not args.stochastic_expert,
                        metadata={
                            "algorithm": args.algorithm,
                            "env_name": args.env_name,
                            "backend": args.eval_env_backend,
                            "absorbing": args.absorbing,
                            "absorbing_indicator": args.absorbing_indicator,
                        },
                    )
                models = {"actor": policy.actor}
                for name in ("qf1", "qf2", "qf1_target", "qf2_target", "critic", "critic_target"):
                    if hasattr(policy, name):
                        models[name] = getattr(policy, name)
                summary = {
                    "config": config,
                    "restored_checkpoint": restored,
                    "total_steps": steps,
                    "total_updates": trainer.total_updates if not args.evaluate else 0,
                    "replay_rows": trainer.replay_rows if not args.evaluate else 0,
                    "mask_encoding": policy.replay_mask_encoding,
                    "model_devices": {
                        name: sorted({v.value.device for v in model.variables})
                        for name, model in models.items()
                    },
                    "iterations": records,
                    "evaluations": evaluations,
                    "final_evaluation": final_evaluation,
                    "expert_export": experts,
                }
                (args.output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            key: value
                            for key, value in summary.items()
                            if key not in ("iterations", "evaluations")
                        }
                    ),
                    flush=True,
                )

            return summary
        finally:
            writer.close()
