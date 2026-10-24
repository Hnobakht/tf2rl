"""Construct imitation experiment components without owning execution or resources.

These functions retain the historical constructor keywords, model creation order,
and AIRL/GAIL trainer selection. Algorithm implementations live in ``tf2rl.algos``.
"""

import tensorflow as tf

from tf2rl.algos.airl import AIRL
from tf2rl.algos.airl_eql import AIRL_EQL
from tf2rl.algos.gail import GAIL
from tf2rl.algos.gail_eql import GAIL_EQL
from tf2rl.algos.sac import SAC
from tf2rl.algos.sac_irl import SAC_IRL
from tf2rl.experiments.airl_trainer import AIRLTrainer
from tf2rl.experiments.demonstrations import restore_latest_n_traj
from tf2rl.experiments.irl_trainer import IRLTrainer
from tf2rl.experiments.runtime import ConfigurationError


def _csv(value, cast):
    """Parse comma-separated values using the requested conversion function."""
    return [cast(item.strip()) for item in value.split(",")]


def make_policy(args, env, *, imitation, absorbing):
    """Build the original SAC/SAC_IRL generator with the launcher defaults."""
    units = _csv(args.units, int)
    policy_class = SAC_IRL if imitation else SAC
    absorbing_kwargs = {"is_absorbed_state": absorbing} if imitation else {}
    return policy_class(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.shape[0],
        max_action=float(env.action_space.high[0]),
        actor_units=units,
        critic_units=units,
        gpu=args.gpu,
        n_warmup=args.n_warmup,
        batch_size=args.batch_size,
        memory_capacity=args.memory_capacity,
        auto_alpha=args.auto_alpha_disable,
        alpha=args.alpha,
        lr=args.lr,
        lr_alpha=args.lr_alpha,
        is_xla=args.xla,
        **absorbing_kwargs,
    )


def make_discriminator(args, env):
    """Select the dense or equation AIRL/GAIL discriminator."""
    common = {
        "state_shape": env.observation_space.shape,
        "action_dim": env.action_space.shape[0],
        "gpu": args.gpu,
        "batch_size": args.batch_size_ail,
        "lr": args.lr_ail,
        "n_training": args.n_training,
        "is_debug": args.is_debug,
        # Preserve historical effective defaults while honoring explicit delays
        # for dense models too (their CLI value used to be silently discarded).
        "update_delay": (args.update_delay if args.update_delay is not None
                         else (3 if args.algorithm.startswith("eqn-") else 1)),
    }
    eql = {
        "v": _csv(args.v, int),
        "num_layers": len(args.v.split(",")),
        "drop_out": _csv(args.dropout, float),
        "exclude": [item.split("-") if item else [] for item in args.exclude.split(",")]
        if args.exclude
        else None,
        "lmbda": args.eqn_l1,
        "training_stages": _csv(args.eqn_stages, int) if args.eqn_stages else None,
        "prune_threshold": args.eqn_prune_threshold,
    }
    if args.algorithm == "eqn-airl":
        return AIRL_EQL(
            **common, **eql, is_state_only=args.state_only, units=_csv(args.units_ail, int)
        )
    elif args.algorithm == "airl":
        return AIRL(**common, state_only=args.state_only, units=_csv(args.units_ail, int))
    elif args.algorithm == "eqn-gail":
        return GAIL_EQL(**common, **eql, grad_penalty_coeff=args.gp_coef)
    else:
        return GAIL(**common, units=_csv(args.units_ail, int), grad_penalty_coeff=args.gp_coef)


def make_imitation_trainer(args, policy, irl, env, test_env, absorbing):
    """Load demonstrations and select the trainer required by the discriminator."""
    data = restore_latest_n_traj(
        args.expert_path_dir,
        n_path=args.n_expert_trajectories,
        is_logps=irl.requires_log_probs,
        env=env,
        absorbing=absorbing,
    )
    kwargs = {
        "test_env": test_env,
        "is_episodic": absorbing,
        "discriminator_noise": args.discriminator_noise,
        "random_epoch": args.random_epoch_n,
    }
    if irl.requires_log_probs:
        return AIRLTrainer(
            policy,
            env,
            args,
            irl,
            data["obses"],
            data["next_obses"],
            data["acts"],
            data["logps"],
            **kwargs,
        )
    else:
        return IRLTrainer(
            policy, env, args, irl, data["obses"], data["next_obses"], data["acts"], **kwargs
        )


def restore_discriminator(trainer, irl, directory):
    """Restore the discriminator after checking the equation sparsity mode."""
    checkpoint = tf.train.latest_checkpoint(directory)
    if checkpoint is None:
        raise ConfigurationError("No discriminator checkpoint in --irl-model-dir")
    equation_model = getattr(irl, "equation_model", None)
    saved_sparse = any(
        "/sparsity_updates/" in name for name, _ in tf.train.list_variables(checkpoint)
    )
    if equation_model is not None and equation_model.sparse != saved_sparse:
        raise ConfigurationError(
            "Checkpoint sparsity mode differs; restore with matching --eqn-l1/--eqn-stages options"
        )
    status = trainer._checkpoint_irl.restore(checkpoint)
    status.assert_existing_objects_matched()
