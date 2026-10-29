"""Construct imitation experiment components without owning execution or resources.

These functions retain the historical constructor keywords, model creation order,
and AIRL/GAIL trainer selection. Algorithm implementations live in ``tf2rl.algos``.
"""

import gymnasium as gym
import numpy as np
import tensorflow as tf

from tf2rl.algos.base import OnPolicyAgent
from tf2rl.data.demonstrations import restore_latest_n_traj
from tf2rl.experiments.config import ConfigurationError
from tf2rl.training.imitation import ImitationTrainer


def _csv(value, cast):
    """Parse comma-separated values using the requested conversion function."""
    return [cast(item.strip()) for item in value.split(",")]


def make_imitation_agent(args, env, *, imitation, absorbing):
    """Build an off-policy generator; SAC retains the original launcher defaults."""
    from tf2rl.algos.rl.sac import SAC
    from tf2rl.algos.rl.td3 import TD3
    from tf2rl.algos.rl.ddpg import DDPG

    units = _csv(args.units, int)
    generator = getattr(args, "generator", "sac")
    if generator != "sac":
        if not imitation:
            raise ConfigurationError(
                "--generator selects an imitation policy; use offpolicy-train for RL"
            )
        if args.algorithm in ("airl", "eqn-airl"):
            raise ConfigurationError(
                "AIRL requires action log probabilities; select --generator sac"
            )
        if args.xla:
            raise ConfigurationError("--xla currently supports SAC")
        policy_class = {"td3": TD3, "ddpg": DDPG}[generator]
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
            lr_actor=args.lr,
            lr_critic=args.lr,
            is_absorbed_state=absorbing,
        )
    return SAC(
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
        is_absorbed_state=absorbing,
        critic_layer_norm=not imitation,
        replay_mask_encoding="dac" if imitation else None,
    )


def make_discriminator(args, env):
    """Select the dense or equation AIRL/GAIL discriminator."""
    from tf2rl.algos.imitation.airl import AIRL
    from tf2rl.algos.imitation.airl_eqn import AIRL_EQL
    from tf2rl.algos.imitation.gail import GAIL
    from tf2rl.algos.imitation.gail_eqn import GAIL_EQL

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
        "update_delay": (
            args.update_delay
            if args.update_delay is not None
            else (3 if args.algorithm.startswith("eqn-") else 1)
        ),
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
    return ImitationTrainer(
        policy,
        env,
        args,
        irl,
        data["obses"],
        data["next_obses"],
        data["acts"],
        expert_logp=data.get("logps"),
        **kwargs,
    )


def make_off_policy_agent(
    algorithm,
    observation_space,
    action_space,
    *,
    absorbing=False,
    gpu=-1,
    units=(64, 64),
    learning_rate=3e-4,
    batch_size=256,
    memory_capacity=1000000,
    n_warmup=10000,
    auto_alpha=True,
    alpha=0.2,
    sigma=0.1,
    xla=False,
):
    """Construct an existing algorithm without embedding it in the environment backend."""
    if (
        not isinstance(observation_space, gym.spaces.Box)
        or len(observation_space.shape) != 1
        or not isinstance(action_space, gym.spaces.Box)
        or len(action_space.shape) != 1
    ):
        raise ValueError("This entry point requires vector Box observations and controls")
    if not np.allclose(action_space.high, -action_space.low) or not np.allclose(
        action_space.high, action_space.high[0]
    ):
        raise ValueError("Existing actors require uniform symmetric action bounds")
    if xla and algorithm != "sac":
        raise ValueError("XLA compilation is currently exposed for SAC")
    common = {
        "state_shape": observation_space.shape,
        "action_dim": action_space.shape[0],
        "max_action": float(action_space.high[0]),
        "actor_units": tuple(units),
        "critic_units": tuple(units),
        "gpu": gpu,
        "batch_size": batch_size,
        "memory_capacity": memory_capacity,
        "n_warmup": n_warmup,
    }
    with tf.device(f"/GPU:{gpu}" if gpu >= 0 else "/CPU:0"):
        if algorithm == "sac":
            from tf2rl.algos.rl.sac import SAC

            return SAC(
                **common,
                lr=learning_rate,
                lr_alpha=learning_rate,
                auto_alpha=auto_alpha,
                alpha=alpha,
                is_xla=xla,
                is_absorbed_state=absorbing,
                # Preserve the architecture of existing launcher checkpoints.
                critic_layer_norm=not absorbing,
            )
        if algorithm == "td3":
            from tf2rl.algos.rl.td3 import TD3

            cls = TD3
        elif algorithm == "ddpg":
            from tf2rl.algos.rl.ddpg import DDPG

            cls = DDPG
        else:
            raise ValueError(f"Unknown off-policy algorithm {algorithm!r}")
        return cls(
            **common,
            lr_actor=learning_rate,
            lr_critic=learning_rate,
            sigma=sigma,
            is_absorbed_state=absorbing,
        )


_ALGORITHMS = {}


def register_on_policy_algorithm(name, factory, *, default_epochs=1):
    """Register an agent factory without changing the collector or environment backend.

    Factories receive the standard VPG-style model/training keywords and optional
    policy-specific keywords. They must return an OnPolicyAgent implementing its
    tensor interface. Registration before main() also makes the name a CLI choice.
    """
    if not name or name in _ALGORITHMS or not callable(factory) or default_epochs < 1:
        raise ValueError("Expected a unique algorithm name, callable factory and positive epochs")
    _ALGORITHMS[name] = {"factory": factory, "default_epochs": default_epochs}


def _ppo(**kwargs):
    from tf2rl.algos.rl.ppo import PPO

    return PPO(**kwargs)


def _vpg(**kwargs):
    from tf2rl.algos.rl.vpg import VPG

    return VPG(**kwargs)


register_on_policy_algorithm("ppo", _ppo, default_epochs=10)
register_on_policy_algorithm("vpg", _vpg, default_epochs=1)


def make_on_policy_agent(
    algorithm,
    observation_space,
    action_space,
    *,
    horizon=2048,
    batch_size=256,
    epochs=None,
    units=(64, 64),
    learning_rate=3e-4,
    gpu=-1,
    policy_kwargs=None,
):
    """Create a registered on-policy agent using single-world environment spaces."""
    if algorithm not in _ALGORITHMS:
        raise ValueError(f"Unknown on-policy algorithm {algorithm!r}")
    if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
        raise ValueError("The on-policy launcher requires vector Box observations")
    discrete = isinstance(action_space, gym.spaces.Discrete)
    if not discrete and (
        not isinstance(action_space, gym.spaces.Box)
        or len(action_space.shape) != 1
        or not np.allclose(action_space.high, -action_space.low)
        or not np.allclose(action_space.high, action_space.high[0])
    ):
        raise ValueError("Actors require Discrete or uniform symmetric vector Box actions")
    spec = _ALGORITHMS[algorithm]
    kwargs = {
        "state_shape": observation_space.shape,
        "action_dim": int(action_space.n) if discrete else action_space.shape[0],
        "is_discrete": discrete,
        "max_action": 1.0 if discrete else float(action_space.high[0]),
        "horizon": horizon,
        "batch_size": batch_size,
        "n_epoch": spec["default_epochs"] if epochs is None else epochs,
        "actor_units": units,
        "critic_units": units,
        "hidden_activation_actor": "tanh",
        "hidden_activation_critic": "tanh",
        "lr_actor": learning_rate,
        "lr_critic": learning_rate,
        "entropy_coef": 0.0,
        "gpu": gpu,
    }
    kwargs.update(policy_kwargs or {})
    with tf.device(f"/GPU:{gpu}" if gpu >= 0 else "/CPU:0"):
        policy = spec["factory"](**kwargs)
    if not isinstance(policy, OnPolicyAgent):
        raise TypeError("Registered factories must return an OnPolicyAgent")
    return policy


def preload_imitation_algorithms():
    """Preserve the scalar recipes' import-before-seeding initialization order.

    Some historical models have initializer objects as constructor defaults.
    Vector recipes intentionally import models after seeding instead.
    """
    from tf2rl.algos.imitation import airl, gail_eqn
    from tf2rl.algos.rl import sac
    from tf2rl.algos.imitation import airl_eqn
    from tf2rl.algos.rl import ddpg, td3

    return airl, gail_eqn, sac, airl_eqn, ddpg, td3
