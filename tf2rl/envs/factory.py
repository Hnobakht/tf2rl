"""Construct independent training/evaluation environments for research launchers."""

import gymnasium as gym
from tf2rl.envs.vector import make_vector_env
from tf2rl.envs.wrappers.absorbing import AbsorbingWrapper


def make_environment(args, backend, count, *, absorbing=False):
    """Build a tensor-backed vector environment from experiment options and the chosen backend.

    Args:
        args (argparse.Namespace): Parsed experiment options used to configure this operation.
        backend (str): Simulation backend: gymnasium or mjwarp.
        count (int): Number of environments to construct.
        absorbing (bool): Enable DAC absorbing-state transitions and their corresponding
            interpretation.

    Returns:
        gym.vector.VectorEnv: Batched training/evaluation environment, optionally tensor-backed and
            DAC-augmented, on the selected backend.
    """
    kwargs = dict(args.env_kwargs)
    if args.max_episode_steps is not None:
        kwargs["max_episode_steps"] = args.max_episode_steps
    if backend == "mjwarp":
        kwargs.update(nconmax=args.nconmax, njmax=args.njmax)
    device = (
        f"cuda:{args.env_gpu}"
        if backend == "mjwarp"
        else (f"/GPU:{args.gpu}" if args.gpu >= 0 else "/CPU:0")
    )
    return make_vector_env(
        args.env_name,
        count,
        backend=backend,
        tensors=True,
        device=device,
        absorbing=absorbing,
        absorbing_indicator=getattr(args, "absorbing_indicator", 0.1),
        **kwargs,
    )


def make_single_environment(args, *, evaluation=False, imitation=False):
    """Build a scalar research environment and apply the recipe observation/rendering wrappers.

    Args:
        args (argparse.Namespace): Parsed experiment options used to configure this operation.
        evaluation (bool): Create an evaluation environment with the requested rendering settings.
        imitation (bool): Select the imitation recipe and its observation/replay conventions.

    Returns:
        gym.Env: Scalar environment with the selected registration, observation, reward, and
            rendering wrappers.
    """
    if evaluation:
        mode = (
            "rgb_array" if args.save_test_movie else ("human" if args.show_test_progress else None)
        )
    else:
        mode = "human" if args.show_progress else None
    env = gym.make(args.env_name, render_mode=mode)
    if imitation:
        try:
            env = AbsorbingWrapper(env, args.absorbing_indicator)
        except BaseException:
            env.close()
            raise
    return env
