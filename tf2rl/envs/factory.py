"""Construct independent training/evaluation environments for research launchers."""

import gymnasium as gym
from tf2rl.envs.vector import make_vector_env
from tf2rl.envs.wrappers.absorbing import AbsorbingWrapper


def make_environment(args, backend, count, *, absorbing=False):
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
