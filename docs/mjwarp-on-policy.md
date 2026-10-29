# Shared on-policy MJWarp training

MJWarp is selected by the environment factory. The common on-policy launcher and
`OnPolicyTrainer` support **both on-policy algorithms currently implemented
in this project: PPO and VPG**. The collector does not construct PPO, require its
network layout, or impose its KL stopping rule on other algorithms.

## Training

Use the [GPU container setup and NJU dependency mirror](mjwarp.md#installation).
For example, from the host checkout:

```bash
docker exec -w /main/git_repos/tf2rl tf-gpu-tensorflow-1 \
  /tmp/tf2rl-mjwarp/bin/python -m tf2rl.experiments.train --runner on_policy \
  --algorithm vpg --env-name HalfCheetah-v5 \
  --env-backend mjwarp --env-gpu 0 --gpu -1 \
  --num-envs 256 --rollout-steps 32 --batch-size 1024 \
  --iterations 100 --output results/mjwarp/vpg
```

`--algorithm ppo` selects PPO using the same backend and collector. `--gpu -1`
keeps policy inference and optimization on CPU; `--gpu 0` uses the GPU learner.
`--env-gpu` independently selects the simulation GPU. Use `--env-backend
gymnasium` for native CPU environments. Existing task-adapter coverage and limits
are described in [the backend guide](mjwarp.md).

The installed entry point is `onpolicy-train`. `examples/train_ppo.py` selects
PPO; `python -m tf2rl.experiments.train --runner on_policy --algorithm vpg`
selects VPG. `ppo-train` supplies PPO defaults through the same assembly.

| Algorithm | Actor objective | Default epochs per rollout | Update stopping |
| --- | --- | ---: | --- |
| PPO | Existing clipped likelihood-ratio objective | 10 | Existing target KL |
| VPG | Existing vanilla log-probability policy gradient | 1 | Complete configured epochs |

`--epochs` can override the algorithm default. `--target-kl` is specific to PPO;
other algorithm keywords can be supplied with `--policy-kwargs '{...}'`. VPG does
not acquire PPO clipping or early stopping by using this collector.

A rollout has `num_envs * rollout_steps` physical transitions. Its size must be
divisible by `--batch-size`. Timeouts bootstrap the last physical observation;
true terminations do not. Return/advantage recursion stops at both episode
boundaries, preserving each world's trajectories across same-step autoresets.

The launcher supports vector Box observations and Discrete or uniformly bounded,
symmetric vector Box actions. MJWarp's current tasks use continuous actions.
Native CPU discrete tasks also use the shared trainer. Both separate and shared
actor/value networks are supported through the agent interface.

## Checkpoints, playback and TensorBoard

Runs save `config.json`, `metrics.json`, managed `checkpoints/`, a final
`checkpoint` prefix and `events/`. Configuration records the selected algorithm;
TensorBoard tags use its policy name, such as `VPG/actor_loss` or `PPO/actor_loss`.

To render a checkpoint into a video from the host checkout, using the existing
render-enabled environment:

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=-1 .venv-modern/bin/python -m tf2rl.training.evaluation \
  --run-dir results/mjwarp/vpg --episodes 1 --output results/mjwarp/vpg.mp4
```

Omit `--output` for an interactive viewer with an available display. Saved configurations
without an algorithm field are interpreted as historical PPO runs.
Rendering in a container additionally requires EGL libraries; the existing
TensorFlow container was validated for simulation and host-side playback.

```bash
/tmp/tf2rl-mjwarp/bin/tensorboard --logdir results/mjwarp --host 0.0.0.0 --port 6006
```

## Add another on-policy agent

An `OnPolicyAgent` uses these tensor methods:

- `get_action_and_val_tensor(states, test=False)` returns actions, their log
  probabilities and state values.
- `get_value_tensor(states)` returns bootstrap values. The default delegates to
  deterministic inference; agents can evaluate their value network directly.
- `initialize_tensor_training(states)` builds any lazy networks and optimizer
  slots before the compiled update loop.
- `train_tensor(states, actions, advantages, logp_olds, returns)` performs one
  minibatch update and returns `[actor_loss, critic_loss, estimated_kl]`.
- `should_stop_update(diagnostics)` defines algorithm-specific stopping. The
  default is false; PPO overrides it with its target KL.

The collector depends on this interface instead of actor/critic attributes or
PPO types. For direct Python use, construct the agent and pass it with any
supported vector environment to `OnPolicyTrainer`.

To expose an additional agent through the common launcher, register its factory
before invoking `main`:

```python
from tf2rl.experiments.build import register_on_policy_algorithm
from tf2rl.experiments.train import main

# MyAgent implements the OnPolicyAgent tensor interface and accepts the
# standard model/training constructor keywords, or is wrapped by a factory.
register_on_policy_algorithm("my-agent", MyAgent, default_epochs=1)
main(runner="on_policy", algorithm="my-agent")
```

No environment-backend changes are needed when adding an algorithm. This does
not supply implementations of algorithms absent from the repository.

## Validation

Run the short workflows described in [validation](validation.md). The current
checkout has no unit-test suite. GPU/MJWarp execution requires the optional GPU
runtime; historical GPU results and their limits remain recorded there.

The backend comparison script also accepts `--algorithm ppo` or `--algorithm vpg`:

```bash
/tmp/tf2rl-mjwarp/bin/python -m scripts.benchmark_env_backends \
  --algorithm vpg --env-name HalfCheetah-v5 --num-envs 64 256 \
  --iterations 5 --output results/mjwarp/vpg_benchmark
```
