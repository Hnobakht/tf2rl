# MuJoCo Warp backend

MuJoCo Warp is an optional batched CUDA environment backend. It is independent
of PPO and the imitation-learning algorithms. Select it through
`make_vector_env(..., backend="mjwarp")`; keep the same Gymnasium five-value
step interface. The default factory backend remains `gymnasium`.

The implementation has three reusable parts:

- `MJWarpSimulator` accepts any MJWarp-compatible `mujoco.MjModel` and batches
  physics with configurable world count, frame skip and contact capacities.
- `MJWarpVectorEnv` adds task adapters for observations, rewards, termination,
  random resets and time limits. Built-in adapters currently cover
  `HalfCheetah-v5` and `InvertedPendulum-v5`.
- `OnPolicyTrainer` collects tensor rollouts from either backend through
  `OnPolicyAgent`'s tensor interface. PPO and VPG retain their own losses and
  stopping rules. Observation and action sizes come from the environment spaces.
  See [shared on-policy training](mjwarp-on-policy.md) for algorithm selection,
  checkpoints, and registration of additional agents.
- `OffPolicyTrainer` collects into shared replay for SAC, TD3 and DDPG,
  including correct final observations, optional SAC/TD3/DDPG absorbing states and
  complete expert-episode export. See [off-policy usage](mjwarp-off-policy.md).

Other MuJoCo tasks need a task adapter; unsupported tasks raise an error instead
of silently using HalfCheetah semantics. The raw simulator does not require an
RL task adapter. Rendering and arbitrary Gymnasium wrapper transformations are
not implemented by the GPU vector backend.

## Installation

For the project's TensorFlow 2.18 environment, install the optional extra using
the NJU mirror:

```bash
python -m pip install --index-url https://mirror.nju.edu.cn/pypi/web/simple -e '.[mjwarp]'
```

The existing GPU container uses TensorFlow 2.21. Keep that runtime in place and
install validation dependencies into an isolated environment. Run these commands
inside the container, from its checkout at `/main/git_repos/tf2rl`:

```bash
python -m venv --system-site-packages /tmp/tf2rl-mjwarp
/tmp/tf2rl-mjwarp/bin/python -m pip install -r requirements-mjwarp-container.txt
```

The container requirements select the NJU mirror and do not install or downgrade
TensorFlow. Run modules from the checkout, rather than installing the full
project's TensorFlow 2.18 dependency set into this TensorFlow 2.21 runtime.
TFP 0.25 is tested upstream with TensorFlow 2.18; the GPU checks documented below
exercise the PPO path with the container's TensorFlow 2.21 and tf-keras 2.21.
This does not change the dependency requirements of the research workflows.

## Use the backend independently

```python
import numpy as np
from tf2rl.envs.vector import make_vector_env

env = make_vector_env("HalfCheetah-v5", num_envs=256, backend="mjwarp")
try:
    obs, info = env.reset(seed=0)
    actions = np.zeros((env.num_envs, *env.single_action_space.shape), np.float32)
    obs, rewards, terminated, truncated, info = env.step(actions)
    env.check_overflow()
finally:
    env.close()
```

Use `tensors=True` for eager TensorFlow inputs and outputs on the GPU. Enable
TensorFlow memory growth before creating GPU tensors, so it leaves memory for
MJWarp. The DLPack bridge avoids NumPy transfers for actions and observations,
synchronizes the Warp stream before TensorFlow reads it, and exports owned
snapshots so stored rollouts cannot be overwritten by subsequent steps.

The backend autoresets each finished world on the same step. `obs` contains the
reset state for those worlds; `info["final_obs"]` contains a dense batch of states
before reset, and `info["_final_obs"]` identifies finished worlds. Time-limit
truncations bootstrap `final_obs`; true terminations bootstrap zero. Both stop
advantage and return recursion across episode boundaries.

GPU task observations are float32. MJWarp and native MuJoCo use different random
generators and precision, so equal seeds do not produce matching CPU/GPU reset
samples or bitwise-identical trajectories. Task distributions and reward
parameters are preserved. The adapters read frame skip, reset noise, action
bounds and task options from Gymnasium instead of fixing them in the simulator.

For custom models without RL semantics:

```python
import mujoco
from tf2rl.envs.mjwarp import MJWarpSimulator

model = mujoco.MjModel.from_xml_path("your_model.xml")
sim = MJWarpSimulator(model, num_envs=256, frame_skip=5)
try:
    data = sim.step(np.zeros((256, model.nu), np.float32))
    qpos = data.qpos.numpy()
    sim.check_overflow()
finally:
    sim.close()
```

To add an RL task, implement `allocate(sim, env)`, `before_step(sim)`,
`observe(sim, output)`, `evaluate(sim, reward, terminated)` and
`reset(sim, mask, seeds, episodes)`. These methods launch Warp kernels and must
support CUDA graph capture. Allocate stable buffers once in `allocate`; write
all task outputs every step, and reset only masked worlds. See
`tf2rl/envs/mjwarp_tasks.py` for two different implementations. Register a factory
with `register_mjwarp_task(YourGymnasiumEnvClass, YourTaskAdapter)` or supply an
adapter instance as `task=` to `MJWarpVectorEnv`. An adapter for models with
activation, free joints, mocap or plugin state must handle those reset fields;
the built-in scalar-joint reset helper intentionally rejects free joints and
stateful actuators.

## On-policy training and PPO validation

Run from the checkout in the GPU container:

```bash
/tmp/tf2rl-mjwarp/bin/python -m unittest discover -s tests -p test_mjwarp.py -v

/tmp/tf2rl-mjwarp/bin/python -m tf2rl.experiments.train --runner on_policy --algorithm ppo \
  --env-name HalfCheetah-v5 --env-backend mjwarp --gpu 0 \
  --num-envs 256 --rollout-steps 32 --batch-size 1024 --epochs 4 \
  --iterations 128 --output results/mjwarp/halfcheetah_ppo

/tmp/tf2rl-mjwarp/bin/python -m scripts.benchmark_env_backends \
  --env-name HalfCheetah-v5 --iterations 5
```

The common `tf2rl.experiments.train --runner on_policy` entry point accepts `--algorithm ppo`
or `--algorithm vpg`, other task IDs, and `--env-backend gymnasium`. The `ppo-train` command supplies PPO defaults through the same assembly. It supports vector Box observations, uniform
symmetric continuous action bounds, and native discrete tasks.
It saves TensorBoard events, checkpoint files and `metrics.json`, including
device placement and dependency versions. Its horizon is
`num_envs * rollout_steps`; the batch size must divide that horizon. A warmup
iteration performs real training but is excluded from steady-state timings.
Evaluation uses separate native Gymnasium episodes before and after training.

### GPU simulation with a CPU learner, playback and TensorBoard

`--env-gpu` selects the MJWarp simulation GPU independently of `--gpu`, which
selects the PPO policy and optimizer device. Use `--gpu -1` for CPU inference
and learning while leaving the GPU visible for physics. Rollouts transfer to
the learner device and CPU actions transfer to the simulator GPU. This incurs
CPU/GPU transfers and differs from the earlier fully GPU benchmark.

From the host repository root, start 40,960 simultaneous GPU worlds with a CPU
policy/learner:

```bash
docker exec -it -w /main/git_repos/tf2rl tf-gpu-tensorflow-1 \
  /tmp/tf2rl-mjwarp/bin/python -u -m tf2rl.experiments.train --runner on_policy --algorithm ppo \
  --env-name HalfCheetah-v5 --env-backend mjwarp --env-gpu 0 --gpu -1 \
  --cpu-threads 6 --num-envs 40960 --rollout-steps 32 \
  --batch-size 8192 --epochs 4 --iterations 100 --warmup-iterations 0 \
  --checkpoint-interval 1 --output results/ppo_halfcheetah_mjwarp_40960_cpu
```

Each iteration gathers 1,310,720 transitions. Checkpoints are saved each iteration
under `checkpoints/ckpt-N`; the latest five are retained. `config.json` is written
at startup, and TensorBoard events flush every measured iteration.

On the host, render the latest saved checkpoint to MP4 and open it:

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=-1 .venv-modern/bin/python -m tf2rl.training.evaluation \
  --run-dir results/ppo_halfcheetah_mjwarp_40960_cpu --episodes 3 \
  --output results/ppo_halfcheetah_mjwarp_40960_cpu/policy.mp4
xdg-open results/ppo_halfcheetah_mjwarp_40960_cpu/policy.mp4
```

Omit `--output` and `MUJOCO_GL=egl` to open a live native MuJoCo viewer in a host
terminal with a display. To select an earlier checkpoint, add
`--checkpoint results/ppo_halfcheetah_mjwarp_40960_cpu/checkpoints/ckpt-N`.
Playback uses native Gymnasium and CPU policy inference. The current container
lacks EGL rendering libraries; the host environment was verified to render
and decode the exported MP4, including a checkpoint written by TensorFlow 2.21.

In another terminal, start TensorBoard in the existing container:

```bash
docker exec -it -w /main/git_repos/tf2rl -e CUDA_VISIBLE_DEVICES=-1 tf-gpu-tensorflow-1 \
  /tmp/tf2rl-mjwarp/bin/tensorboard \
  --logdir results/ppo_halfcheetah_mjwarp_40960_cpu --host 0.0.0.0 --port 6006
```

Open <http://localhost:16006> on the host; the container maps host port 16006 to
6006. Useful scalar tags include `PPO/actor_loss`, `PPO/critic_loss`,
`PPO/mean_step_reward`, `PPO/episode_return` and `PPO/steps_per_second`.
Episode returns first appear once worlds complete an episode.

A short 40,960-world smoke run successfully used `cuda:0` simulation and CPU
actor/critic variables and updates, saved a managed checkpoint, and exported a
playable video. It did not run the 100-iteration command above.

The comparison runs sequential jobs with the same model sizes, rollout lengths,
minibatch sizes and four optimization epochs. KL early stopping is disabled for
the comparison to keep optimizer work equal. Both policies use the GPU; the CPU
baseline uses Gymnasium's serial `SyncVectorEnv`. Results are not a comparison
against an optimized multithreaded CPU simulator. Timings cover rollout and PPO
updates, excluding startup/compilation, summary writes, checkpoints, evaluation
and boundary overflow checks. First-use Warp kernel compilation adds startup
time. Small batches can be slower on MJWarp.

The integration tests check native physics/reward agreement, clipped controls,
same-step and partial resets, time-limit bootstrapping, GPU tensor ownership,
a second task with different dimensions, a free-joint raw model without a task
adapter, custom task registration/options, and actual PPO weight updates on the GPU.

## Executed results (2026-10-22)

Hardware: NVIDIA GeForce RTX 2060, 6 GiB, in the existing
`tf-gpu-tensorflow-1` container. Runtime: TensorFlow 2.21.0, tf-keras 2.21.0,
TFP 0.25.0, Gymnasium 1.3.0, MuJoCo/MJWarp 3.16.0, Warp 1.18.0, NumPy 2.4.2.
The benchmark used the settings described above and five measured iterations.

| Worlds | CPU PPO samples/s | MJWarp PPO samples/s | PPO speedup | Rollout speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 279 | 175 | 0.63× | 0.56× |
| 64 | 7,153 | 10,722 | 1.50× | 1.61× |
| 256 | 9,718 | 27,540 | 2.83× | 4.30× |
| 1,024 | 10,541 | 50,289 | 4.77× | 13.85× |

A separate PPO run used 256 worlds, 32 rollout steps, batch size 1,024, four
epochs and the normal KL limit of 0.075. It completed 128 measured iterations
(1,048,576 transitions), plus one 8,192-transition warmup iteration, with no
buffer overflows or nonfinite losses. Mean deterministic evaluation return on
three native CPU HalfCheetah episodes increased from **−433.54 to 560.61**.
Both actor and critic variables were verified on GPU. This is a single-seed
integration run, not a convergence study.
All seven integration tests passed in the GPU container. The saved PPO checkpoint
restored completely, including 4,128 actor optimizer steps, and produced finite
actions; the run also wrote TensorBoard events.

The existing TensorFlow 2.18 environment also completed a CPU `Pendulum-v1` PPO
smoke run without MJWarp installed, verifying that the feature remains optional.
The same PPO entry point completed an `InvertedPendulum-v5` GPU smoke run with
eight worlds, exercising a different observation and action dimension.
See the [recorded configuration and metrics](benchmarks/mjwarp-halfcheetah.json).
Full local run artifacts are under `results/mjwarp/`: `benchmark/comparison.json`,
per-backend logs and metrics, and `halfcheetah_ppo/` checkpoints and TensorBoard
events. `results/` is ignored by Git.

Official references: [MJWarp usage and performance](https://mujoco.readthedocs.io/en/latest/mjwarp/index.html),
[MJWarp API](https://mujoco.readthedocs.io/en/latest/mjwarp/api.html), and
[Warp DLPack interoperability](https://nvidia.github.io/warp/latest/user_guide/interoperability.html).
