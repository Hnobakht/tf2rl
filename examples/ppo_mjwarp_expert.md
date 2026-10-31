# PPO with MJWarp: training, playback, and expert trajectory export

Run commands in Bash from the repository root. This guide trains PPO with batched
GPU physics, saves checkpoints, and exports complete episodes from a frozen PPO
checkpoint. Use [the SAC/imitation guide](sac_imitation.md) for the downstream
GAIL, AIRL, GAIfO, and equation variants.

## 1. Prepare the GPU environment

MJWarp requires a visible NVIDIA CUDA GPU and the optional backend dependencies.
For the project's TensorFlow 2.18 environment:

```bash
source .venv-modern/bin/activate
python -m pip install -e '.[mjwarp,render]'
python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
```

The last command must show a GPU. If using the existing TensorFlow GPU container,
follow the separate [container installation instructions](../docs/mjwarp.md#installation)
instead; they preserve its TensorFlow version. Use that environment's `python`
and `tensorboard` for the commands below.

Built-in MJWarp task adapters cover `HalfCheetah-v5` and `InvertedPendulum-v5`.
Other task IDs require an adapter. GPU physics does not provide a renderer;
checkpoint playback uses native Gymnasium/MuJoCo.

## 2. Train PPO and save checkpoints

```bash
python examples/train_ppo.py \
  --env-name HalfCheetah-v5 \
  --env-backend mjwarp --env-gpu 0 --gpu 0 \
  --num-envs 256 --rollout-steps 32 \
  --batch-size 1024 --epochs 4 --units 64 64 \
  --learning-rate 0.0003 --seed 7 \
  --iterations 100 --warmup-iterations 1 \
  --checkpoint-interval 10 --eval-episodes 5 \
  --output results/examples/ppo_mjwarp
```

`--env-gpu 0` selects the physics GPU. `--gpu 0` selects PPO inference and
optimization; use `--gpu -1 --env-gpu 0` for a CPU policy with GPU simulation.
Keep the GPU visible even with a CPU learner: do not set `CUDA_VISIBLE_DEVICES=-1`
for MJWarp training or export.

One rollout contains `256 * 32 = 8192` physical transitions. The batch size must
divide that number. `--iterations` counts measured rollout/update iterations;
each of the `--warmup-iterations` also **trains**, but is excluded from timing
reports. The command therefore collects 827,392 training transitions in total.
Checkpoint intervals count measured iterations, not environment steps.

The output is written directly into the chosen directory:

```text
results/examples/ppo_mjwarp/
├── config.json              # Environment, architecture, and training settings
├── metrics.json             # Timing, losses, and initial/final evaluation returns
├── checkpoints/             # ckpt-10, ckpt-20, ...; .index plus .data files
├── checkpoint.index         # Final policy; accompanied by a .data file
└── events/                  # TensorBoard events
```

The initial/final evaluations use independent native Gymnasium episodes.
Check their returns before treating the trained policy as an expert. The example
budget is not a guarantee of a strong HalfCheetah policy. A new `--output` directory
keeps experiments separate.

## 3. Watch or record a saved checkpoint

With a display, play the latest managed checkpoint:

```bash
python -m tf2rl.training.evaluation \
  --run-dir results/examples/ppo_mjwarp --episodes 3
```

The viewer rebuilds the policy from `config.json` and uses CPU policy inference
with a native MuJoCo environment. To select a particular checkpoint or save an
MP4 on a headless machine:

```bash
MUJOCO_GL=egl python -m tf2rl.training.evaluation \
  --run-dir results/examples/ppo_mjwarp \
  --checkpoint results/examples/ppo_mjwarp/checkpoints/ckpt-100 \
  --episodes 3 --output results/examples/ppo_mjwarp/policy.mp4
```

Supply a checkpoint **prefix**, without `.index` or `.data-*`. To play the final
checkpoint explicitly, use `--checkpoint results/examples/ppo_mjwarp/checkpoint`.
MP4 export requires the rendering dependencies and working EGL libraries.
The viewer prints episode returns; it does not save expert `.pkl` trajectories.

## 4. Export expert trajectories from PPO using MJWarp

The on-policy CLI currently has no `--save-expert` flag. Export in a separate
process after training, using the shared trajectory evaluator. The small adapter
below exposes PPO's tensor action method and continuous actor log-density method
to that evaluator; it does not change the policy or run optimizer updates.

This block targets the default continuous PPO actor used by the training command
above. It restores the run configuration and final checkpoint, creates four
MJWarp worlds, and saves 20 complete episodes:

```bash
export PPO_RUN_DIR="$PWD/results/examples/ppo_mjwarp"
export PPO_EXPERT_DIR="$PWD/results/examples/ppo_expert_data"
python - <<'PY'
import json
import os
from pathlib import Path
from types import SimpleNamespace

import tensorflow as tf

from tf2rl.envs.vector import make_vector_env
from tf2rl.experiments.build import make_on_policy_agent
from tf2rl.training.evaluation import evaluate_vector_policy
from tf2rl.utils.tensorflow import configure_tensorflow

run = Path(os.environ["PPO_RUN_DIR"])
config = json.loads((run / "config.json").read_text())
configure_tensorflow(config["seed"], config["cpu_threads"])
env_kwargs = dict(config.get("env_kwargs", {}))
if config.get("max_episode_steps") is not None:
    env_kwargs["max_episode_steps"] = config["max_episode_steps"]
env = make_vector_env(
    config["env_name"], num_envs=4, backend="mjwarp", tensors=True,
    device=f"cuda:{config.get('env_gpu', 0)}",
    nconmax=config.get("nconmax", 64), njmax=config.get("njmax", 128),
    **env_kwargs,
)
try:
    policy_kwargs = dict(config.get("policy_kwargs", {}))
    if config.get("target_kl") is not None:
        policy_kwargs["target_kl"] = config["target_kl"]
    policy = make_on_policy_agent(
        config["algorithm"], env.single_observation_space, env.single_action_space,
        horizon=config["num_envs"] * config["rollout_steps"],
        batch_size=config["batch_size"], epochs=config["epochs"],
        units=config["units"], learning_rate=config["learning_rate"],
        gpu=-1, policy_kwargs=policy_kwargs,
    )
    policy.get_action_and_val_tensor(
        tf.zeros((1, *env.single_observation_space.shape)), test=True
    )
    prefix = str(run / "checkpoint")  # Or run / "checkpoints/ckpt-100".
    tf.train.Checkpoint(policy=policy).read(prefix).assert_existing_objects_matched().expect_partial()
    expert = SimpleNamespace(
        device=policy.device,
        get_action_tensor=lambda states, test=False: policy.get_action_and_val_tensor(
            states, test=test
        )[:2],
        compute_action_log_probs=policy.actor.compute_log_probs,
    )
    steps = config["num_envs"] * config["rollout_steps"] * (
        config["iterations"] + config["warmup_iterations"]
    )
    report = evaluate_vector_policy(
        expert, env, episodes=20, seed=config["seed"] + 20000,
        output_dir=os.environ["PPO_EXPERT_DIR"], total_steps=steps,
        deterministic=False,
        metadata={"algorithm": "ppo", "env_name": config["env_name"],
                  "backend": "mjwarp", "checkpoint": prefix},
    )
    report_path = Path(os.environ["PPO_EXPERT_DIR"]) / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
finally:
    env.close()
PY
```

Export uses CPU policy inference here (`gpu=-1`) and GPU physics. This can restore
weights trained on the GPU. `configure_tensorflow` enables GPU memory growth
before creating tensors, leaving memory available for MJWarp. The saved training
`horizon` reconstructs the policy; the independent export batch can have four
worlds regardless of the training batch size.

`deterministic=False` samples the Gaussian PPO policy. Change it to `True` to
export evaluation actions. Densities are evaluated at the executed, clipped
actions using this project's actor convention; the evaluator does not reuse a
pre-clipping sampled-action density. These are the framework's density values,
not a correction for probability mass at clipping boundaries.

The exporter saves one compressed joblib `.pkl` per complete episode, named
`step_*_epi_*_return_*.pkl`. Each file contains `obs`, `act`, `next_obs`, `rew`,
`done`, `terminated`, `truncated`, and `logp`, plus format/provenance metadata.
Same-step autoresets preserve the **physical final observation** in `next_obs`;
reset states are not substituted for terminal successors. Absorbing self-loops
are added later by the imitation loader, not by this exporter.

The report lists saved paths, episode returns, and episode lengths. Use a fresh
export directory to avoid mixing checkpoints or seeds with an earlier dataset.
For a CPU-only export smoke check, change `backend="mjwarp"` to
`backend="gymnasium"`, set `device="/CPU:0"`, and omit `nconmax`/`njmax`.
Update the metadata backend to `gymnasium` as well. That checks policy restoration
and file export, but does not exercise MJWarp.

## 5. Use the PPO demonstrations in imitation learning

The learner must use the same environment and compatible observations/actions as
the exported data. For example:

```bash
python examples/train_imitation.py \
  --algorithm gail --env-name HalfCheetah-v5 --gpu -1 \
  --expert-path-dir results/examples/ppo_expert_data --n-expert-trajectories 20 \
  --max-steps 100000 --n-warmup 1000 --batch-size 128 \
  --units 64,64 --units-ail 64,64 \
  --test-interval 10000 --save-model-interval 10000 \
  --logdir results/examples/gail_from_ppo
```

The imitation generator is SAC even though the expert was PPO. AIRL can also load
these files because the adapter exports `logp`; GAIfO uses their state transitions.
Use the [imitation guide](sac_imitation.md#2-train-gail-airl-eqn-gail-and-eqn-airl)
for the other algorithms, keeping `HalfCheetah-v5` and the PPO expert path.
The current imitation trainer runs a scalar Gymnasium environment; GPU expert
collection does not automatically make imitation training vectorized.

## 6. Monitor PPO with TensorBoard

In another terminal using the same environment:

```bash
tensorboard --logdir results/examples/ppo_mjwarp/events --host 127.0.0.1 --port 6006
```

Open <http://127.0.0.1:6006>. Useful tags include `PPO/actor_loss`,
`PPO/critic_loss`, `PPO/kl`, `PPO/episode_return`, `PPO/mean_step_reward`, and
`PPO/steps_per_second`. Training events are emitted for measured iterations;
initial/final evaluation returns are stored in `metrics.json`. To view PPO and
the downstream imitation runs together, use `--logdir results/examples` instead.
