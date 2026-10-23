# Off-policy training and expert trajectories with MJWarp

The shared vector collector supports SAC, TD3 and DDPG with either
`--env-backend gymnasium` or `--env-backend mjwarp`. It uses uniform, one-step
host replay and the existing algorithm implementations. Simulation and learning
devices are independent: `--env-gpu 0 --gpu -1` uses GPU simulation and CPU
inference/optimization. MJWarp task adapters currently cover HalfCheetah-v5 and
InvertedPendulum-v5; other tasks require adapters as described in [the backend
guide](mjwarp.md).

## Train and save expert episodes

Use the [existing container setup](mjwarp.md#installation), whose dependency file
uses the NJU mirror. From the host checkout, for example:

```bash
docker exec -w /main/git_repos/tf2rl tf-gpu-tensorflow-1 \
  /tmp/tf2rl-mjwarp/bin/python -m tf2rl.experiments.off_policy \
  --algorithm sac --env-name HalfCheetah-v5 \
  --env-backend mjwarp --env-gpu 0 --gpu -1 \
  --num-envs 256 --max-steps 10000000 \
  --n-warmup 10000 --batch-size 256 --units 256 256 \
  --updates-per-transition 0.015625 \
  --eval-env-backend mjwarp --eval-episodes 5 \
  --save-expert --expert-episodes 20 \
  --output results/sac_mjwarp
```

`--algorithm td3` and `--algorithm ddpg` use the same collector. The module also
has an `offpolicy-train` installed entry point and an
`examples/off_policy_run.py` launcher. `examples/sac_run.py` selects the vector
entry point when passed `--env-backend` or `--num-envs`; its existing scalar
interface remains available.

The update ratio above is an example, not an established optimum:

- A vector step advances every world once. With 256 worlds it supplies 256 new
  physical transitions, which can come from unfinished episodes.
- Warmup, `--max-steps`, checkpoint/evaluation intervals and TensorBoard steps
  count **physical transitions summed over worlds**. A limit can be exceeded by
  at most `num_envs - 1`, because all worlds advance together.
- `--updates-per-transition` counts gradient **minibatches** per new transition
  after warmup. Fractional update credit carries forward. The example spends
  four updates per vector step; at batch size 256 that samples about four replay
  rows per new physical transition on average. The default ratio 1.0 is much
  more expensive with a CPU learner.
- Increasing worlds while keeping the ratio fixed increases optimizer work
  proportionally. Lowering the ratio can improve throughput, but changes the
  learning schedule and can reduce sample efficiency. Evaluate returns when
  tuning it. `--collection-steps` controls progress-report frequency; collection
  and replay updates alternate after every vector step.
- Replay stores all physical samples using the single-world spaces. An episode
  boundary uses its final observation before autoreset. Timeouts bootstrap;
  true terminations stop ordinary SAC/TD3/DDPG bootstrapping.

The run writes `config.json`, `metrics.json`, TensorBoard `events/`, managed
`checkpoints/`, a final `checkpoint` prefix, and optional expert files under
`expert/`. Expert episodes are collected with the frozen final policy in an
independent evaluation environment. Only complete episodes are exported. Returns
are reported so the policy's quality can be assessed before using it as an expert.

```bash
docker exec tf-gpu-tensorflow-1 /tmp/tf2rl-mjwarp/bin/tensorboard \
  --logdir /main/git_repos/tf2rl/results/sac_mjwarp/events \
  --host 0.0.0.0 --port 6006
```

The container's port 6006 must be published to view it from the host. TensorBoard
includes collection/update timing, physical sample and update counts, episode
returns, evaluation returns, and algorithm losses at progress reports. Summary
writing is sampled to avoid emitting every CPU minibatch update.

## Learner execution options

For CPU optimization, `--cpu-threads` controls threads within an operation and
`--cpu-inter-threads` controls workers executing independent graph operations.
Both default to one. SAC's independent critic computations can benefit from
more graph workers. On the validation laptop, try these options together:

```bash
--gpu -1 --cpu-threads 1 --cpu-inter-threads 4
```

For GPU optimization, SAC also accepts `--xla` to compile its TensorFlow graphs:

```bash
--gpu 0 --xla
```

These switches preserve network sizes, minibatch size, gradient-update counts
and absorbing-target formulas. Compilation adds startup cost; compare warm
throughput separately. Floating-point results and random-number streams can
differ with compilation. CPU thread counts need measurement for the workload;
more threads are not always faster. Single-thread CPU XLA did not improve the
validation probe, whereas GPU XLA substantially reduced update time.

## Absorbing states

Add `--absorbing` to SAC training to select the existing DAC-aware SAC targets.
Normal observations gain a zero indicator; the absorbing observation is
`[0, ..., 0.1]` by default (`--absorbing-indicator` changes it). This works with
both vector backends.

| Event | Replay mask | Handling |
| --- | ---: | --- |
| Continuing or timeout | 1 | Preserve the physical next observation |
| True termination | 0 | Replace next observation with the absorbing state |
| Synthetic absorbing loop | -1 | Absorbing state to itself, zero action/reward |

Exactly one loop is inserted per true terminal transition. A simultaneous
termination and timeout is terminal. Synthetic loops do not increase environment
steps, episode lengths or update budgets. The existing DAC target suppresses
next-state entropy at terminals/loops and excludes loops from actor updates.
Ordinary SAC, TD3 and DDPG receive boolean termination flags instead of DAC masks.
DAC training is currently supported by SAC; `--absorbing` with TD3/DDPG is
rejected. Their ordinary exports can still be augmented for DAC imitation learning
by the expert loader.

## Export from a checkpoint

Pass matching algorithm, `--units`, and absorbing options when restoring.
For the non-absorbing SAC example above:

```bash
docker exec -w /main/git_repos/tf2rl tf-gpu-tensorflow-1 \
  /tmp/tf2rl-mjwarp/bin/python -m tf2rl.experiments.off_policy \
  --algorithm sac --env-name HalfCheetah-v5 --units 256 256 \
  --env-backend mjwarp --eval-env-backend mjwarp --gpu -1 \
  --model-dir results/sac_mjwarp --evaluate \
  --save-expert --expert-episodes 20 \
  --output results/sac_mjwarp_export
```

`--model-dir` accepts a run directory, managed checkpoint directory, or checkpoint
prefix. Without `--evaluate`, it starts a new training run from those policy and
optimizer weights, with fresh replay, counters and warmup. Replay and simulator
state are not checkpointed. Use separate output directories for new runs/exports.

## Use the trajectories for IL and IRL

Each compressed joblib file contains one world's complete episode, including
`obs`, `act`, `next_obs`, `rew`, boolean `done`, explicit `terminated`/`truncated`,
`format_version=2`, and collection metadata. The last observation is the physical
final state, not the next episode's reset state. Exported episodes contain no
synthetic absorbing transitions.

SAC exports also contain `logp`, evaluated at the exact executed actions for AIRL.
By default the expert chooses deterministic actions; `--stochastic-expert` samples
from the policy instead. TD3/DDPG exports have no stochastic action density and
can be used with GAIL/GAIfO, but do not supply AIRL's required `logp` field.

The existing demonstration loader accepts raw and already augmented observations,
adds the indicator only once, and applies the same DAC rules as online replay.
Timeouts do not create absorbing loops. Match the absorbing indicator between
expert training and imitation training when exporting augmented observations.

For example, inside the container:

```bash
/tmp/tf2rl-mjwarp/bin/python -m tf2rl.experiments.cli \
  --algorithm gail --env-name HalfCheetah-v5 \
  --expert-path-dir results/sac_mjwarp/expert --n-expert-trajectories 20 \
  --gpu -1 --logdir results/gail_from_mjwarp
```

Change `gail` to `airl`, `eqn-gail`, or `eqn-airl` for the corresponding existing
workflow. This feature accelerates off-policy expert generation; the IL/IRL
discriminator trainers retain their existing scalar collection path.

## Validation

Run `python -m unittest discover -s tests`. The off-policy tests cover scalar vs
batched DAC replay, warmup/update accounting, termination vs timeouts, per-world
expert export and IL/IRL loading, initial target copies, delayed TD3 actor updates,
parameter changes and checkpoint restoration. Optional GPU tests run real MJWarp
HalfCheetah collection with CPU SAC/TD3/DDPG updates and InvertedPendulum terminal,
timeout and absorbing transitions. GPU tests skip without MJWarp/CUDA.

The TensorFlow container also completed CLI smoke runs for ordinary SAC and
absorbing SAC, checkpoint/export operations, and short GAIL/AIRL training runs
using the GPU-generated HalfCheetah expert files. These checks verify integration
and file compatibility; they do not establish expert quality or convergence.

## Benchmark simulation and learner devices

To compare CPU/CPU, CPU/GPU, GPU/CPU and GPU/GPU with the same sample and update
budgets, run inside the container:

```bash
/tmp/tf2rl-mjwarp/bin/python scripts/benchmark_off_policy_devices.py \
  --num-envs 64 256 1024 --output results/mjwarp_off_policy/device_benchmark
```

The default uses SAC on HalfCheetah, two 256-unit hidden layers, minibatch size
256 and one gradient update per 64 new physical transitions. Each configuration
runs in a separate, sequential process. Two 10,240-transition windows warm up
tracing, kernels and replay; three additional windows provide 30,720 measured
transitions and 480 gradient updates. TensorFlow CPU training uses one thread.

`comparison.json`, `comparison.csv` and `comparison.md` include collection
throughput, training milliseconds per update and combined loop throughput. Logs,
commands, hardware, versions and actual model devices are saved alongside them.
Collection includes policy inference, transfers, simulation and replay insertion;
training includes replay sampling, transfers and optimization. Initialization,
warmup, evaluation, checkpoint I/O and progress-file writing are excluded. The
CPU environment baseline is the current serial Gymnasium `SyncVectorEnv`, not
a multiprocessing implementation. These measurements compare speed under the
specified update schedule, not policy quality.

The benchmark accepts `--xla` and `--cpu-inter-threads` as well, so these learner
execution settings can be compared with the same sampling and update budgets.
