# Executed validation, 2026-10-15

This is a historical record. Unit-test suites and their pytest dependency were
removed at the project owner's request on 2026-10-16; the pytest commands below
describe the earlier validation and are no longer runnable in the current tree.
Current smoke workflows and cleanup validation are described in [cleanup.md](history/cleanup.md).

The repository-local `.venv-modern` contains the base dependencies and all declared
`test`, `render`, `mujoco`, and `dev` extras. `pip check` passes. Runtime versions:
Python 3.10.12, TensorFlow 2.18.1, TensorFlow Probability 0.25.0, tf-keras 2.18.0,
Keras 3.12.4, Gymnasium 1.3.0, cpprb 11.0.0, NumPy 2.0.0, MuJoCo 3.15.0,
TensorBoard 2.18.0, setuptools 81.0.0 and pytest 7.1.3. Training used CPU.

The initial dependency download was truncated and failed checksum verification.
Upgrading project-local pip allowed resuming the download successfully. TensorBoard
then exposed its dependency on `pkg_resources`; the project now requires
`setuptools>=77,<82`. See the [upstream removal record](https://setuptools.pypa.io/en/latest/history.html#v82-0-0).

## Actual SAC → GAIL → EQN-GAIL experiment

Each stage completed 20,000 environment steps on Pendulum-v1 with 200-step episodes,
1,000 random warmup steps, 64/64 actor and critic widths, batch size 128, replay
capacity 50,000, automatic entropy tuning, and checkpoints/evaluations every 5,000
steps. Imitation used batch size 32, one discriminator update per step for dense
GAIL, and one every three steps for EQN-GAIL. EQN width `2,1`, dropout `.5,0`,
singular penalties, log-odds rewards and default GP=0 remained active. Dense GAIL
retains its log(D) rewards. Both imitation runs use the same final SAC data.

| Algorithm | Seed | Return at 5k | At 10k | At 15k | At 20k | Evaluation episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAC | 7 | -530.130 | -120.113 | -138.303 | -177.608 | 20 |
| GAIL | 11 | -176.629 | -186.705 | -210.726 | -134.889 | 10 |
| EQN-GAIL | 13 | -1041.970 | -233.519 | -157.197 | -205.524 | 10 |

These are mean deterministic-policy evaluation returns. Evaluation episodes and
seeds differ across algorithms; these numbers are integration evidence, not a
controlled algorithm ranking or a reproduction of the paper's benchmark.

Records are retained under:

- `results/validated_pipeline/sac/20261015T093529.962998_SAC_pendulum_seed7`
- `results/validated_pipeline/gail/20261015T093747.748649_SAC_pendulum_seed11`
- `results/validated_pipeline/eqn-gail/` (the run directory records its exact command).

The SAC run exports 20 final ordered demonstrations, 4,000 transitions total, with
explicit termination/truncation flags and action log probabilities. Earlier
evaluation exports are retained too; the imitation loader selects the latest 20.
Checkpoints include optimizer state and target critics. Initial 300-step smoke runs
under `results/validation` predate the complete target-checkpoint repair; use the
longer `results/validated_pipeline` runs for further work.

All logged scalar values are finite: 12 metric series for SAC and 17 for each
imitation run. TensorBoard startup and its HTTP scalar endpoint were checked.
The dashboard is served at <http://127.0.0.1:6006>. Restart it with:

```bash
export TMPDIR="$PWD/.tmp" MPLCONFIGDIR="$PWD/.cache/matplotlib" KERAS_HOME="$PWD/.cache/keras"
.venv-modern/bin/tensorboard --logdir results/validated_pipeline --host 127.0.0.1 --port 6006
```

Use `scripts/run_validation_pipeline.py` to repeat the three stages. Its complete
300-step rehearsal also passed under `results/pipeline_script_smoke`. Use
`scripts/inspect_training_records.py results/validated_pipeline` to independently
check all logged values, final steps, checkpoint paths and demonstration integrity.
It writes `validation_summary.json` and `environment_packages.json` into the log
directory. Exact experiment arguments and source diffs accompany every run.

## Tests and absorbing states

The focused suite covers research algorithms, SAC, dense GAIL, D2RL-SAC, utilities
and custom MuJoCo tasks:

```bash
.venv-modern/bin/python -m pytest tests/research tests/algos/test_sac.py \
  tests/algos/test_gail.py tests/algos/test_d2rl_sac.py tests/misc \
  tests/envs/test_utils.py -q --basetemp=.tmp/pytest-core-final
```

The initial focused run passed 95 tests. The two added world-geometry tests and the
nonzero L1 value/gradient regression test also passed separately.
The research suite now contains 61 tests. In particular:

- EQN operator values, singular penalties and finite-difference gradients match
  independent calculations. AIRL/GAIL losses, gradient penalty and DAC target/actor
  formulas pass analytic checks.
- All four discriminators initialize and update. EQN delayed updates are tested.
- Saving and restoring policy, discriminator, optimizer, online critic and target
  critic weights passes full checkpoint consumption checks.
- A two-step terminating environment drives both GAIL collectors through real
  training. Each episode produces replay masks `[1, 0, -1]`, exactly one zero-action
  absorbing self-loop and the expected terminal next state. Expert loading produces
  the same masks, including simultaneous terminated/truncated flags.
- Terminal/self-loop targets use the zero action and omit next-policy entropy.
  All-absorbing batches produce finite updates and leave actor weights unchanged.
- TensorBoard integration checks require finite loss/return/mask metrics at the
  final step, and ensure the caller's summary context is restored.
- All 18 custom MuJoCo tasks pass Gymnasium's checker. Two independent tests compare
  compiled body and geometry world positions to the original global-coordinate XML.

Pendulum has no true termination; its absorbing/terminal batch fractions correctly
remain zero. The terminating-environment tests are essential evidence for that
branch. Wrapper bounds, float dtypes, short demonstrations, legacy metadata fallback,
already augmented inputs and disabled absorption are separately covered.

## Remaining limits

The complete suite was executed: **185 passed, 10 failed, 17 setup errors**.
Its report is `.work/full-tests.xml`. All research tests present at that point
passed within the full run; the L1 regression was added and passed afterward.
Remaining failures concern Ape-X multiprocessing, PPO/VPG return interfaces,
TD3's optimizer API, unavailable Atari registration, noisy layers, discrete SAC,
VAIL and legacy layer serialization tests. Those paths are not certified by these
SAC/GAIL/EQN runs. They were not hidden or marked passing.

GPU/XLA, rendering, legacy checkpoint migration, full multi-seed MuJoCo learning
curves and the paper's original datasets/experiments remain unverified. Native
MuJoCo geometry checks do not establish numerical equivalence to older physics.
Seeded Python/NumPy/TensorFlow/environment setup does not promise bitwise identical
results across library versions or replay-buffer RNG implementations. Resuming a
checkpoint restores weights/optimizers, not replay contents or environment/RNG state.


## Research architecture refactor (2026-10-29)

The implementation now follows the research package layout documented in
[architecture](architecture.md). CPU validation used the existing TensorFlow
2.18.1, TensorFlow Probability 0.25.0, Gymnasium 1.3.0 environment.

- Seven matched before/after runs produced exactly equal values for all 810 saved
  policy, target-network, optimizer and discriminator tensors. Coverage included
  vector SAC, PPO, scalar expert SAC, AIRL, GAIL, EQN-AIRL and EQN-GAIL. Runs used
  16 physical transitions, small networks, and the same baseline demonstrations.
- The existing 300-step-per-stage SAC → GAIL → EQN-GAIL workflow completed.
  Its record inspector reported finite metrics, saved checkpoints and expert I/O.
- Focused executed checks covered true termination, truncation, simultaneous
  termination/truncation, absorbing self-loops, physical-step accounting, and
  on-policy truncation bootstrapping.
- All 66 inspected algorithm, network, data, training, task and wrapper modules
  imported. Hopper, HalfCheetah and maze research environments reset and stepped
  through their relocated task definitions and original XML assets.
- All ten Hydra presets and their three hyphenated aliases composed. A Hydra PPO
  child run and the custom-algorithm example completed. A pre-refactor SAC
  checkpoint restored and evaluated through the new entry point.
- Existing example smoke commands, compilation, focused static checks and
  wheel/source package builds passed. The extracted wheel also completed a VPG
  training run from outside the checkout.

Matched-checkpoint records and pipeline metrics are under
`.work/architecture-refactor/`. These checks establish compatibility for the
exercised short CPU paths. GPU/MJWarp execution and the older experimental methods
were not revalidated in this refactor.

## Architecture cleanup (2026-10-29)

Removed 108 Python files outside the agreed layout: old import/launcher shims,
obsolete examples and package markers, the duplicate scalar on-policy trainer,
the unused dm_control viewer, and `setup.py`. Removed the three old Hydra runner
templates and three hyphenated preset aliases. The canonical algorithm
implementations and the experimental methods included in the design remain in
their designated packages.

Scripts, benchmark commands, debugger configurations and current documentation
now use `tf2rl.experiments.train` with an explicit runner. Playback is available
through `python -m tf2rl.training.evaluation`. The editable installation was
refreshed so local console scripts import the current entry points. Console
wrappers also discard programmatic result objects so successful training exits
with status zero.

Validation after pruning:

- No runtime references to deleted modules remain; all local Python import paths
  resolve. Compilation, static checks and example smoke commands passed.
- Short SAC and PPO example runs completed. Installed SAC, PPO and expert console
  commands completed with zero exit status; the on-policy command and viewer
  displayed help successfully.
- All ten canonical Hydra presets composed, and an installed Hydra PPO job ran.
- The 300-step SAC → GAIL → EQN-GAIL pipeline completed with finite metrics.
- Wheel and source archive builds passed; neither contains any deleted Python
  path or removed Hydra template. Packaging uses `pyproject.toml` directly.

Local pipeline records and the removal inventory are under
`.work/architecture-prune/`. GPU/MJWarp execution was not rerun for this cleanup.
