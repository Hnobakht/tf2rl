# Source cleanup, 2026-10-16

The project owner requested consolidation of Python implementations, removal of
unnecessary code/comments, and removal of all unit-test suites. Git branches were
not merged. Existing expert demonstrations, checkpoints, experiment arguments,
TensorBoard records, environment assets and local environments were preserved.

## Consolidated implementations

- `tf2rl/experiments/irl_trainer.py` contains the current DAC collector and training
  loop. `airl_trainer.py` provides its AIRL-specific input adapter. The previous
  `_v2` modules only re-export those classes for existing callers.
- `tf2rl/networks/eqn.py` remains the shared equation implementation for EQN-AIRL
  and EQN-GAIL. `networks/layer.py` keeps only compatibility aliases and the distinct
  regularized output layer still imported by experimental GAIfO/VAIL code.
- Removed the unused alternate replay-buffer module and obsolete equation-layer
  implementations from the runtime package.
- Moved 37 historical algorithm/trainer/launcher files to `legacy/python` and
  preserved the old combined layer module there as well. Its manifest verifies all
  38 originals by SHA-256. The archive is excluded from wheel and source packages.
  See [the version map](../legacy/python/README.md).

Different scientific variants were not treated as equivalent. Old log/div branches,
reversed AIRL, old SAC critic arrangements, and different GAIfO/VAIL objectives are
preserved as historical source rather than merged into current equations. The
expert SAC and DAC SAC-IRL remain distinct because their critics and targets differ.
Archived launchers require their historical interfaces and dependencies.

## Removed material

- Project unit tests, their helpers and configuration, and bundled Gym unit tests:
  110 test/support/configuration files. Gym's toy `envs/unittest` environments are
  environment implementations, not test suites, and remain with the historical source.
- The `test` dependency extra, pytest requirement and pytest packaging configuration.
  Installed third-party dependencies inside virtual environments were left alone.
- More than 500 lines of commented-out code and disabled decorators. Each comment-only
  edit was checked for identical Python syntax trees before and after removal.
- 27 unused imports, including an unnecessary external `EQL.layer` import.

Comments explaining equations, checkpoint tracking, termination semantics and
resource ownership remain. The local source backup created before cleanup is
`.work/before-source-cleanup-20261016.tar.gz`.

## Validation

Before deleting the suites, 107 selected research, SAC/GAIL/D2RL-SAC, utility and
environment checks passed. The report is `.work/cleanup-validation.xml`. This is
historical evidence; the unit-test sources no longer ship with the project.

After removal, the following executable workflows passed without pytest:

- SAC, GAIL and EQN-GAIL: 300 training steps each on Pendulum-v1, including expert
  export/import, evaluation, checkpoints and TensorBoard records. The inspector
  verified 12 finite scalar series for SAC and 17 for each imitation run.
  Results: `results/cleanup-smoke/validation_summary.json`.
- AIRL and state-only EQN-AIRL: 40 training steps each using those demonstrations,
  with policy and discriminator checkpoints saved under `results/cleanup-smoke`.
- Canonical trainer classes and their v2 import aliases resolve to the same objects.
- All five example entry points and the renamed smoke shell script pass. Static
  project-module import checks and historical-source hash verification pass.
- Requirements resolution against the local environment and `pip check` pass.
- Wheel and source-distribution builds pass; their contents exclude removed tests,
  historical variants and the legacy archive.

CI now runs the short SAC/GAIL/EQN-GAIL workflow, CLI startup and package builds.
The old `test_run_examples.sh` is now `scripts/smoke_examples.sh`. These are actual
experiment workflows, not a replacement unit-test suite. Short runs validate
execution, not learning quality or reproduction of the paper. Experimental
Atari/DMC/VAIL/WAIL/GAIfO paths still have the previously documented limitations.
