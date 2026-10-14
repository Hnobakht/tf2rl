# Validation record

Validation was performed in a fresh repository-local `.venv-modern` environment,
Python 3.10.12, with system site packages disabled. Offline cached wheels were copied
into `.work/wheels` and installed locally; the external cache and installations were
not modified. Available runtime versions are Gymnasium 1.3.0, NumPy 2.0.0,
pytest 7.1.3, SciPy 1.15.3 and joblib 1.6.0.

## Executed successfully

The following selected suite passed 20 tests. Eighteen native MuJoCo cases were
skipped because the optional MuJoCo runtime was not installed:

```bash
.venv-modern/bin/python -m pytest \
  tests/research/test_absorbing.py \
  tests/research/test_data_and_wrappers.py \
  tests/research/test_research_mujoco.py \
  tests/envs/test_utils.py tests/misc/test_discount_cumsum.py \
  tests/misc/test_utils.py --basetemp=.tmp/pytest-modern -q
```

It checks wrapper bounds/dtypes/API, terminal and truncation replay masks, absorbing
self-loops, disabled absorption, short/legacy/already-augmented demonstrations,
maximum-step slicing, aligned noise mixing without replacement, zero-noise handling,
epoch-penalty sampling, file selection, frame stacking, learned rewards space types and discounted cumulative sums.

All tf2rl/examples/tests Python modules were parsed without syntax errors. Original
XML assets were compared with their packaged copies to verify byte preservation.
Editable installation (`--no-deps --no-build-isolation`) and wheel building passed
without importing TensorFlow. The wheel metadata and `eqn-train` console entry point
were verified; all 31 XML assets are included and the legacy Gym tree is excluded.
`git diff --check` passed. Installing without dependencies verifies packaging only;
the resulting environment is not yet a complete training runtime.

## Not verified

The research suite cannot collect its TensorFlow modules in the current environment:
`ModuleNotFoundError: No module named 'tensorflow'`. The full suite was also run with `--continue-on-collection-errors`: **20 passed,
18 skipped, 27 collection errors** (missing TensorFlow/cpprb/OpenCV dependencies).
The raw output is saved locally in `.work/full-test-results.txt`. cpprb and TensorFlow Probability are also unavailable.
Network installation retries timed out; a 615 MB TensorFlow download progressed at
about 34 kB/s and was stopped rather than waiting several more hours.

Tests are included for analytic EQN outputs/penalties and finite-difference gradients,
AIRL/GAIL losses and gradients, agent-only GP, DAC actor/critic formulas, model
initialization and updates, delayed updates, all-absorbing batches, TD errors, and
complete export/import/train/evaluate/checkpoint workflows for all four algorithms.
These tests have **not passed locally**; their presence is not evidence that model
execution is validated. CLI/training smoke execution also requires these runtimes.
The research GitHub Actions workflow installs them and runs this suite, but no CI run
has been triggered or observed during this session.

Native custom tasks have 18 optional runtime tests. MuJoCo execution/rendering,
legacy checkpoint migration, GPU/XLA behavior, optional Atari/DMC paths, full-length
learning curves and the paper's experiment results remain unverified. Original
experiments and demonstrations are necessary for scientific replication.
