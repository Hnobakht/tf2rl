# Compact Pendulum equation discriminators

The first sparsity run removed weights but retained almost all operators. Its
raw learned logit had 64 operations, including 24 nonlinear operations. Counting
only zero weights understated the size of the resulting formula.

To prioritize the smallest useful equation, the compact experiments restrict the
existing configurable EQN vocabulary and depth, then train with staged L1 and
persistent pruning. The adversarial loss, SAC generator and DAC masks are
unchanged. This is an explicit architecture/pruning choice, not evidence that the
full operator vocabulary automatically discovered the same structure. Standard
EQN defaults remain available and unchanged.

## Reproduce a compact run

Set `EXPERT_DIR` to the existing Pendulum SAC demonstration directory. The shared
parameters below reproduce the selected two-term candidate:

```bash
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/.tmp"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export KERAS_HOME="$PWD/.cache/keras"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$KERAS_HOME"
EXPERT_DIR=results/equation-learning/sac/20261017T083343.306645_SAC_

.venv-modern/bin/python -m tf2rl.experiments.train --runner imitation \
  --algorithm eqn-gail --env-name Pendulum-v1 --seed 13 \
  --expert-path-dir "$EXPERT_DIR" \
  --max-steps 20000 --n-warmup 1000 --units 64,64 \
  --batch-size 128 --batch-size-ail 32 --memory-capacity 50000 \
  --v 1 --dropout 0 --exclude sin-cos-sig-log-mult-div \
  --eqn-l1 .003 --eqn-stages 1000,3500 --eqn-prune-threshold .15 \
  --export-equation --test-interval 5000 --test-episodes 10 \
  --save-model-interval 5000 --save-summary-interval 100 \
  --logdir results/compact-equations/minimal
```

This starts with 8 parameters and an identity-only equation layer. The network is
affine in the observations and action; Pendulum's observations already contain
`cos(theta)` and `sin(theta)`. The absorbing-indicator input is still present.
Pruning can remove individual inputs. The less aggressive candidate used `.03`
and a separate `--logdir`. Large thresholds can remove all useful dependencies,
so judge the resulting expression and policy together.

The comparison quadratic candidate changes only `--exclude` to
`sin-cos-sig-log-div` and uses threshold `.03`, retaining identity and
squared-binomial operators. It starts
with 21 parameters. The original squared-binomial formula is unchanged.

## Inspect and compare

Use actual run directories, each containing `args.txt` and policy checkpoints:

```bash
.venv-modern/bin/python scripts/inspect_equation.py "$COMPACT_RUN"
.venv-modern/bin/python scripts/compare_equation_policies.py \
  "$PREVIOUS_RUN" "$COMPACT_RUN" \
  --episodes 50 --first-seed 1000 \
  --output results/compact-equations/comparison.json
.venv-modern/bin/tensorboard --logdir results/compact-equations --port 6006
```

The comparison restores each latest policy and uses deterministic actions from
identically seeded Pendulum starts. It also evaluates uniform-random actions.
The report contains every episode return, variation across starts, raw-expression
complexity and retained-parameter counts. The 50 starts are an additional check
of these trained candidates, not independent training seeds or a benchmark claim.

`inspect_equation.py` now reports `raw_expression` complexity. For future runs,
`--export-equation` also records these checkpoint-time TensorBoard metrics:
`EQN/expression_operations`, `EQN/expression_nonlinear_operations`,
`EQN/expression_affine_coefficients`, and `EQN/expression_inputs`.
The counts follow dependencies of the actual raw output and exclude GAIL's fixed
sigmoid/log-odds wrapper. An affine operation can have multiple coefficients;
these counts are not polynomial monomial counts.

The raw expression is a discriminator logit `z`, not exactly the final GAIL reward.
The implementation still uses

\[
D=\frac{1}{1+e^{-z}},\qquad
r=\log(D+10^{-8})-\log(1-D+10^{-8}).
\]

The exported JSON retains the stored float32 epsilon and every relaxation
constant. Use the existing JSON-to-LaTeX tool to view the learned raw expression:

```bash
.venv-modern/bin/python scripts/equation_json_to_latex.py \
  "$COMPACT_RUN/equations/step_00020000.json" \
  --select raw_output --expanded --precision 6 --document
```

`--precision 6` rounds only the displayed formula. Check the unrounded JSON or
NumPy module for numerical evaluation; the trained checkpoint is unchanged.


## Selected compact result (2026-10-17)

The aggressive candidate (`--eqn-prune-threshold .15`) completed 20,000 steps and
retained 3 of its 8 trainable parameters. After composing the two affine layers,
the logit has just **two nonzero coefficients**:

\[
z(\theta,a)\approx 0.499064\cos(\theta)+0.0636128a.
\]

There is no bias term. Sine, angular velocity and the absorbing indicator were
pruned from this Pendulum discriminator. The action is the environment torque
in `[-2,2]`. GAIL's sigmoid/log-odds reward transformation still applies; the
training objective did not substitute this logit for the implemented reward.
This is a compact learned discriminator, not Pendulum's true reward equation.

Using each candidate's final checkpoint on the same 50 starts (seeds 1000–1049):

| Candidate | Retained parameters | Raw graph operations | Mean return |
| --- | ---: | ---: | ---: |
| Previous sparse EQN, full operator set | 165 | 64 | −156.155 |
| Compact linear, threshold .03 | 6 | 2 | −175.555 |
| Compact quadratic, threshold .03 | 18 | 6 | −430.026 |
| Selected compact linear, threshold .15 | 3 | 2 | −165.276 |
| Uniform-random actions | Not applicable | Not applicable | −1276.466 |

This supports using the two-term candidate for the requested compactness-first
tradeoff. It is the smallest useful model among the tested candidates, not a
proof that no smaller model can work. All candidates have one trained seed;
multiple training seeds and new evaluation starts are needed for broader claims.

The selected run is
`results/compact-equations/minimal/20261017T152314.227816_SAC_`.
Its `equations/step_00020000.{json,txt,py}` files retain the complete original
inference graph; `compact_formula.tex` and `compact_formula.pdf` present the
composed formula. `equation_validation.json` records a maximum exported-reward
error of `5.36e-7` across 1,001 diagnostic inputs. Masks remained zero, all three
sparsity phases were recorded, and all recorded training metrics were finite.
`results/compact-equations/comparison.json` retains all evaluation returns.

Play the selected policy:

```bash
.venv-modern/bin/python -m tf2rl.experiments.train --runner imitation \
  --algorithm eqn-gail --env-name Pendulum-v1 --units 64,64 \
  --evaluate --show-test-progress --test-episodes 5 \
  --model-dir results/compact-equations/minimal/20261017T152314.227816_SAC_ \
  --logdir results/playback
```

During validation, Python exports written while their source was being edited
were regenerated from unchanged JSON graphs. The exporter now captures evaluator
source at module import, preventing that mid-run source-edit failure. Saved model
parameters and JSON coefficients were unaffected.
