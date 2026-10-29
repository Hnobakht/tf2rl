# Sparse equation learning

EQN-GAIL and EQN-AIRL now support L1 regularization, a three-stage sparsity
schedule, persistent pruning, and inference-equation export. These are optional
algorithmic extensions. Existing commands default to `--eqn-l1 0`, without masks
or a sparsity schedule, and retain the existing objectives and checkpoint layout.
No new dependencies are required.

## Run on Pendulum

From the repository root, using the existing environment:

```bash
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/.tmp"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export KERAS_HOME="$PWD/.cache/keras"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$KERAS_HOME"
.venv-modern/bin/python scripts/run_equation_learning.py
```

This runs 20,000 environment steps each of SAC, baseline EQN-GAIL and staged
EQN-GAIL. SAC supplies its final 20 evaluation trajectories. Both imitation runs
use the same demonstrations, seed 13, architecture and training budget. Run
arguments, checkpoints, TensorBoard events and equations are retained under
`results/equation-learning`. To reuse an existing Pendulum SAC expert, pass
`--expert-dir results/expert/ACTUAL_RUN_DIRECTORY`.

To train only the staged variant:

```bash
.venv-modern/bin/python -m tf2rl.experiments.train --runner imitation \
  --algorithm eqn-gail --env-name Pendulum-v1 \
  --expert-path-dir "$EXPERT_DIR" --seed 13 \
  --max-steps 20000 --n-warmup 1000 --units 64,64 \
  --batch-size 128 --batch-size-ail 32 --memory-capacity 50000 \
  --eqn-l1 .001 --eqn-stages 1000,4000 --eqn-prune-threshold .01 \
  --export-equation --test-interval 5000 --test-episodes 10 \
  --save-model-interval 5000 --save-summary-interval 100 \
  --logdir results/equation-learning/sparse
```

The same new options work with `--algorithm eqn-airl`, with or without
`--state-only`. Its dense shaping potential remains unchanged.

For substantially smaller formulas, see the [compact Pendulum experiments](compact-equations.md),
which use a restricted one-layer operator vocabulary and compare saved policies
on shared evaluation starts.

## Training semantics

With the new mode enabled, the discriminator objective is

\[
L=L_{\mathrm{original\ adversarial}}+P_{\mathrm{original\ singularity}}
 +\lambda_t\sum_l\left(\lVert W_l\rVert_1+\lVert b_l\rVert_1\right).
\]

The sum includes every equation hidden layer and the output layer, including
biases. It is not divided by the parameter count. The original singularity
penalty remains active throughout; it serves a separate purpose from sparsity.
A nonzero Python `lmbda` now enables this network-wide L1 objective, fixing its
previously ignored hidden-layer/GAIL behavior.

`--eqn-stages W,S` counts **completed discriminator optimizer updates**, not
environment steps or episodes. Skipped updates do not advance the schedule;
`--n-training` repeated updates do. The schedule is:

| Phase | Updates | Behavior |
| --- | --- | --- |
| 0 | First W | Original loss, L1 coefficient zero |
| 1 | Next S | Original loss plus configured L1 |
| 2 | Remaining updates | Prune once, disable L1, fit surviving parameters |

At entry to phase 2, weights and biases with absolute value **strictly below**
`--eqn-prune-threshold` are set to zero and masked permanently. The effective
forward parameters use these masks, and projection after every optimizer step
prevents Adam momentum from regrowing a removed parameter. Masked values therefore
stay zero in both training and inference. This is parameter pruning; dense tensor
shapes and allocated optimizer state are retained.

With the Pendulum command above, the discriminator updates every three steps.
It performs 6,333 updates: 1,000 initial updates, 4,000 L1 updates, and 1,333 refit
updates. Pruning happens at environment step 16,002. Choose stages that leave
refitting time for your actual budget. Setting only `--eqn-l1` enables continuous
L1 without automatic pruning; a schedule requires a positive coefficient.

Masks, completed updates, phase, boundaries, coefficient and threshold are saved
with the discriminator and optimizer under `irl/`. Resume using `--model-dir`
and `--irl-model-dir`, the same architecture, and the sparsity options. Saved
schedule values take precedence over their constructor defaults. The CLI rejects
restoring a sparse checkpoint into a model without sparsity, or vice versa.
Replay, environment state and random streams are still not restored, so resumed
online learning is not an exact continuation of an interrupted rollout.

The legacy EQN-GAIL `atol` inference-copy threshold and `tau` target averaging
cannot be combined with this mode. They keep their previous behavior for existing
non-sparse models. Dense AIRL/GAIL objectives, SAC, DAC transition masks, dropout,
and the EQN operator definitions are unchanged.

The phase design follows the sparsity/refitting idea in
[Sahoo, Lampert and Martius, Section 2.3](https://proceedings.mlr.press/v80/sahoo18a/sahoo18a.pdf).
This implementation adapts it to adversarial updates and regularizes biases as
well as weights. It is not a full reproduction of that paper's supervised
training, division curriculum, penalty epochs or model selection, nor Werner et
al.'s structured L0 gates. The repository's relaxed log/division and
squared-binomial operators remain exactly the implemented research operators.

## Equations and TensorBoard

With `--export-equation`, every checkpoint and the final training step write:

- `equations/step_XXXXXXXX.txt`: readable expressions with shared intermediate terms.
- `equations/step_XXXXXXXX.json`: coefficients, operator graph, named inputs, outputs
  and training metadata.
- `equations/step_XXXXXXXX.py`: standalone NumPy module exposing `predict(batch)`;
  it needs neither TensorFlow nor this project.

Exports use inference mode, so dropout is disabled. Zero-weight edges are removed,
constant expressions are folded, repeated terms are shared, and unreachable
expressions are discarded. Surviving coefficients are not rounded or thresholded
again. This simplifies the actual graph; it does not promise a minimal canonical
symbolic formula or perform unrestricted algebraic cancellation.

For EQN-GAIL, outputs include the raw logit, probability `D`, and the actual reward
`log(D + 1e-8) - log(1 - D + 1e-8)`. Keeping the epsilon matters near saturation.
For EQN-AIRL the exported reward is the raw equation-network output; the separate
dense shaping potential is not included. Pendulum input order is
`cos_theta, sin_theta, theta_dot, absorbing_indicator, action_0`; state-only AIRL
omits the action. Other environments use observation/action indices. Inputs must
use the same observation preprocessing and action units as training.

```bash
.venv-modern/bin/tensorboard --logdir results/equation-learning --host 127.0.0.1 --port 6006
```

Open <http://127.0.0.1:6006>. In addition to returns and discriminator losses,
sparse runs record `EQN/phase`, `EQN/l1_coefficient`, `EQN/l1_norm`,
`EQN/active_parameters`, `EQN/pruned_fraction`, and `EQN/discriminator_updates`.
Active-parameter counts describe retained mask entries, including parameters in
branches that export may subsequently discard. Scalars follow the existing summary
interval and discriminator update delay; very short stages may have no recorded
sample. Use a summary interval divisible by the update delay to sample every
scheduled summary point.

## Convert exported JSON to LaTeX

The standalone tool uses only Python's standard library:

```bash
.venv-modern/bin/python scripts/equation_json_to_latex.py "$EQN_RUN_DIR/equations" --document
```

It converts every `.json` directly inside the equation directory into a neighboring
`.tex` file. You can also pass one JSON filename or multiple filenames/globs. JSON
inputs are unchanged; existing destination `.tex` files are replaced. Outputs must
remain inside the repository.

By default, shared terms `t_i` keep the formula readable, and all stored coefficient
precision is retained. Fractions, natural logarithms, maximum/clamping, squares,
and the explicit sigmoid formula preserve the exported graph's operations,
including relaxation constants and reward epsilon. Pendulum inputs use
`cos(theta)`, `sin(theta)`, angular velocity, absorbing indicator, and action notation.

For a rounded display of only the reward:

```bash
.venv-modern/bin/python scripts/equation_json_to_latex.py \
  "$EQN_RUN_DIR/equations/step_00020000.json" \
  --select reward --precision 6 --document -o .work/reward.tex
```

`--precision` means significant digits, not decimal places. Rounded outputs are
marked with an approximation sign; this is a presentation choice, not a change to
the trained model. Omit it for full precision. `--expanded` substitutes every
intermediate term into the output formula. Expanded networks can be very long and
exceed a page's width; shared terms are preferable for the Pendulum discriminator.
`--max-chars` limits generated formula size (default 200,000 characters).

With `--document`, the file includes an `amsmath` preamble and can be compiled with
LaTeX. Without it, the output is an `align*` fragment for a document that already
loads `amsmath`. `--output`/`-o` is supported for a single JSON input.

Validation rendered all 12 saved EQN-GAIL/EQN-AIRL export graphs at full and rounded
precision. The full Pendulum document, a rounded document, and a small expanded
fraction compiled successfully with `pdflatex`; no training is needed for conversion.

## Validation and limits

```bash
.venv-modern/bin/python scripts/audit_equation_learning.py
.venv-modern/bin/python scripts/inspect_equation.py "$EQN_RUN_DIR"
.venv-modern/bin/python scripts/inspect_training_records.py results/equation-learning
```

The first command checks loss/gradient behavior, delayed updates, masked Adam
steps, exact optimizer/checkpoint continuation, independent NumPy export, both
AIRL input modes, gradient penalties, constant simplification, and DAC termination
versus truncation. It also fits a known quadratic with a deliberately restricted
operator vocabulary. Results go into `.work/equation-learning-audit/report.json`.
These executable diagnostics do not restore the removed unit-test suite.

`inspect_equation.py` restores a Pendulum discriminator, compares its export on
1,000 physical observations/actions and one synthetic absorbing input, checks
persistent zeros and update counts, and verifies TensorBoard sparsity records.
It writes `equation_validation.json` in the run directory. The training workflow
runs this check for both EQN-GAIL variants and checks finite training metrics.

Pendulum terminates through a time limit, so it exercises truncation and should
have zero absorbing self-loops in collected data. The numerical diagnostic
separately checks true termination and simultaneous termination/truncation.
A sparse discriminator remains an adversarially learned reward: recovering
Pendulum's physical reward equation, a uniquely identifiable expression, or a
particular policy score is not guaranteed. Single-seed results establish an
integration check, not a benchmark improvement or paper reproduction.

## Executed results (2026-10-17)

CPU runs in the project-local `.venv-modern`, Gymnasium `Pendulum-v1`:

| Run | Environment steps | Final mean evaluation return | Retained equation parameters |
| --- | ---: | ---: | ---: |
| SAC expert, seed 7 | 20,000 | −177.608 (20 episodes) | Not applicable |
| Original EQN-GAIL, seed 13 | 20,000 | −205.524 (10 episodes) | 251 / 251 |
| Staged EQN-GAIL, seed 13 | 20,000 | −205.584 (10 episodes) | 165 / 251 |

The staged run froze **86 parameters at zero (34.26%)** and completed all three
phases. The final returns were similar in this one comparison; learning curves
varied, so this does not establish equal performance or superiority. The exported
graph retained 79 expression nodes versus 80 for the baseline: removing connections
did not remove most operator nodes, and the Pendulum expression is still fairly
large. A compact physical reward equation has not been recovered.

On 1,001 diagnostic inputs, maximum absolute exported-reward error was
`5.96e-7` for the sparse discriminator and `7.15e-7` for the baseline. Saved masks
remained zero and TensorBoard recorded phases 0, 1 and 2. All recorded training
scalars were finite. The expert supplied 4,000 transitions in its final 20 episodes.

A 1,200-step state-only EQN-AIRL smoke run also completed sparsification, final
checkpoint saving outside the periodic save interval, and export validation
(maximum reward difference `5.96e-8`). Its return is not a convergence result.
A CLI continuation of the sparse EQN-GAIL run added 31 discriminator updates,
retained all 86 pruned zeros, and kept phase 2 and its zero L1 coefficient.
The optimizer diagnostic exercised both state-only and state-action EQN-AIRL,
nonzero GAIL gradient penalty, masked-zero persistence, and exact checkpoint
continuation. A separate comparison against the previous committed equation
network produced bitwise-identical losses and parameters over ten default-mode
updates and restored its default checkpoint successfully.

The restricted supervised quadratic diagnostic recovered approximately
`1.5*x**2 - 0.7*x + 0.2`, with maximum error `9.54e-7` on `[-2,2]` after training
on `[-1,1]`. Its nine parameters survived the chosen pruning threshold; the
separate discriminator diagnostic exercised 122 genuinely pruned parameters.
This establishes coefficient learning for a selected vocabulary, not general
symbolic discovery from demonstrations.

Local evidence and artifacts:

- `results/equation-learning/validation_summary.json`: metrics and evaluation curves.
- `results/equation-learning/sparse/20261017T083946.144912_SAC_/equation_validation.json`:
  restored-model/export comparison and sparsity counts.
- `results/equation-learning/sparse/20261017T083946.144912_SAC_/equations/step_00020000.txt`:
  final learned expression; adjacent JSON and Python files provide executable forms.
- `.work/equation-learning-audit/report.json` and `backward_compatibility.json`:
  numerical diagnostics and previous-network comparison.

Results and `.work` artifacts remain ignored by Git; the commands above reproduce
new runs without relying on those local artifacts. No legacy implementation,
experiment configuration or existing result was removed for this feature.
