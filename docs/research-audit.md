# Research code review — 2026-10-24

This review covers the active EQN/AIRL/GAIL workflow, its SAC generator, the
shared on/off-policy launchers, collection boundaries, and equation export.
It adds mathematical documentation and small execution repairs. No tests or test
dependencies were added. Existing experiment scripts and temporary inline Python
commands supplied the execution evidence; artifacts are under the ignored local
directory `results/research-audit-20261024/`.

Passing these checks establishes execution and selected numerical invariants.
It does not establish convergence, reward identifiability, paper reproduction,
or correctness of every retained experimental algorithm and configuration.

## Reading the implementation

| Component | Scientific interpretation and implementation contract |
| --- | --- |
| `algos/irl_losses.py` | Expert-positive binary classification; stable AIRL normalization; agent-only squared probability-gradient penalty. |
| `algos/airl.py` | Reward/potential decomposition `f=r+gamma*V(next)-V(state)`; inference returns the unshaped reward. |
| `algos/sac.py` | One-step soft Bellman targets and reparameterized actor objective with ordinary termination flags. `train()` historically returns targets, not TD errors. |
| `algos/sac_irl.py` | Three-valued DAC masks, absorbing-state bootstrapping, and physical-row actor averaging. |
| `policies/tfp_gaussian_actor.py` | Gaussian reparameterization, tanh Jacobian, normalized-action density convention, and latent entropy. |
| `networks/eqn.py` | Exact operator dictionary, relaxed domains, domain penalties, and discriminator-update-based sparsity stages. |
| `experiments/vector_on_policy_trainer.py` | GAE recursion with separate bootstrap and episode-boundary masks. |
| `algos/ppo.py` | Clipped likelihood-ratio surrogate and finite-sample KL stopping diagnostic. |

The corresponding docstrings state equations, shape assumptions, and paper
references. The implementation remains the authority for reproducing this
repository's experiments; paper formulas must not silently replace its choices.

## Scientific qualifications to retain in experiments

**Policy density convention.** With latent Gaussian action `u` and executed action
`a=c*tanh(u)`, the code reports
`log N(u;mu,sigma)-sum(log(1-tanh(u)^2+epsilon))`. The constant
`sum(log(abs(c)))` is omitted. `compute_log_probs` first divides out `c`, so it
uses the same convention. This affects absolute entropy and AIRL density values;
likelihood ratios with a fixed scale cancel the constant. The analytic
`compute_entropy` method is the entropy of the latent Gaussian, before either
transformation. Clipping the inverse tanh at saturated actions is approximate.

**AIRL sampling.** Canonical AIRL uses the current generator's density in the
discriminator denominator. Here expert log densities come from demonstrations
and agent densities from replay. Warmup even stores a fresh sampled-action log
density rather than the density of the random executed action. These historical
choices are documented, not replaced. Deterministic expert export records the
stochastic actor's density at its mean action, not a density of the deterministic
collection process. Avoid claiming canonical AIRL's theoretical guarantees for
this off-policy approximation. [Fu et al.](https://arxiv.org/abs/1710.11248)

**Reward and termination conventions.** Expert label one is used throughout the
active adversarial losses. Dense GAIL supplies `log(D+epsilon)` to the generator;
EQN-GAIL supplies stabilized log odds. Neither reward is interchangeable with
the other. In absorbing mode, `Q_abs=r_abs/(1-gamma)` explains why terminal rows
bootstrap into the learned absorbing state. Time limits instead bootstrap the
last physical observation. Synthetic self-loops do not contribute actor samples
or environment-step counts; the historical temperature update remains unmasked.
[Ho & Ermon](https://arxiv.org/abs/1606.03476),
[Kostrikov et al.](https://arxiv.org/abs/1809.02925)

**Equation interpretation.** The operator called `mult` is `(a+b)^2`. Logarithms
and division use the executable 0.1 relaxation; the domain penalty uses 0.001.
These thresholds and learned coefficients depend on input scaling. A sparse
formula is not, by itself, evidence for a dimensionally consistent physical law
or extrapolation outside the visited state-action distribution. The related
EQL literature motivates interpretable operator networks but does not certify
this repository's modified dictionary.
[Sahoo et al.](https://proceedings.mlr.press/v80/sahoo18a.html)

**On-policy targets.** GAE uses a termination mask for the next-state value and
an episode-boundary mask for recursion. Truncation therefore bootstraps but does
not leak advantages into the reset episode. The value target is discounted
reward-to-go, not `advantage+value` when lambda is below one. PPO's sampled KL can
be negative, and clipping is not a hard trust-region constraint.
[Schulman et al., GAE](https://arxiv.org/abs/1506.02438),
[Schulman et al., PPO](https://arxiv.org/abs/1707.06347)

**Twin critics and replay.** Existing initializer reuse can initialize twin
critics identically; this review preserves the established RNG/initialization
contract. Uniform one-step replay is the main validated path. The generic
n-step replay option does not establish matching n-step Bellman targets in
every algorithm. SAC now rejects non-unit importance weights rather than
silently ignoring them. DDPG applies them to its critic loss. These are separate
from validating prioritized-replay learning curves.
[Haarnoja et al.](https://arxiv.org/abs/1812.05905)

## Executed validation

Host runtime: Python 3.10, TensorFlow 2.18.1, TFP 0.25.0, Gymnasium 1.3.0,
NumPy 2.0.0, using `.venv-modern`. The existing container's isolated
`/tmp/tf2rl-mjwarp` environment supplies TensorFlow 2.21.0, TFP 0.25.0,
tf-keras 2.21.0, NumPy 2.4.2, MuJoCo/MJWarp 3.16.0, and Warp 1.18.0.
The container stack was not changed; successful runs on it do not change the
package's declared TensorFlow compatibility range.

| Check | Result and scope |
| --- | --- |
| Dependency consistency | Host `pip check` passed. |
| Syntax and undefined names | Compilation and Ruff `E9,F63,F7,F82` passed for `tf2rl`, `examples`, and `scripts`. This is not a full style-lint claim. |
| Runtime imports | 94 modules imported; five require optional `skimage`, `dm_control`, `mujoco_warp`, or `warp` missing from the host environment. |
| Existing example smoke script | CLI help and short SAC demonstration generation passed. |
| Existing three-stage pipeline | SAC, GAIL, and EQN-GAIL each ran 300 Pendulum steps; all 46 recorded metric series were finite and checkpoint inspection passed. |
| Imitation matrix | Dense AIRL/GAIL and EQN-AIRL/EQN-GAIL each ran 60 steps with explicit delay 3; EQN paths exercised staged sparsity/export and GAIL paths enabled gradient penalty. |
| Shared on-policy launcher | PPO and VPG each completed two 32-transition iterations on both Pendulum and CartPole, including short time limits and evaluation. |
| Shared off-policy launcher | SAC, TD3, and DDPG each ran 64 transitions across two Pendulum worlds, with evaluation, checkpoints, and two exported expert episodes. |
| Retained single-environment trainer | PPO and VPG each completed an eight-transition rollout after the VPG interface repair. |
| Checkpoint reload | SAC/TD3/DDPG restored and evaluated through the CLI; PPO/VPG checkpoints on Pendulum/CartPole passed `assert_consumed()` and finite inference. Retained VPG evaluation returned without creating replay or training. |
| Registered custom environments | All 26 repository-defined task IDs reset and completed three zero-action steps with finite observations/rewards. This does not verify full trajectories or physics equivalence. |
| Equation audit | Existing `audit_equation_learning.py` passed staged pruning, checkpoint continuation, finite gradient penalty, DAC boundaries, constant export, and polynomial fitting. |
| Equation export accuracy | Maximum absolute model/NumPy errors: EQN-GAIL `5.96e-8`, state-action EQN-AIRL `1.12e-8`, state-only EQN-AIRL `1.49e-8`; restricted-polynomial holdout error `9.54e-7`. |
| Direct numerical checks | Stable AIRL at scores ±1000; GAIL class means; analytic gradient penalty; all three DAC masks; terminal/truncated rollout recursion; update-delay scheduling; weighted DDPG loss; SAC weight rejection; all-absorbing actor/Adam invariance; CNN forward pass. |
| GPU/container | GPU SAC on Pendulum and GPU PPO with HalfCheetah MJWarp completed. CPU SAC with InvertedPendulum MJWarp completed 64 physical transitions, 56 updates, and 11 absorbing self-loops, plus expert export. |
| Package build | Source distribution and wheel built with `python -m build --no-isolation`. |

The pipeline returns (approximately -1109 SAC, -1244 GAIL, -1063 EQN-GAIL) are
short-run diagnostics, not evidence of solved Pendulum or an algorithm ranking.
The matrix is small and single-seed; it cannot substitute for full-budget,
multi-seed learning curves with uncertainty estimates.

The Hopper MJWarp probe correctly raised `NotImplementedError`: the shipped
task adapters cover HalfCheetah-v5 and InvertedPendulum-v5. Hopper would need a
task adapter; no new backend or task implementation was added during this review.

Retained VAIL/WAIL/GAIfO variants, distributed Ape-X, CURL, model-based trainers,
Atari/DMC and pixel-policy paths, rendering, generic n-step replay, and legacy
archives have not received complete execution/learning validation. In particular,
old optional paths still contain removed NumPy aliases and unsupported operations.
Use the active launchers and the scope above when describing validated results.

## Repeating the existing checks

```bash
python -m compileall -q tf2rl examples scripts
ruff check tf2rl examples scripts --select E9,F63,F7,F82
bash scripts/smoke_examples.sh
python scripts/run_validation_pipeline.py --steps 300 --warmup 32 \
  --interval 300 --units 32,32 --output results/research-audit-repeat/pipeline
python scripts/audit_equation_learning.py --output results/research-audit-repeat/equations
python -m build --no-isolation
```

Activate `.venv-modern` first. Individual runs retain their arguments and metrics;
the review's CLI matrix and container commands are also recorded in their local
log files. No executable test suite was introduced.
