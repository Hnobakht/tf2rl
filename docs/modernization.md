# Scientific contract and modernization details

The reference implementation is the supplied working tree, including its uncommitted
research files, rather than the older Git HEAD. The attached paper was read as
scientific context. Its executable-code differences are recorded rather than resolved
by silently changing equations. The original source is backed up locally under
`.work/original-working-tree.tar.gz`; the upstream README is retained separately.

## Architecture

`tf2rl.networks.eqn` contains the shared equation layer, scalar output and network.
`AIRL_EQL` specializes the common AIRL implementation with an EQN reward and the
original dense potential. `GAIL_EQL` specializes the common GAIL implementation
with an EQN discriminator and log-odds reward. Historical class/import names remain
available, with `EQNAIRL` and `EQNGAIL` aliases.

`irl_losses.py` names the AIRL/GAIL objectives and agent-only gradient penalty.
`IRLTrainerV2` handles DAC collection, replay, expert/noise sampling, discriminator
updates and evaluation for both algorithms. `AIRLTrainerV2` supplies the stored
expert log probabilities to that shared pipeline. The generator is the custom
`SAC_IRL`, retaining its separate conventional and absorbing targets. Expert SAC
retains its distinct critic LayerNormalization architecture.

`demonstrations.py` handles trajectory I/O independently from rendering.
`expert_sampling.py` preserves sampling without replacement, integer noise counts
and periodic warmup-distribution substitution. Active AIRL/GAIL examples share a
CLI and create distinct training/evaluation environments.

## Preserved equations and conventions

- EQN unary operators are identity, sine, cosine, sigmoid and
  `log(max(z, 0) + 0.1)`. Binary `mult(a,b)` is **`(a+b)^2`**;
  `div(a,b)` is `(a+0.1)/(max(b,0)+0.1)`. Operator ordering and affine widths
  follow the original v2 layer, including exclusions and repeated groups.
- Singular-unit loss is the batch average of `max(0.001-z, 0)` for log inputs
  and division denominators, summed across layers. The active objectives add it
  with coefficient one. The old v2 EQN layers do not apply their L1/mask arguments;
  this is retained. State-only AIRL's output retains its separate L1 behavior.
- AIRL shaping is `f=r+discount*V(next_state)-V(state)`. Its objective is the
  original expert/agent log-sum-exp classification loss. Policy inference returns
  the raw learned reward, not the shaped discriminator reward.
- Dense GAIL policy reward is `log(D+1e-8)`; EQN-GAIL reward is
  `log(D+1e-8)-log(1-D+1e-8)`. Discriminator loss is the negative expert log
  probability plus negative agent log complement. These conventions are not merged.
- Gradient penalty is the mean squared input-gradient norm of discriminator
  probability on agent inputs only. It is zero-centered, without interpolation.
- Keras resets dynamic model losses at each forward call. The old loss uses the
  last call's singular penalties (the additional agent call when GP is enabled).
  The refactor intentionally retains this behavior; it does not add expert and
  agent penalties together. EQN dropout is enabled for discriminator training and
  disabled for inference.
- Policy updates precede discriminator updates. Delayed discriminator updates and
  the existing optimizer choices remain. Expert/noise arrays stay aligned.
- Replay masks are continuation=1, terminal=0, absorbing=-1. Absorbing targets
  bootstrap at zero action and suppress next-state policy entropy on terminal and
  absorbing rows. Actor loss excludes absorbing self-loops; alpha loss retains
  the original unmasked mean. Conventional targets clamp continuation masks at zero.
- The actor's historical log-probability scale convention is retained: tanh
  correction is applied, but the constant action-scale log Jacobian is omitted.
  `get_logp(obs)` retains the fresh-action warmup sample, which need not be the
  log density of the random environment action. `get_logp(obs, actions)` explicitly
  evaluates supplied actions without changing the historical training default.

## Paper/code differences retained

The paper's displayed log/division definitions differ from the relaxed executable
operators above, especially for nonpositive inputs and the division numerator.
The paper describes the singular threshold using its relaxation constant; the
code uses 0.001, while operator relaxation is 0.1. The paper's discriminator labels
and presentation should not be used to flip the working code's expert/agent labels.
The default GP coefficient in the supplied EQN-GAIL constructor is zero, even
though the paper discusses nonzero GP. No paper formula is silently substituted.

## Behavior repairs that are not merely structural

These fixes affect broken or ambiguous executions and can change numerical results:

1. Models are built before target synchronization and before gradient selection.
   The old SAC code copied empty weight lists in the reversed direction; copying
   built online critics into targets now initializes the intended SAC targets.
   Initializer objects are reused across the twin critics and targets as before;
   their construction now follows the CLI seed. Initialization/RNG sequences can
   still differ across Keras versions.
2. State-only EQN-AIRL no longer fails because of the unsupported `l1_regularizers`
   layer keyword. Keras 3 variable tracking and public API replacements are applied.
3. All-absorbing actor batches have zero actor objective and skip the optimizer
   update instead of dividing by zero. Alpha updates retain the existing rule.
   SAC TD errors use the twin target critics instead of a removed value network.
4. Explicit Gymnasium termination controls replay masks. A true terminal exactly
   at the horizon, including simultaneous flags, is terminal. The old length-only
   rule mistakenly treated it as continuing. Pure truncation still bootstraps.
5. Expert and online augmentation share the same absorbing-mode setting. The old
   loader inserted absorbing expert rows even when collection was disabled.
   New exports contain explicit boundary metadata; old files cannot resolve a
   genuine terminal at the horizon without additional information.
6. Short demonstrations are accepted; complete trajectories are exported in order,
   without sampling or padding a fixed horizon. `max_steps` is honored before
   augmentation. Already augmented observations are not augmented twice. File and
   dataset ordering may therefore differ from the old stacking implementation.
7. Evaluation no longer advances or resets the training environment. `test=True`
   now actually reaches SAC_IRL's actor and selects the mean; historically it was
   passed as the actor's training argument and evaluation remained stochastic.
   Training remains stochastic, including the historical AIRL collector behavior.
8. The CLI honors `--memory-capacity`, `--gp-coef`, and `--n-training`; previously
   some examples ignored them. Default replay capacity remains the actual old
   constructor default of 2,000,000. Default GP=0 and discriminator updates=1 remain.
   Historical configs containing ignored flags must be audited before rerunning.
9. The retained on-policy trainer uses explicit termination and bootstraps truncated
   horizons from the next-state value. This corrects its previous zero-bootstrap
   boundary behavior and is separate from the EQN off-policy objective.
10. Custom MuJoCo tasks use native MuJoCo/Gymnasium rather than mujoco-py. Original
    XML geometry, rewards, reset-distribution formulas and observation order remain.
    The eight legacy global-coordinate Hopper/Walker assets are converted in memory
    to local coordinates by subtracting parent/body origins. Their body frames have
    no rotations; rotated frames are rejected. Tests compare compiled body/geom world
    positions to the original XML coordinates. RNG `randn` becomes `standard_normal`, and observation width is inferred without
    the old constructor's random step. Physics versions and seeded RNG sequences
    cannot be assumed equivalent. Time counters in maze/rllab tasks are truncations.
11. Keras 3 list tracking does not provide TensorFlow checkpoint paths for every
    nested layer. EQN layers, SAC actors and critic hidden layers now also have
    explicit attributes. Full checkpoint tests compare online and target weights
    and require optimizer state to be consumed. Checkpoints from the earlier
    incomplete modernization may have omitted target weights and should be regenerated.
12. State-only EQN-AIRL output L1 is expressed as the same TensorFlow absolute-value
    sum inside `call`; this avoids Keras 3 integer-coefficient/tracing failures.
    Its nonzero coefficient value and gradients are tested analytically.
13. Summary writers are scoped to each trainer invocation and restore the caller's
    writer/step. Metrics flush after evaluations for live TensorBoard viewing.
    The TensorBoard 2.18 runtime requires setuptools below 82 because it imports
    the removed `pkg_resources` API. This bound is a dependency repair.
14. Retained D2RL-SAC construction now supplies twin Q targets and a joint critic
    optimizer required by its current SAC parent. Its unreachable V-network setup
    and broken `CriticV` import were removed. This changes that legacy extension's
    wiring; it does not enter the EQN/GAIL training paths.

These repairs are explicitly disclosed; full learning-curve equivalence is not claimed.
Unsupported research schedules and incompatible DAC replay modes raise descriptive
errors instead of silently ignoring flags or corrupting masks. The DAC path requires
uniform, one-step replay and the custom SAC_IRL generator.

## Dependencies, files and compatibility

Package metadata is declarative in `pyproject.toml`; `setup.py` is a compatibility
shim and no longer imports TensorFlow during installation. The TensorFlow 2.18 /
TFP 0.25 / tf-keras 2.18 family follows upstream compatibility notes. Gymnasium,
NumPy, cpprb, SciPy and joblib are bounded. Rendering, test and MuJoCo dependencies
are optional. This targeted runtime family has not been fully installed/tested locally.

TensorFlow Addons is replaced by a project-local wrapper with its historical
power-iteration/assignment behavior. Private TensorFlow APIs in SNDense are replaced
with public tensor operations. Importing tf2rl no longer configures GPU memory growth
or eagerly imports MuJoCo tasks. Python 2 compatibility code is removed from output
setup; reproducibility environment logging records only relevant runtime settings.

- `gym/` moved to `legacy/gym/`, retaining the fork and license for provenance.
  Its runtime XML files are copied unchanged into packaged `tf2rl/envs/assets/`.
- `tf2rl/algos/airl_old_v1.py` was an exact byte-for-byte duplicate at baseline and
  had no incoming references; it was removed. Distinct historical variants remain.
- `.travis.yml` was superseded by modern GitHub Actions validation.
- The active examples' duplicate orchestration is replaced by CLI forwarding.
- Existing `examples/old` moves/deletions were already present before this task.
- No experiment dataset or result was deleted. Old plotting paths pointing outside
  the repository are replaced with local `results/` CSV paths.
- Old CI jobs targeting Python 3.7 / TensorFlow 2.2 were replaced with compatible
  builds and research checks; release jobs produce artifacts without automatic
  upstream package publication.

Historical version-1 trainers, Atari/DMC integrations and experimental VAIL/WAIL,
GAIfO/reversed-AIRL paths remain source artifacts and are not all certified on this
stack. The modern four AIRL/GAIL paths use the unified version-2 trainer. External
reward wrapper sign conventions were not silently changed; the new native
`LearnedRewardWrapper` accepts an explicitly restored IRL object.

Current checkpoints contain policy/discriminator objects and optimizer state. Replay,
environment and RNG state are not saved, so continuation is not bitwise replayable.
Older checkpoints may require explicit weight mapping. No original demonstrations,
complete experiment configs or result artifacts were available for paper replication.
