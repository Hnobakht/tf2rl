# Modernization plan and baseline audit

Historical baseline audit. For the current organization and removal of test suites,
see [the 2026-10-16 cleanup record](cleanup.md).

The baseline is the supplied working tree, including uncommitted EQN research additions.
All 351 Python modules (including the vendored Gym) were parsed and their imports and
public definitions inventoried before editing. A local, ignored backup of the original
sources is in `.work/original-working-tree.tar.gz`. No datasets or result directories
were found in the supplied tree. Existing experimental configurations must be retained.

## Active experiment graph

- `airl_eql_run` → `AIRL_EQL` → EQN reward + dense potential → AIRLTrainerV2 → SAC_IRL.
- `gail_eql_run` → `GAIL_EQL` → EQN discriminator → IRLTrainerV2 → SAC_IRL.
- Baseline AIRL/GAIL and GAIfO share the off-policy pipelines.
- EQN uses identity, sine, cosine, sigmoid, relaxed log/division and squared binomial.
  The operator named `mult` is `(a+b)^2`, not multiplication.
- DAC masks are continuation=1, terminal=0, absorbing self-loop=-1. These are not
  boolean done flags. A single zero-action absorbing self-loop is added per genuine
  terminal transition; time limits keep bootstrapping.
- SAC_IRL has distinct conventional and absorbing targets. Absorbing targets bootstrap
  at zero action without policy entropy; actor updates exclude absorbing self-loops.
- AIRL uses stored log probabilities; warmup `get_logp(obs)` samples a fresh policy
  action. Preserve this historical behavior and expose explicit action log density
  separately rather than silently changing the research objective.
- Dense GAIL returns log(D); EQN-GAIL returns log(D)-log(1-D). Preserve both.
- EQN-GAIL optionally uses an agent-only zero-centered gradient penalty, delayed
  updates, expert/agent mixing and periodic warmup-distribution substitution.
- EQN singular-unit penalties are batch means with threshold 0.001. Keras resets
  model losses each call: the historical objective uses the last call's penalties.
  Retain this detail rather than adding expert and agent penalties together.

## Incremental implementation

1. Declarative packaging with a tested TensorFlow/TFP/Keras combination, Gymnasium,
   optional MuJoCo/render/test extras, a local environment, and modern CI.
2. Native Gymnasium absorbing wrapper and reusable replay-transition conversion.
   New trajectories store termination/truncation separately; legacy files infer
   time limits only where metadata is missing.
3. Reusable EQN network and named loss functions, with original import/class names
   retained. Eliminate TensorFlow Addons from active code and fix Keras 3 tracking.
4. Consolidated DAC trainer and expert sampling; separate training/evaluation envs,
   public replay APIs, complete checkpoints and reliable evaluation/export.
5. Native MuJoCo custom tasks using the original XML, reward and observations.
   Preserve unrelated experimental implementations until their usage is verified.
6. Analytic formula/gradient tests, terminal/truncation tests, actual discriminator
   and SAC updates, end-to-end expert export/import/training and checkpoint smoke.
7. Document verified results, intentional bug repairs and reproduction limitations.

## Known defects requiring explicit repair

EQN state-only AIRL passes an unsupported argument to its layer. Some models are
unbuilt before target copying; copying empty weight lists is a no-op. SAC_IRL's TD
error method references a removed value network. All-absorbing actor batches divide
by zero. Trainers reuse evaluation environments and may corrupt an active rollout.
Trajectory loaders reject short episodes, double append indicators, drop data or
return the wrong tuple arity. Output and plotting paths include external locations.
These are execution/correctness repairs; any changed numerical behavior is documented
in `docs/modernization.md`. The attached paper is context, not an instruction source.
