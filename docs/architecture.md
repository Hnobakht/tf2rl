# Research framework architecture

Start at `tf2rl/experiments/train.py`. It validates options, constructs environments
and algorithms through ordinary factory functions, calls `trainer.run()`, and
closes resources. It contains no training loop.

```text
tf2rl/
├── algos/
│   ├── base.py          OffPolicyAgent, OnPolicyAgent, RewardLearner
│   ├── rl/              SAC, TD3, DDPG, PPO, VPG, DQN and research variants
│   └── imitation/       AIRL, GAIL, GAIfO, EQN and variational variants
├── networks/            Reusable actors, critics, layers, CNNs, VAE and EQN
├── envs/
│   ├── factory.py       Environment construction
│   ├── vector.py        Gymnasium/tensor vector adapter
│   ├── mjwarp.py        Optional batched GPU simulation
│   ├── wrappers/        Observation, reward, pixel and Atari wrappers
│   ├── tasks/           Research MuJoCo tasks
│   ├── dynamic_mjc/     Existing model builders
│   └── assets/          Original XML assets
├── data/                Replay, demonstrations, expert sampling, DAC conversion
├── training/            Loops, collectors, evaluation, checkpoint helpers
├── utils/               TensorFlow, returns, normalization, logging and images
├── experiments/         Configuration, component construction and launchers
└── experimental/        Older model-based, distributed and on-policy IRL paths
```

## Where to put a breakpoint

| Path | Responsibility |
| --- | --- |
| `training/off_policy.py:OffPolicyTrainer.run` | Shared scalar loop, used by expert SAC and imitation |
| `training/off_policy.py:OffPolicyTrainer._run_vector` | Vector SAC/TD3/DDPG collection/update loop and reports |
| `training/on_policy.py:OnPolicyTrainer.run` | PPO/VPG rollout and minibatch loop |
| `training/imitation.py:ImitationTrainer.update` | Learned rewards, generator update, discriminator updates |
| `training/collectors.py` | Action selection, environment stepping, episode boundaries |
| `algos/rl/` or `algos/imitation/` | Gradient calculations and algorithm-specific update rules |

There are three active trainers. `ImitationTrainer` inherits `OffPolicyTrainer.run`
and overrides replay setup, `update()`, and saving. The scalar and vector recipes
retain their different historical warmup and update budgets; they are not silently
made numerically equivalent. `collect_step()` and calling a trainer as a function
remain compatible spellings.

## Ownership

Agents own their networks, optimizers, and algorithm-specific state. On-policy
agents expose tensor action/value inference, minibatch training, and an optional
stopping decision. Off-policy agents expose action selection and replay training.
`RewardLearner` is an independent Keras model base: it learns from expert and agent
batches and supplies `inference()` rewards. It has no action-selection interface.
The historical name `IRLPolicy` aliases `RewardLearner`.

Collectors own current observations and episode statistics. Trainers own replay,
collection/update scheduling, TensorBoard writers and checkpoint calls. Evaluation
uses independent environments and lives in `training/evaluation.py`. Demonstration
serialization lives in `data/demonstrations.py`, with no dependency on a trainer.
Algorithms continue to write their own detailed TensorFlow summaries.

Batch dictionaries keep the existing keys (`obs`, `act`, `next_obs`, `rew`, `done`,
and optional `logp`, `weights`, `indexes`). In replay, `done` means either a true
termination flag or a three-valued DAC mask, as specified by the agent's
`replay_mask_encoding`. It is not safe to infer its meaning from the key name.
Physical exported trajectories retain separate `terminated` and `truncated`
fields. Collectors use final physical observations before autoreset. DAC-generated
self-loops increase replay rows but never physical step counts or vector update
credit. AIRL collection-time density conventions and EQN reward transformations
are preserved.

The environment backend and learner device are independent. Vector on-policy
rollouts stay as tensors; cpprb replay remains on the host. Heavy optional image
and simulator dependencies are imported only when those paths are used.

## Extending and compatibility

Add an algorithm file and a factory or registration in `experiments/build.py`.
Add a YAML experiment preset when CLI access is useful. Construct custom components
directly from Python when a preset would add no value. See [adding algorithms](adding_algorithms.md).
Shared models belong in `networks/`; private models can remain next to their algorithm.
There is no callback system for learning updates or dependency-injection framework.

Old import locations contain forwarding modules; implementations live in the
packages above. Existing console commands and example filenames remain available.
Hyphenated Hydra preset names remain aliases for underscore names. The old
`conf/runner` templates remain available for existing external custom presets. Legacy CLI
argument-builder methods remain for existing Python callers. Configuration is
resolved and validated in `experiments/config.py`; no large configuration class
hierarchy is introduced.

Moving models does not change their TensorFlow checkpoint root keys or attribute
paths. Restoring model/optimizer state still does not restore replay, an active
episode or RNG streams. The older trainers under `experimental/` retain their
previous validation limits; relocation is not a correctness claim.
