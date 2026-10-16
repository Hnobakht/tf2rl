# Historical Python implementations

These source files were moved out of the runtime package during the 2026-10-16
cleanup. They preserve different research objectives, old interfaces and experiment
recipes; they are not installed or imported by the current training CLI.
`manifest.json` records each original path and SHA-256 hash. The archived source is
unchanged and may require the historical dependencies and surrounding source tree.

The supported implementations are:

| Historical implementation | Current implementation |
| --- | --- |
| SAC version 1 | `tf2rl/algos/sac.py` |
| SAC-IRL version 1 | `tf2rl/algos/sac_irl.py` |
| GAIL version 1 | `tf2rl/algos/gail.py` |
| EQN-AIRL version 1 | `tf2rl/algos/airl_eql.py` and `tf2rl/networks/eqn.py` |
| Versioned off-policy IRL/AIRL trainers | `tf2rl/experiments/irl_trainer.py` and `airl_trainer.py` |
| Duplicate replay helper | `tf2rl/misc/get_replay_buffer.py` |

This is not a claim that the historical versions have identical mathematics or
checkpoint formats. In particular, `_layer.py` has different negative-input log/div
branches, older SAC versions have different critic arrangements, and reversed AIRL
is a distinct experiment. They have not been silently substituted into the current
EQN objectives. Old GAIfO/VAIL variants and experiment launchers are also preserved
here; their retained counterparts in `tf2rl` remain experimental.

The current v2 trainer import paths are small aliases to the canonical trainers,
so existing scripts using those names still reach the same implementation. The
old version-1 DDPG imitation trainers are archived rather than presented as DAC
training implementations.
