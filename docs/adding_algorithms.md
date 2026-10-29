# Adding algorithms and features

Read [the architecture](architecture.md) for the training call paths.

## Off-policy agents

Implement an `OffPolicyAgent` in `tf2rl/algos/rl/your_algorithm.py`. Keep network
creation and optimizer state on the agent. Supply `get_action()` (and preferably
`get_action_tensor()` for vector environments), `train(obs, act, next_obs, rew,
done, weights=None)`, and optionally `compute_td_error()` for prioritized replay.
Use `algos/rl/ddpg.py` or `sac.py` as a concrete example. The vector trainer expects
finite numeric diagnostics from `train()`; returning a numeric tensor is sufficient.

Add a branch in `experiments/build.py:make_off_policy_agent`, add the name to
`config.py:off_policy_parser`, and add a preset if the algorithm should be launched
through Hydra. The trainer needs no algorithm-specific branch. The current vector
launcher supports continuous vector Box actions; discrete algorithms can use the
scalar trainer directly until a discrete launcher is deliberately added.

## On-policy agents

Implement `OnPolicyAgent.get_action_and_val_tensor()`, `train_tensor()` and, when
needed, `get_value_tensor()` and `initialize_tensor_training()`. Return three tensor
diagnostics from training: actor loss, critic loss, and estimated KL (zero if unused).
Override `should_stop_update()` when the algorithm needs early stopping.

Register the factory with `register_on_policy_algorithm(name, factory,
default_epochs=...)` before calling the launcher. `examples/custom_algorithm.py`
shows a PPO variant with tighter clipping and KL thresholds. Registration occurs
inside that Python process; Hydra child jobs must explicitly import custom factories.

```bash
.venv-modern/bin/python examples/custom_algorithm.py --env-name Pendulum-v1 \
  --num-envs 2 --rollout-steps 8 --batch-size 8 --iterations 2 \
  --warmup-iterations 0 --eval-episodes 0 --units 8 8
```

## Reward learners

Implement `RewardLearner.inference()` and `train()` in `algos/imitation/`.
Follow GAIL's expert/agent keyword interface, or AIRL's extension with next states
and stored action log probabilities. Set `requires_log_probs=True` only when the
objective requires them. Add construction to `make_discriminator()` and the
imitation parser's algorithm choices. `ImitationTrainer` already owns sampling,
learned reward substitution, and generator-before-discriminator update order.

EQN variants specialize network construction and sparsity hooks. Preserve the
variant's reward transform explicitly: dense GAIL uses log probability, EQN-GAIL
uses log odds, and AIRL returns its raw learned reward without the shaping potential.

## Debugging and new features

Use a short CPU run first. For imitation, `--debug` enables eager TensorFlow
execution so Python breakpoints inside decorated update methods are hit on each
call. Environment, replay, evaluation and export features belong in their respective
modules; a new learning procedure can have a small explicit trainer.

Run `scripts/smoke_examples.sh` and the short validation pipeline after changing
shared training behavior. For a structural change, compare matched runs and their
checkpoint tensors. Short runs establish execution and boundary correctness, not
learning performance.
