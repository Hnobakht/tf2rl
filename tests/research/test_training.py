from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import tensorflow as tf

from tf2rl.algos.airl import AIRL
from tf2rl.algos.airl_eql import AIRL_EQL
from tf2rl.algos.gail import GAIL
from tf2rl.algos.gail_eql import GAIL_EQL
from tf2rl.algos.sac_irl import SAC_IRL
from tf2rl.envs.absorbing_wrapper import AbsorbingWrapper
from tf2rl.experiments.airl_trainer_v2 import AIRLTrainerV2
from tf2rl.experiments.irl_trainer_v2 import IRLTrainerV2
from tf2rl.experiments.trainer import Trainer
from tf2rl.experiments.demonstrations import restore_latest_n_traj


def generator(shape=(4,), absorbing=True):
    return SAC_IRL(shape, 1, actor_units=(8, 8), critic_units=(8, 8), gpu=-1,
                    memory_capacity=64, batch_size=4, n_warmup=2,
                    auto_alpha=True, is_absorbed_state=absorbing)


@pytest.mark.parametrize('cls,state_only', [(AIRL, False), (AIRL, True),
                                           (AIRL_EQL, False), (AIRL_EQL, True),
                                           (GAIL, False), (GAIL_EQL, False)])
def test_discriminator_initialization_and_real_updates(cls, state_only):
    kwargs = dict(state_shape=(4,), action_dim=1, gpu=-1, batch_size=4)
    if cls in (AIRL, AIRL_EQL):
        kwargs['is_state_only' if cls is AIRL_EQL else 'state_only'] = state_only
    irl = cls(**kwargs)
    reward = irl.inference(np.zeros(4, np.float32), np.zeros(1, np.float32))
    assert reward.shape == (1, 1) and np.isfinite(reward).all()
    before = [weight.numpy().copy() for weight in irl.trainable_variables]
    states = np.ones((4, 4), np.float32)
    actions = np.ones((4, 1), np.float32) * .2
    kwargs = dict(agent_states=states, agent_acts=actions, expert_states=states * .5,
                  expert_acts=actions * -.5, itr=1)
    if irl.requires_log_probs:
        kwargs.update(agent_next_states=states + .1, expert_next_states=states * .5 + .2,
                      agent_logps=-np.ones((4, 1), np.float32), expert_logps=-np.ones((4, 1), np.float32)*.5)
    result = irl.train(**kwargs)
    assert all(np.isfinite(value).all() for value in result)
    assert any(not np.array_equal(old, new.numpy()) for old, new in zip(before, irl.trainable_variables))


def test_delayed_update_and_gradient_penalty():
    irl = GAIL_EQL((4,), 1, gpu=-1, update_delay=3, grad_penalty_coeff=.1)
    states, acts = np.ones((4, 4), np.float32), np.ones((4, 1), np.float32)
    before = [weight.numpy().copy() for weight in irl.disc.weights]
    assert irl.train(states, acts, states/2, acts/2, itr=1) is None
    assert all(np.array_equal(old, new.numpy()) for old, new in zip(before, irl.disc.weights))
    assert np.isfinite(irl.train(states, acts, states/2, acts/2, itr=3)[0])


@pytest.mark.parametrize('absorbing', [False, True])
def test_sac_target_initialization_real_update_td_error_and_greedy_action(absorbing):
    policy = generator(absorbing=absorbing)
    for online, target in ((policy.qf1, policy.qf1_target), (policy.qf2, policy.qf2_target)):
        assert online.weights
        for a, b in zip(online.weights, target.weights):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
    state = np.ones(4, np.float32)
    action1, logp1 = policy.get_action(state, test=True)
    action2, logp2 = policy.get_action(state, test=True)
    np.testing.assert_array_equal(action1, action2)
    np.testing.assert_allclose(logp1, policy.get_logp(state, action1), atol=1e-5)
    states, acts = np.ones((4, 4), np.float32), np.ones((4, 1), np.float32)*.2
    masks = np.array([[1], [0], [-1], [1]], np.float32)
    rewards = np.ones((4, 1), np.float32)
    old = [weight.numpy().copy() for weight in policy.qf1.weights]
    target = policy.train(states, acts, states+.1, rewards, masks)
    assert np.isfinite(target).all()
    assert any(not np.array_equal(a, b.numpy()) for a, b in zip(old, policy.qf1.weights))
    td = policy.compute_td_error(states, acts, states+.1, rewards, masks)
    assert td.shape == (4,) and np.isfinite(td).all()
    if absorbing:
        old_actor = [weight.numpy().copy() for weight in policy.actor.weights]
        assert np.isfinite(policy.train(states, acts, states+.1, rewards, -np.ones((4, 1), np.float32))).all()
        assert all(np.array_equal(a, b.numpy()) for a, b in zip(old_actor, policy.actor.weights))


@pytest.mark.parametrize('algorithm', ['airl', 'gail', 'eqn-airl', 'eqn-gail'])
def test_export_import_train_evaluate_checkpoint_workflow(tmp_path, algorithm):
    # Export raw, ordered, complete demonstrations with explicit time-limit flags.
    raw = gym.make('Pendulum-v1', max_episode_steps=5)
    expert = generator(shape=(3,), absorbing=False)
    export = Trainer(expert, raw, dict(max_steps=2, test_episodes=2, episode_max_steps=5,
                                       save_test_path=True, logdir=str(tmp_path / 'expert')),
                     test_env=gym.make('Pendulum-v1', max_episode_steps=5))
    export.evaluate_policy(10)
    env = AbsorbingWrapper(gym.make('Pendulum-v1', max_episode_steps=5), .1)
    is_airl = 'airl' in algorithm
    data = restore_latest_n_traj(export._output_dir, env=env, is_logps=is_airl)
    assert len(data['obses']) == 10 and data['obses'].shape[1] == 4
    assert np.all(data['not_dones'] == 1)  # all paths ended by truncation
    classes = {'airl': AIRL, 'gail': GAIL, 'eqn-airl': AIRL_EQL, 'eqn-gail': GAIL_EQL}
    irl = classes[algorithm]((4,), 1, gpu=-1, batch_size=4)
    policy = generator()
    args = dict(max_steps=6, test_interval=3, test_episodes=1, episode_max_steps=5,
                save_model_interval=6, logdir=str(tmp_path / algorithm))
    kwargs = dict(test_env=AbsorbingWrapper(gym.make('Pendulum-v1', max_episode_steps=5), .1),
                  is_episodic=True, discriminator_noise=.25)
    if is_airl:
        trainer = AIRLTrainerV2(policy, env, args, irl, data['obses'], data['next_obses'],
                                data['acts'], data['logps'], **kwargs)
    else:
        trainer = IRLTrainerV2(policy, env, args, irl, data['obses'], data['next_obses'], data['acts'], **kwargs)
    replay = trainer()
    assert replay.get_stored_size() == 6
    before = policy.get_action(np.ones(4, np.float32), test=True)[0]
    reward_before = irl.inference(np.ones(4, np.float32), np.ones(1, np.float32)).numpy()
    saved = trainer.checkpoint_manager.latest_checkpoint
    irl_saved = trainer.checkpoint_manager_irl.latest_checkpoint
    clone = generator()
    tf.train.Checkpoint(policy=clone).restore(saved).assert_existing_objects_matched()
    np.testing.assert_allclose(clone.get_action(np.ones(4, np.float32), test=True)[0], before)
    irl_clone = classes[algorithm]((4,), 1, gpu=-1, batch_size=4)
    tf.train.Checkpoint(irl=irl_clone).restore(irl_saved).assert_existing_objects_matched()
    np.testing.assert_allclose(irl_clone.inference(np.ones(4, np.float32), np.ones(1, np.float32)), reward_before)
    export.close()
    trainer.close()
