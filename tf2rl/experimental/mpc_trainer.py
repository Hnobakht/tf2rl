import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from cpprb import ReplayBuffer

from tf2rl.training.off_policy import OffPolicyTrainer as Trainer
from tf2rl.data.replay import get_space_size


class DynamicsModel(tf.keras.Model):
    """Neural model predicting environment dynamics from state-action pairs."""
    def __init__(self, input_dim, output_dim, units=[32, 32], name="DymamicsModel", gpu=0):
        """Build a state-action prediction network on the selected TensorFlow device.

        Args:
            input_dim (int): Number of input features in the dynamics model.
            output_dim (int): Number of predicted output features in the dynamics model.
            units (sequence[int]): Hidden-layer widths in forward order.
            name (str): Name assigned to the model, operation, or registered component.
            gpu (int): TensorFlow GPU index; a negative index selects the CPU.

        Returns:
            None: Initializes the instance.
        """
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation="relu")
        self.l2 = Dense(units[1], name="L2", activation="relu")
        self.l3 = Dense(output_dim, name="L3", activation="linear")

        with tf.device(self.device):
            self(tf.constant(np.zeros(shape=(1, input_dim), dtype=np.float32)))

    @tf.function
    def call(self, inputs):
        """Predict dynamics outputs for the supplied state-action batch.

        Args:
            inputs (array_like): Concatenated observation/action features, with an optional leading
                batch axis for predict.

        Returns:
            tf.Tensor: Predicted observation deltas shaped (batch, output_dim).
        """
        features = self.l1(inputs)
        features = self.l2(features)
        return self.l3(features)

    def predict(self, inputs):
        """Predict the dynamics model output for a batch of state-action inputs.

        Args:
            inputs (array_like): Concatenated observation/action features, with an optional leading
                batch axis for predict.

        Returns:
            np.ndarray: Predicted observation deltas; one feature vector returns one output vector,
                while batches retain their batch axis.
        """
        assert isinstance(inputs, np.ndarray)
        if inputs.ndim == 1:
            inputs = np.expand_dims(inputs, axis=0)

        with tf.device(self.device):
            outputs = self.call(inputs)

        if inputs.shape[0] == 1:
            return outputs.numpy()[0]
        else:
            return outputs.numpy()


class RandomPolicy:
    """Sample independent continuous actions uniformly within a symmetric bound."""
    def __init__(self, max_action, act_dim):
        """Store the continuous action bound and number of action coordinates.

        Args:
            max_action (float): Symmetric continuous-action bound or output scale.
            act_dim (int): Width of one continuous action vector.

        Returns:
            None: Initializes the instance.
        """
        self._max_action = max_action
        self._act_dim = act_dim
        self.policy_name = "RandomPolicy"

    def get_action(self, obs):
        """Sample one continuous action uniformly between minus and plus max_action.

        Args:
            obs (array_like): Current observation, or batch of observations with a leading batch axis.
                This implementation does not read this argument.

        Returns:
            np.ndarray: Uniformly sampled continuous action vector shaped (act_dim,).
        """
        return np.random.uniform(
            low=-self._max_action,
            high=self._max_action,
            size=self._act_dim)

    def get_actions(self, obses):
        """Sample one bounded continuous action vector per observation in the batch.

        Args:
            obses (array_like): Observation batch used for model prediction or random action selection.

        Returns:
            np.ndarray: Independent uniform action samples shaped (len(obses), act_dim).
        """
        batch_size = obses.shape[0]
        return np.random.uniform(
            low=-self._max_action,
            high=self._max_action,
            size=(batch_size, self._act_dim))


class MPCTrainer(Trainer):
    """Train dynamics models and collect data using sampled model-predictive control."""
    def __init__(
            self,
            policy,
            env,
            args,
            reward_fn,
            buffer_size=int(1e6),
            n_dynamics_model=1,
            lr=0.001,
            **kwargs):
        """Configure training, replay storage, evaluation, and checkpoint handling.

        Args:
            policy (Policy): Agent supplying action selection and learning operations.
            env (gym.Env or gym.vector.VectorEnv): Environment instance providing observations, action
                spaces, and interaction state.
            args (argparse.Namespace): Parsed experiment options used to configure this operation.
            reward_fn (callable): Function computing reward from states, actions, and predicted
                successors.
            buffer_size (int): Capacity of the local replay or dynamics buffer.
            n_dynamics_model (int): Number of models in the learned dynamics ensemble.
            lr (float): Optimizer learning rate.
            **kwargs (dict): Additional keyword arguments forwarded to Trainer construction.

        Returns:
            None: Initializes the instance.
        """
        super().__init__(policy, env, args, **kwargs)

        self.dynamics_buffer = ReplayBuffer(**self._prepare_dynamics_buffer_dict(buffer_size=buffer_size))
        self._n_dynamics_model = n_dynamics_model

        # Reward function
        self._reward_fn = reward_fn
        self._prepare_dynamics_model(gpu=args.gpu, lr=lr)

    def _prepare_dynamics_buffer_dict(self, buffer_size):
        # Prepare buffer that stores transitions (s, a, s')
        """Define the replay fields used to train the dynamics models.

        Args:
            buffer_size (int): Capacity of the local replay or dynamics buffer.

        Returns:
            dict: Replay schema for observation/action/successor triples used to fit the dynamics
                ensemble.
        """
        rb_dict = {
            "size": buffer_size,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {"shape": get_space_size(self._env.observation_space)},
                "next_obs": {"shape": get_space_size(self._env.observation_space)},
                "act": {"shape": get_space_size(self._env.action_space)}}}
        return rb_dict

    def _prepare_dynamics_model(self, gpu=0, lr=0.001):
        # Dynamics model
        """Create the dynamics models and their optimizers.

        Args:
            gpu (int): TensorFlow GPU index; a negative index selects the CPU.
            lr (float): Optimizer learning rate.

        Returns:
            None.
        """
        obs_dim = self._env.observation_space.high.size
        act_dim = self._env.action_space.high.size
        self._dynamics_models = [
            DynamicsModel(
                input_dim=obs_dim + act_dim,
                output_dim=obs_dim,
                gpu=gpu) for _ in range(self._n_dynamics_model)]
        self._optimizers = [
            tf.keras.optimizers.Adam(learning_rate=lr) for _ in range(self._n_dynamics_model)]

    def _set_check_point(self, model_dir):
        # Save and restore model
        """Create the TensorFlow checkpoint manager and restore a checkpoint when requested.

        Args:
            model_dir (str or Path or None): Directory containing policy weights to restore; None starts
                with current initialization.

        Returns:
            None.
        """
        if isinstance(self._policy, tf.keras.Model):
            super()._set_check_point(model_dir)

    def __call__(self):
        """Run the training loop, including data collection, policy updates, and evaluation.

        Returns:
            None.
        """
        total_steps = 0
        tf.summary.experimental.set_step(total_steps)
        # Gather dataset of random trajectories
        self.logger.info("Ramdomly collect {} samples...".format(self._n_random_rollout * self._episode_max_steps))
        self.collect_episodes(n_rollout=self._n_random_rollout)

        for i in range(self._max_iter):
            # Train dynamics f(s, a) according to eq.(2)
            mean_loss = self.fit_dynamics(n_epoch=1)

            total_rew = 0.

            # Collect new sample
            obs = self._env.reset()
            for _ in range(self._episode_max_steps):
                total_steps += 1
                act = self._mpc(obs)
                next_obs, rew, done, _ = self._env.step(act)
                self.dynamics_buffer.add(
                    obs=obs, act=act, next_obs=next_obs)
                total_rew += rew
                if done:
                    break
                obs = next_obs

            tf.summary.experimental.set_step(total_steps)
            tf.summary.scalar("mpc/total_rew", total_rew)
            self.logger.info("iter={0: 3d} total_rew: {1:4.4f} loss: {2:2.8f}".format(i, total_rew, mean_loss))

    def predict_next_state(self, obses, acts):
        """Predict successor states from observations and actions using the learned dynamics.

        Args:
            obses (array_like): Observation batch used for model prediction or random action selection.
            acts (array_like): Actions aligned with the observation batch.

        Returns:
            np.ndarray: Predicted next observations obtained by adding the learned dynamics delta to
                obses.
        """
        obs_diffs = np.zeros_like(obses)
        inputs = np.concatenate([obses, acts], axis=1)
        for dynamics_model in self._dynamics_models:
            obs_diffs += dynamics_model.predict(inputs)
        obs_diffs /= self._n_dynamics_model
        return obses + obs_diffs

    def _mpc(self, obs):
        """Choose the first action of the sampled sequence with the best predicted return.

        Args:
            obs (array_like): Current observation, or batch of observations with a leading batch axis.

        Returns:
            np.ndarray: First action from the candidate sequence with the highest accumulated model-
                predicted reward.
        """
        obses = np.tile(obs, (self._n_sample, 1))
        init_actions = self._policy.get_actions(obses)
        total_rewards = np.zeros(shape=(self._n_sample,))

        for i in range(self._horizon):
            if i == 0:
                acts = init_actions
            else:
                acts = self._policy.get_actions(obses)
            assert obses.shape[0] == acts.shape[0]
            next_obses = self.predict_next_state(obses, acts)
            rewards = self._reward_fn(obses, acts)
            assert rewards.shape == total_rewards.shape
            total_rewards += rewards
            obses = next_obses

        idx = np.argmax(total_rewards)
        return init_actions[idx]

    def _set_from_args(self, args):
        """Copy trainer settings from the parsed command-line arguments.

        Args:
            args (argparse.Namespace): Parsed experiment options used to configure this operation.

        Returns:
            None.
        """
        super()._set_from_args(args)
        self._max_iter = args.max_iter
        self._horizon = args.horizon
        self._n_sample = args.n_sample
        self._n_random_rollout = args.n_random_rollout
        self._batch_size = args.batch_size

    def collect_episodes(self, n_rollout=1):
        """Collect real-environment episodes for dynamics learning.

        Args:
            n_rollout (int): Number of complete episodes to collect.

        Returns:
            None.
        """
        for _ in range(n_rollout):
            obs = self._env.reset()
            for _ in range(self._episode_max_steps):
                act = self._policy.get_action(obs)
                next_obs, _, done, _ = self._env.step(act)
                self.dynamics_buffer.add(
                    obs=obs, act=act, next_obs=next_obs)
                obs = next_obs
                if done:
                    break

    @tf.function
    def _fit_dynamics_body(self, inputs, labels):
        """Update each dynamics model once and return the ensemble loss vector.

        Args:
            inputs (array_like): Dynamics-training input features, organized by ensemble member and
                sample.
            labels (array_like): Dynamics-model targets, usually successor-minus-current observation
                differences.

        Returns:
            tf.Tensor: One supervised dynamics loss per ensemble member, after applying their optimizer
                updates.
        """
        losses = []
        for dynamics_model, optimizer in zip(self._dynamics_models, self._optimizers):
            with tf.GradientTape() as tape:
                predicts = dynamics_model(inputs)
                loss = tf.reduce_mean(0.5 * tf.square(labels - predicts))
            grads = tape.gradient(
                loss, dynamics_model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, dynamics_model.trainable_variables))
            losses.append(loss)
        return tf.convert_to_tensor(losses)

    def _make_inputs_output_pairs(self, n_epoch):
        """Create dynamics inputs and state-difference targets from replay samples.

        Args:
            n_epoch (int): Number of learning epochs or dataset repetitions used by this operation. This
                implementation does not read this argument.

        Returns:
            tuple[np.ndarray, np.ndarray]: Concatenated state/action inputs and successor-minus-current
                observation labels, shuffled/repeated for ensemble fitting.
        """
        samples = self.dynamics_buffer.sample(self.dynamics_buffer.get_stored_size())
        inputs = np.concatenate([samples["obs"], samples["act"]], axis=1)
        labels = samples["next_obs"] - samples["obs"]

        return inputs, labels

    def fit_dynamics(self, n_epoch=1):
        """Train the dynamics ensemble on transitions from the replay buffer.

        Args:
            n_epoch (int): Number of learning epochs or dataset repetitions used by this operation.

        Returns:
            numpy scalar: Mean supervised loss over the dynamics fitting iterations.
        """
        inputs, labels = self._make_inputs_output_pairs(n_epoch)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat(n_epoch)

        mean_losses = np.zeros(shape=(self._n_dynamics_model,), dtype=np.float32)
        for batch, (x, y) in enumerate(dataset):
            _mean_losses = self._fit_dynamics_body(x, y)
            mean_losses += _mean_losses.numpy()
        mean_losses /= (batch + 1)

        for model_idx, mean_loss in enumerate(mean_losses):
            tf.summary.scalar("mpc/model_{}_loss".format(model_idx), mean_loss)
        return np.mean(mean_losses)

    @staticmethod
    def get_argument(parser=None):
        """Add this component's command-line options and return the argument parser.

        Args:
            parser (argparse.ArgumentParser or None): Parser to extend; None constructs a new parser.

        Returns:
            argparse.ArgumentParser: Parser extended with the supported component options.
        """
        parser = Trainer.get_argument(parser)
        parser.add_argument('--gpu', type=int, default=0,
                            help='GPU id')
        parser.add_argument("--max-iter", type=int, default=100)
        parser.add_argument("--horizon", type=int, default=20)
        parser.add_argument("--n-sample", type=int, default=1000)
        parser.add_argument("--n-random-rollout", type=int, default=1000)
        parser.add_argument("--batch-size", type=int, default=512)
        return parser
