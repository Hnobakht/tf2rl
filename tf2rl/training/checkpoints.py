"""Small checkpoint helpers preserving the existing TensorFlow object paths."""

from pathlib import Path

import tensorflow as tf


def checkpoint_prefix(directory):
    """Find managed checkpoints or the final prefix used by the experiment entry points.

    Args:
        directory (str or Path): Checkpoint prefix or run/checkpoint directory to search for saved
            policy state.

    Returns:
        str: TensorFlow checkpoint prefix, accepting either a prefix directly or the newest
            checkpoint in a directory.
    """
    directory = Path(directory)
    if Path(str(directory) + ".index").exists():
        return str(directory)
    prefix = tf.train.latest_checkpoint(str(directory / "checkpoints"))
    prefix = prefix or tf.train.latest_checkpoint(str(directory))
    if prefix is None and (directory / "checkpoint.index").exists():
        prefix = str(directory / "checkpoint")
    if prefix is None:
        raise ValueError(f"No policy checkpoint in {directory}")
    return prefix


def policy_checkpoint(policy, directory, max_to_keep=10):
    """Return the policy checkpoint and manager, retaining the historical root key.

    Args:
        policy (Policy): Agent supplying action selection and learning operations.
        directory (str or Path): Directory containing or receiving checkpoint files.
        max_to_keep (int): Maximum number of managed checkpoints retained on disk.

    Returns:
        tuple[tf.train.Checkpoint, tf.train.CheckpointManager]: Trackable policy checkpoint and its
            manager for saving and restoration.
    """
    checkpoint = tf.train.Checkpoint(policy=policy)
    manager = tf.train.CheckpointManager(checkpoint, str(directory), max_to_keep=max_to_keep)
    return checkpoint, manager


def restore_discriminator(trainer, irl, directory):
    """Restore the discriminator after checking the equation sparsity mode.

    Args:
        trainer (ImitationTrainer): Trainer owning the initialized discriminator checkpoint object
            to restore.
        irl (RewardLearner): Constructed learner whose equation sparsity mode must match the saved
            checkpoint.
        directory (str or Path): Directory containing a managed discriminator checkpoint.

    Returns:
        None.

    Raises:
        ConfigurationError: No discriminator checkpoint in --irl-model-dir. Checkpoint sparsity mode
            differs; restore with matching --eqn-l1/--eqn-stages options.
    """
    from tf2rl.experiments.config import ConfigurationError

    checkpoint = tf.train.latest_checkpoint(directory)
    if checkpoint is None:
        raise ConfigurationError("No discriminator checkpoint in --irl-model-dir")
    equation_model = getattr(irl, "equation_model", None)
    saved_sparse = any(
        "/sparsity_updates/" in name for name, _ in tf.train.list_variables(checkpoint)
    )
    if equation_model is not None and equation_model.sparse != saved_sparse:
        raise ConfigurationError(
            "Checkpoint sparsity mode differs; restore with matching --eqn-l1/--eqn-stages options"
        )
    status = trainer._checkpoint_irl.restore(checkpoint)
    status.assert_existing_objects_matched()
