"""Small setup helpers shared by experiment launchers, separate from training."""

import json
from pathlib import Path

import tensorflow as tf


class ConfigurationError(ValueError):
    """An invalid experiment setting that a CLI can report without a traceback."""


def configure_tensorflow(seed, cpu_threads, cpu_inter_threads=1):
    """Configure a fresh vector-training process before constructing any tensors."""
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(cpu_inter_threads)
    tf.keras.utils.set_random_seed(seed)


def save_run_config(args):
    """Save the effective launcher options alongside vector-training artifacts."""
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return config
