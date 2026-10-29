"""Compose research experiments with Hydra and launch each in a fresh process.

The shared train entry point owns argument validation and experiment assembly.
Keeping TensorFlow in the child also isolates device initialization and RNG state
between jobs in Hydra's sequential multirun launcher.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

_RUNNERS = ("imitation", "on_policy", "off_policy")
_LOGGER = logging.getLogger(__name__)


def cli_arguments(options):
    """Translate YAML values to the existing CLI's option syntax.

    Booleans mean flag presence (including historical ``*-disable`` flags).
    Null leaves a CLI default untouched. Lists become nargs values, dictionaries
    become JSON, and scalar strings retain commas and spaces without shell parsing.

    Args:
        options (dict): Runner argument names mapped to Hydra scalar or sequence values.

    Returns:
        list[str]: CLI tokens: True emits a flag, False/None omit it, lists become separate nargs
            values, dictionaries become JSON, and scalar options use --key=value syntax.
    """
    arguments = []
    for name, value in options.items():
        option = "--" + name.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            arguments.append(option)
        elif isinstance(value, list):
            arguments.append(option)
            arguments.extend(str(item) for item in value)
        else:
            encoded = json.dumps(value) if isinstance(value, dict) else str(value)
            arguments.append(f"{option}={encoded}")
    return arguments


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(config: DictConfig):
    """Save the resolved configuration and run the selected experiment.

    Args:
        config (omegaconf.DictConfig): Composed Hydra job configuration with runner and argument
            values.

    Returns:
        None.
    """
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    runner = resolved["runner"]
    if runner not in _RUNNERS:
        raise ValueError(f"Unknown runner {runner!r}; choose one of {tuple(_RUNNERS)}")
    command = [
        sys.executable,
        "-m",
        "tf2rl.experiments.train",
        "--runner",
        runner,
        *cli_arguments(resolved["args"]),
    ]
    output = Path(HydraConfig.get().runtime.output_dir)
    OmegaConf.save(OmegaConf.create(resolved), output / "resolved.yaml")
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    _LOGGER.info("Launching %s; records: %s", resolved["args"]["algorithm"], output)
    # Resolve data/checkpoint paths against the invocation directory even when a
    # user explicitly enables hydra.job.chdir. No shell evaluates the arguments.
    subprocess.run(command, cwd=get_original_cwd(), check=True)


if __name__ == "__main__":
    main()
