"""
Utility functions for loading YAML configuration files used by the command-line
scripts.
"""

from pathlib import Path

import yaml


def load_config(filename):
    """
    Load a YAML configuration file and merge it with its common configuration.

    Parameters
    ----------
    filename : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Complete configuration dictionary used by the processing scripts.
    """

    filename = Path(filename)

    # Read the user configuration file.
    # This file contains the parameters specific to the current run.
    with open(filename) as f:
        config = yaml.safe_load(f)

    # Return already complete configurations unchanged.
    if "include" not in config:
        return config

    # Read the common configuration file referenced by the "include" key.
    # The included file contains common parameters shared by all runs.
    include_file = filename.parent / config["include"]

    with open(include_file) as f:
        base_config = yaml.safe_load(f)

    # Update the common configuration with values from the current run.
    # Parameters defined in the run configuration overwrite common values.
    # The "include" entry is only used to locate the common file and is not
    # kept in the final configuration dictionary.
    merged_config = base_config.copy()
    merged_config.update(
        {key: value for key, value in config.items() if key != "include"}
    )

    return merged_config
