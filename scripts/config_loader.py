"""
Utility functions for loading YAML configuration files used by the command-line
scripts.
"""

from pathlib import Path

import yaml


def load_config(filename):

    filename = Path(filename)

    with open(filename, "r") as f:
        config = yaml.safe_load(f)

    # Load included default configuration
    include_file = filename.parent / config["include"]

    with open(include_file, "r") as f:
        default_config = yaml.safe_load(f)

    default_config.update(
        {
            key: value
            for key, value in config.items()
            if key != "include"
        }
    )

    config = default_config

    return config