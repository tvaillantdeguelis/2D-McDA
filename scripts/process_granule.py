#!/usr/bin/env python3

"""
Command-line entry point for processing one CALIOP granule.

This script:
    1. Reads a YAML configuration file.
    2. Loads the complete configuration.
    3. Calls the scientific processing pipeline.
"""

import argparse

from config_loader import load_config
from twod_mcda.pipeline import run_granule_pipeline


def main():
    """Parse command-line arguments and process one CALIOP granule."""

    # Create a command-line parser for this script.
    # It handles arguments passed when running the script like:
    # python process_granule.py /path/to/config.yaml
    parser = argparse.ArgumentParser(
        description="Process one CALIOP granule."
    )

    # Declare a required positional argument named config_file.
    parser.add_argument(
        "config_file",
        help="Path to the YAML configuration file.",
    )

    # Parse the command-line arguments provided by the user.
    args = parser.parse_args()

    cfg = load_config(args.config_file)

    run_granule_pipeline(cfg)


if __name__ == "__main__":
    main()
