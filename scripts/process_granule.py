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
from twod_mcda.pipeline import process_granule


def main():
    """Parse command-line arguments and process one CALIOP granule."""

    parser = argparse.ArgumentParser(
        description="Process one CALIOP granule."
    )

    parser.add_argument(
        "config_file",
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config_file)

    process_granule(cfg)


if __name__ == "__main__":
    main()
