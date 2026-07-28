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

    # Create the command-line argument parser
    parser = argparse.ArgumentParser(
        description="Process one CALIOP granule"
    )

    # YAML configuration file containing the processing parameters
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file",
    )

    # Parse command-line arguments
    args = parser.parse_args()


    # Load YAML configuration.
    cfg = load_config(args.config)


    # Run the scientific processing pipeline.
    process_granule(cfg)


if __name__ == "__main__":
    main()
