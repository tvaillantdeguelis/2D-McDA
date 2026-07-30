#!/usr/bin/env python3

import argparse
import os
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from config_loader import load_config
from submit_granule import submit_granule
from twod_mcda.io.granule_finder import find_granules_between_dates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"


def count_user_jobs():
    """
    Return the number of pending or running Slurm jobs for the current user.

    Returns
    -------
    int
        Number of jobs currently visible in the Slurm queue.
    """

    result = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--user",
            os.environ["USER"],
            "--states",
            "PENDING,RUNNING",
            "--format",
            "%i",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return len(result.stdout.splitlines())


def create_granule_config(cfg, granule):
    """
    Create a YAML configuration file for one CALIOP granule.

    Parameters
    ----------
    cfg : dict
        Base processing configuration.
    granule : str
        CALIOP granule identifier.

    Returns
    -------
    Path
        Path to the generated YAML configuration file.
    """

    run_cfg = deepcopy(cfg)
    run_cfg["granule"] = granule

    year = granule[:4]
    month = granule[:7]

    run_dir = RUNS_ROOT / year / month
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_file = run_dir / f"{granule}.yaml"

    with config_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            run_cfg,
            file,
            sort_keys=False,
        )

    return config_file


def submit_period(config_file):
    """
    Submit one Slurm processing job per granule in a given period.

    Parameters
    ----------
    config_file : str or Path
        Path to the YAML period configuration file.
    """

    config_file = Path(config_file).resolve()

    if not config_file.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_file}"
        )

    cfg = load_config(config_file)

    start_date = datetime.strptime(
        cfg["period"]["start_date"],
        "%Y-%m-%d",
    )

    # Use an exclusive upper bound so the complete final day is included.
    end_date = datetime.strptime(
        cfg["period"]["end_date"],
        "%Y-%m-%d",
    ) + timedelta(days=1)

    max_parallel_jobs = cfg["slurm"]["max_parallel_jobs"]

    granules = find_granules_between_dates(
        cfg,
        start_date,
        end_date,
    )

    print(f"Found {len(granules)} granules.")

    for granule in granules:
        while count_user_jobs() >= max_parallel_jobs:
            time.sleep(5)

        granule_config_file = create_granule_config(
            cfg,
            granule,
        )

        print(f"Submitting {granule}")

        submit_granule(granule_config_file)


def main():
    """Parse command-line arguments and submit a processing period."""

    parser = argparse.ArgumentParser(
        description=(
            "Submit CALIOP granule processing jobs for a given period."
        )
    )

    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to the YAML period configuration file.",
    )

    args = parser.parse_args()

    submit_period(args.config_file)


if __name__ == "__main__":
    main()
