#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

from config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SBATCH_SCRIPT = PROJECT_ROOT / "scripts" / "process_granule.sbatch"
LOG_ROOT = PROJECT_ROOT / "logs" / "slurm"


def build_job_name(cfg):
    """Build a Slurm job name from the granule and optional subset."""

    granule = cfg["granule"]
    subset = cfg.get("subset")

    if not subset or not subset.get("activate", True):
        return f"2D-McDA_{granule}"

    mode_labels = {
        "longitude": "lon",
        "profindex": "prof",
    }
    mode = mode_labels[subset["mode"]]

    return (
        f"2D-McDA_{granule}_{mode}_"
        f"{subset['start']}_{subset['end']}"
    )


def submit_granule(config_file):
    """
    Submit one CALIOP granule processing job to Slurm.

    Parameters
    ----------
    config_file : str or Path
        Path to the YAML configuration file.
    """

    config_file = Path(config_file).resolve()

    cfg = load_config(config_file)
    granule = cfg["granule"]

    year = granule[:4]
    month = granule[:7]

    log_dir = LOG_ROOT / year / month
    log_dir.mkdir(parents=True, exist_ok=True)

    job_name = build_job_name(cfg)

    command = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={log_dir / f'{job_name}_%j.out'}",
        f"--error={log_dir / f'{job_name}_%j.err'}",
        str(SBATCH_SCRIPT),
        str(config_file),
    ]

    subprocess.run(
        command,
        check=True,
    )


def main():
    """Parse command-line arguments and submit one granule job."""

    parser = argparse.ArgumentParser(
        description="Submit one CALIOP granule processing job."
    )

    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    submit_granule(args.config_file)


if __name__ == "__main__":
    main()
