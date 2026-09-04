#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import yaml

from config_loader import load_config
from submit_granule import submit_granule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs" / "case_studies"


def create_case_study_config(cfg, case_study, occurrence=None):
    """Create a processing configuration for one case study."""

    run_cfg = deepcopy(cfg)
    run_cfg.pop("case_studies", None)
    run_cfg["granule"] = case_study["granule"]
    run_cfg["subset"] = {
        "activate": True,
        "mode": case_study["mode"],
        "start": case_study["start"],
        "end": case_study["end"],
    }

    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    granule = case_study["granule"]
    suffix = f"_{occurrence}" if occurrence is not None else ""
    config_file = RUNS_ROOT / f"{granule}{suffix}.yaml"

    with config_file.open("w", encoding="utf-8") as file:
        yaml.safe_dump(run_cfg, file, sort_keys=False)

    return config_file


def submit_case_studies(config_file):
    """Submit one Slurm processing job per enabled case study."""

    config_file = Path(config_file).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    cfg = load_config(config_file)
    case_studies = cfg.get("case_studies")
    if not isinstance(case_studies, list):
        raise ValueError("The configuration must define a case_studies list.")

    enabled_case_studies = [
        case_study for case_study in case_studies if case_study.get("enabled", False)
    ]
    print(f"Found {len(enabled_case_studies)} enabled case studies.")

    granule_counts = Counter(
        case_study["granule"] for case_study in enabled_case_studies
    )
    granule_occurrences = defaultdict(int)

    for case_study in enabled_case_studies:
        granule = case_study["granule"]
        granule_occurrences[granule] += 1
        occurrence = (
            granule_occurrences[granule] if granule_counts[granule] > 1 else None
        )
        granule_config_file = create_case_study_config(
            cfg,
            case_study,
            occurrence,
        )
        print(f"Submitting {granule}")
        submit_granule(granule_config_file)


def main():
    """Parse command-line arguments and submit enabled case studies."""

    parser = argparse.ArgumentParser(
        description="Submit enabled CALIOP case studies as Slurm jobs."
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to the YAML case-studies configuration file.",
    )
    args = parser.parse_args()
    submit_case_studies(args.config_file)


if __name__ == "__main__":
    main()
