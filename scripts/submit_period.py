#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timedelta
import argparse
import re
import subprocess
import yaml

from config import load_config


GRANULE_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z[ND]"
)


def find_granules(config):

    folder = Path(
        config["cal_lid_l1"]["folder"]
    )

    path_format = config["cal_lid_l1"]["path_format"]

    version = config["cal_lid_l1"]["version"]


    start = datetime.fromisoformat(
        config["period"]["start_date"]
    )

    end = datetime.fromisoformat(
        config["period"]["end_date"]
    )


    granules = []


    current = start

    while current <= end:

        directory = folder / path_format.format(
            version=version,
            year=current.year,
            month=current.month,
            day=current.day,
        )


        if directory.exists():

            for filename in directory.iterdir():

                match = GRANULE_PATTERN.search(
                    filename.name
                )

                if match:
                    granules.append(
                        match.group(0)
                    )


        current += timedelta(days=1)


    return sorted(set(granules))


def create_granule_configs(
    config_file,
    granules,
    output_dir,
):

    output_dir.mkdir(
        exist_ok=True
    )


    yaml_files = []


    for index, granule in enumerate(granules):

        filename = (
            output_dir /
            f"granule_{index:06d}.yaml"
        )


        content = {
            "include": "../config/default.yaml",
            "granule": {
                "granule": granule
            },
            "slicing": {
                "type": "profindex",
                "start": None,
                "end": None,
            },
        }


        with open(filename, "w") as f:
            yaml.safe_dump(
                content,
                f,
            )


        yaml_files.append(filename)


    return yaml_files


def submit_array(yaml_files, max_parallel_jobs):

    list_file = Path("tmp/yaml_files.txt")

    with open(list_file, "w") as f:

        for filename in yaml_files:
            f.write(
                str(filename) + "\n"
            )


    nb_jobs = len(yaml_files)


    subprocess.run(
        [
            "sbatch",
            f"--array=0-{nb_jobs-1}%{max_parallel_jobs}",
            "scripts/run_granule_array.sbatch",
            str(list_file),
        ],
        check=True,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    config = load_config(args.config)

    max_parallel_jobs = config["slurm"]["max_parallel_jobs"]

    granules = find_granules(config)

    print(f"{len(granules)} granules found")

    yaml_files = create_granule_configs(
        args.config,
        granules,
        Path("tmp"),
    )

    submit_array(yaml_files, max_parallel_jobs)


if __name__ == "__main__":
    main()