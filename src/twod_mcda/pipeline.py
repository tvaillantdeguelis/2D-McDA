"""Top-level processing pipeline.

This module currently acts as a compatibility bridge to the complete legacy
algorithm.  Its deliberately small surface makes it possible to replace the
legacy stages one at a time during the refactoring.
"""

from datetime import datetime
from pathlib import Path
import re
import runpy
import time

from .io.granule_finder import find_granule_file, find_neighbor_granules
from .version import get_full_version


_GRANULE_ID_PATTERN = re.compile(
    r"\.(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z[DN])\.hdf$"
)


def _granule_id(file_path):
    """Extract the full legacy granule identifier, including day/night."""

    if file_path is None:
        return None

    file_path = Path(file_path)
    match = _GRANULE_ID_PATTERN.search(file_path.name)
    if match is None:
        raise ValueError(f"Invalid CALIOP filename format: {file_path.name}")

    return match.group(1)


def _legacy_version(version):
    """Return a version using the upper-case prefix expected by legacy code."""

    if version[:1].lower() == "v":
        return f"V{version[1:]}"
    return f"V{version}"


def _legacy_config(cfg, current_file, previous_file, next_file):
    """Translate the refactored YAML configuration to legacy parameters."""

    processing_cfg = cfg.get("processing", {})
    output_cfg = cfg["output"]
    slicing_cfg = cfg.get("slicing", {})
    caliop_cfg = cfg["cal_lid_l1"]

    return {
        "granule_date": _granule_id(current_file),
        "cal_lid_l1_version": _legacy_version(str(caliop_cfg["version"])),
        "cal_lid_l1_type": caliop_cfg["type"],
        "folder_path": str(Path(current_file).parent),
        "previous_granule": _granule_id(previous_file),
        "previous_folder_path": (
            str(Path(previous_file).parent) if previous_file is not None else None
        ),
        "next_granule": _granule_id(next_file),
        "next_folder_path": (
            str(Path(next_file).parent) if next_file is not None else None
        ),
        "slice_type": slicing_cfg.get("type", "profindex"),
        "slice_start": slicing_cfg.get("start"),
        "slice_end": slicing_cfg.get("end"),
        "save_development_data": processing_cfg.get(
            "save_development_data", False
        ),
        "version_2d_mcda": _legacy_version(get_full_version()),
        "type_2d_mcda": output_cfg.get("type", "Dev"),
        "out_folder": output_cfg["folder"],
        "index30m_alt_max": (
            None if processing_cfg.get("process_up_to_40km", False) else 600
        ),
    }


def process_granule(cfg):
    """Process one granule by invoking the complete legacy implementation."""

    start_time = datetime.now().astimezone()
    start_tic = time.perf_counter()
    print(f"Start time: {start_time}")

    granule_time = datetime.strptime(cfg["granule"], "%Y-%m-%dT%H-%M-%SZN")
    current_file = find_granule_file(cfg, granule_time)
    previous_file, next_file = find_neighbor_granules(cfg, current_file)

    print(f"Previous granule: {previous_file}")
    print(f"Current granule : {current_file}")
    print(f"Next granule    : {next_file}")

    legacy_config = _legacy_config(
        cfg,
        current_file,
        previous_file,
        next_file,
    )

    print(f"2D-McDA version: {legacy_config['version_2d_mcda']}")
    runpy.run_module(
        "legacy.process_granule_old",
        init_globals={"LEGACY_CONFIG": legacy_config},
        run_name="__main__",
    )

    end_time = datetime.now().astimezone()
    total_time = time.perf_counter() - start_tic
    print(f"End time: {end_time}")
    print(f"Total runtime: {total_time:.1f} s")
