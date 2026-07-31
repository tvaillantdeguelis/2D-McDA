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


def _altitude_index(max_altitude_km):
    """Map a supported maximum altitude to the legacy grid index."""

    if max_altitude_km == 30:
        return 600
    if max_altitude_km == 40:
        return None

    raise ValueError(
        "processing.max_altitude_km must be either 30 or 40."
    )


def _output_directory(output_cfg, granule_date, version):
    """Build the output directory from the configured root and path format."""

    _validate_output_filetype(output_cfg)

    relative_path = output_cfg["path_format"].format(
        version=version.removeprefix("V"),
        year=granule_date.year,
        month=granule_date.month,
        day=granule_date.day,
    )

    return Path(output_cfg["root_directory"]) / relative_path


def _validate_output_filetype(output_cfg):
    """Reject output formats that are not implemented."""

    if output_cfg["filetype"] != "HDF":
        raise ValueError('output.filetype must be "HDF".')


def _validate_config_schema(cfg):
    """Reject removed configuration keys with actionable error messages."""

    renamed_keys = (
        (cfg.get("processing", {}), "process_up_to_40km", "max_altitude_km"),
        (cfg.get("cal_lid_l1", {}), "folder", "root_directory"),
        (cfg.get("cal_lid_l1", {}), "type", "product_type"),
        (cfg.get("output", {}), "folder", "root_directory"),
        (cfg.get("output", {}), "type", "product_type"),
    )

    for section, old_key, new_key in renamed_keys:
        if old_key in section:
            raise ValueError(
                f'Configuration key "{old_key}" was renamed to "{new_key}".'
            )

    if "slicing" in cfg:
        raise ValueError(
            'Configuration section "slicing" was renamed to "subset", '
            'and its "type" key was renamed to "mode".'
        )

    if "type" in cfg.get("subset", {}):
        raise ValueError(
            'Configuration key "subset.type" was renamed to "subset.mode".'
        )

    if "make_classification" in cfg.get("processing", {}):
        raise ValueError(
            'Configuration key "processing.make_classification" was removed.'
        )


def _legacy_config(cfg, current_file, previous_file, next_file):
    """Translate the refactored YAML configuration to legacy parameters."""

    _validate_config_schema(cfg)

    processing_cfg = cfg.get("processing", {})
    output_cfg = cfg["output"]
    subset_cfg = cfg.get("subset", {})
    caliop_cfg = cfg["cal_lid_l1"]
    version = _legacy_version(get_full_version())
    granule_date = _granule_id(current_file)
    granule_time = datetime.strptime(
        granule_date[:19],
        "%Y-%m-%dT%H-%M-%S",
    )

    return {
        "granule_date": granule_date,
        "cal_lid_l1_version": _legacy_version(str(caliop_cfg["version"])),
        "cal_lid_l1_type": caliop_cfg["product_type"],
        "folder_path": str(Path(current_file).parent),
        "previous_granule": _granule_id(previous_file),
        "previous_folder_path": (
            str(Path(previous_file).parent) if previous_file is not None else None
        ),
        "next_granule": _granule_id(next_file),
        "next_folder_path": (
            str(Path(next_file).parent) if next_file is not None else None
        ),
        "slice_type": subset_cfg.get("mode", "profindex"),
        "slice_start": subset_cfg.get("start"),
        "slice_end": subset_cfg.get("end"),
        "save_development_data": processing_cfg.get(
            "save_development_data", False
        ),
        "version_2d_mcda": version,
        "type_2d_mcda": output_cfg.get("product_type", "Dev"),
        "out_folder": str(
            _output_directory(output_cfg, granule_time, version)
        ),
        "index30m_alt_max": _altitude_index(
            processing_cfg["max_altitude_km"]
        ),
    }


def process_granule(cfg):
    """Process one granule by invoking the complete legacy implementation."""

    _validate_config_schema(cfg)
    _validate_output_filetype(cfg["output"])

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
