"""Resolve pipeline configuration into a processing request."""

from pathlib import Path
import re

import numpy as np

from twod_mcda.caliop.constants import (
    LIDAR_DATA_ALTITUDES,
    REGION_4_ALTITUDE_BOUNDARIES,
)
from twod_mcda.caliop.discovery import (
    find_granule_file,
    find_neighbor_granules,
    parse_granule_time,
)
from twod_mcda.caliop.grids import alt_to_regular_30m_vertical_grid
from twod_mcda.version import get_full_version
from twod_mcda.workflow.models import ProcessingRequest


_GRANULE_ID_PATTERN = re.compile(
    r"\.(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z[DN])\.hdf$"
)


def _granule_id(file_path):
    """Extract the full granule identifier, including day/night."""

    if file_path is None:
        return None

    file_path = Path(file_path)
    match = _GRANULE_ID_PATTERN.search(file_path.name)
    if match is None:
        raise ValueError(f"Invalid CALIOP filename format: {file_path.name}")

    return match.group(1)


def _normalized_version(version):
    """Return a version with the upper-case prefix used in product metadata."""

    if version[:1].lower() == "v":
        return f"V{version[1:]}"
    return f"V{version}"


def _resolve_max_altitude_km(max_altitude_km):
    """Resolve the configured maximum altitude, defaulting to the top of region 4."""

    if max_altitude_km is None:
        return REGION_4_ALTITUDE_BOUNDARIES[0]
    return max_altitude_km


def _altitude_index(max_altitude_km):
    """Map a maximum altitude in km to its regular 30 m grid index.

    The lidar altitude grid (``lidar_data_altitudes.pkl``) is expanded to
    the regular 30 m vertical grid used by the reader, ordered bottom to
    top. The returned index is the number of grid bins at or below
    ``max_altitude_km``, suitable for slicing that grid as ``data[:index]``.
    An altitude at or above the top of the grid keeps the whole profile,
    reported as ``None``.
    """

    regular_grid_altitudes = alt_to_regular_30m_vertical_grid(LIDAR_DATA_ALTITUDES)

    index = int(
        np.searchsorted(regular_grid_altitudes, max_altitude_km, side="right")
    )

    if index <= 0:
        raise ValueError(
            "processing.max_altitude_km must be above "
            f"{regular_grid_altitudes[0]:.3f} km, got {max_altitude_km}."
        )

    return index if index < regular_grid_altitudes.size else None


def _output_directory(output_cfg, granule_date, version):
    """Build the output directory from the configured root and path format."""

    relative_path = output_cfg["path_format"].format(
        version=version.removeprefix("V"),
        year=granule_date.year,
        month=granule_date.month,
        day=granule_date.day,
    )

    return Path(output_cfg["root_directory"]) / relative_path


def resolve_processing_request(cfg):
    """Resolve input paths and build a processing request from configuration."""

    current_file = find_granule_file(cfg)
    previous_file, next_file = find_neighbor_granules(cfg)

    processing_cfg = cfg.get("processing", {})
    output_cfg = cfg["output"]
    subset_cfg = cfg.get("subset")
    subset_active = (
        subset_cfg is not None
        and subset_cfg.get("activate", True)
    )
    caliop_cfg = cfg["cal_lid_l1"]
    version = _normalized_version(get_full_version())
    granule_date = cfg["granule"]
    granule_time = parse_granule_time(granule_date)
    maximum_altitude_km = _resolve_max_altitude_km(
        processing_cfg["max_altitude_km"]
    )

    return ProcessingRequest(
        granule_date=granule_date,
        caliop_version=_normalized_version(str(caliop_cfg["version"])),
        current_directory=Path(current_file).parent,
        previous_granule=_granule_id(previous_file),
        previous_directory=(
            Path(previous_file).parent if previous_file is not None else None
        ),
        next_granule=_granule_id(next_file),
        next_directory=(
            Path(next_file).parent if next_file is not None else None
        ),
        subset_active=subset_active,
        subset_mode=(
            subset_cfg.get("mode", "profindex")
            if subset_active else "profindex"
        ),
        subset_start=(subset_cfg.get("start") if subset_active else None),
        subset_end=(subset_cfg.get("end") if subset_active else None),
        save_development_data=processing_cfg.get(
            "save_development_data", False
        ),
        output_version=version,
        output_product_type=output_cfg.get("product_type", "Dev"),
        output_directory=_output_directory(output_cfg, granule_time, version),
        maximum_altitude_km=maximum_altitude_km,
        maximum_altitude_index=_altitude_index(maximum_altitude_km),
    )
