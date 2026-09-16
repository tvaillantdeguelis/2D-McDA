"""Resolve pipeline configuration into a processing request."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from twod_mcda.caliop.constants import (
    LIDAR_DATA_ALTITUDES,
    REGION_4_ALTITUDE_BOUNDARIES,
)
from twod_mcda.caliop.grids import alt_to_regular_30m_vertical_grid
from twod_mcda.reading.discovery import (
    find_granule_file,
    find_neighbor_granules,
    parse_granule_time,
)
from twod_mcda.version import get_full_version

_GRANULE_IN_FILENAME_PATTERN = re.compile(
    r"\.(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z[DN])\.hdf$"
)


@dataclass(frozen=True)
class ProcessingRequest:
    """Resolved inputs and options for one CALIOP granule."""

    granule: str
    caliop_version: str
    current_granule_directory: Path
    previous_granule: str | None
    previous_granule_directory: Path | None
    next_granule: str | None
    next_granule_directory: Path | None
    subset_active: bool
    subset_mode: str
    subset_start: int | float | None
    subset_end: int | float | None
    save_development_data: bool
    output_version: str
    output_product_type: str
    output_directory: Path
    maximum_altitude_km: int | float
    maximum_altitude_index: int | None


def _get_granule(file_path):
    """Extract the granule, including day/night, from a CALIOP file path."""

    if file_path is None:
        return None

    file_path = Path(file_path)
    match = _GRANULE_IN_FILENAME_PATTERN.search(file_path.name)
    if match is None:
        raise ValueError(f"Invalid CALIOP filename format: {file_path.name}")

    return match.group(1)


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

    index = int(np.searchsorted(regular_grid_altitudes, max_altitude_km, side="right"))

    if index <= 0:
        raise ValueError(
            "processing.max_altitude_km must be above "
            f"{regular_grid_altitudes[0]:.3f} km, got {max_altitude_km}."
        )

    return index if index < regular_grid_altitudes.size else None


def _output_directory(output_cfg, granule_time, version):
    """Build the output directory from the configured root and path format."""

    relative_path = output_cfg["path_format"].format(
        version=version,
        year=granule_time.year,
        month=granule_time.month,
        day=granule_time.day,
    )

    return Path(output_cfg["root_directory"]) / relative_path


def resolve_processing_request(cfg):
    """Resolve input paths and build a processing request from configuration."""

    current_file = find_granule_file(cfg)
    previous_file, next_file = find_neighbor_granules(cfg)

    processing_cfg = cfg.get("processing", {})
    output_cfg = cfg["output"]
    subset_cfg = cfg.get("subset")
    subset_active = subset_cfg is not None and subset_cfg.get("activate", True)
    caliop_cfg = cfg["cal_lid_l1"]
    caliop_version = caliop_cfg["version"]
    if not isinstance(caliop_version, str):
        raise ValueError(
            f"cal_lid_l1.version must be a string, got {type(caliop_version).__name__}: "
            f"{caliop_version!r}"
        )
    output_version = get_full_version().removeprefix("v")
    granule = cfg["granule"]
    granule_time = parse_granule_time(granule)
    maximum_altitude_km = _resolve_max_altitude_km(
        processing_cfg.get("max_altitude_km", None)
    )

    return ProcessingRequest(
        granule=granule,
        caliop_version=caliop_version,
        current_granule_directory=Path(current_file).parent,
        previous_granule=_get_granule(previous_file),
        previous_granule_directory=(
            Path(previous_file).parent if previous_file is not None else None
        ),
        next_granule=_get_granule(next_file),
        next_granule_directory=(
            Path(next_file).parent if next_file is not None else None
        ),
        subset_active=subset_active,
        subset_mode=(
            subset_cfg.get("mode", "profindex") if subset_active else "profindex"
        ),
        subset_start=(subset_cfg.get("start") if subset_active else None),
        subset_end=(subset_cfg.get("end") if subset_active else None),
        save_development_data=processing_cfg.get("save_development_data", False),
        output_version=output_version,
        output_product_type=output_cfg.get("product_type", "Dev"),
        output_directory=_output_directory(output_cfg, granule_time, output_version),
        maximum_altitude_km=maximum_altitude_km,
        maximum_altitude_index=_altitude_index(maximum_altitude_km),
    )
