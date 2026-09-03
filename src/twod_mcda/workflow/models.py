"""Data structures passed through the processing workflow."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr


ArrayMapping = xr.Dataset


@dataclass(frozen=True)
class ProcessingRequest:
    """Resolved inputs and options for one CALIOP granule."""

    granule_date: str
    caliop_version: str
    current_directory: Path
    previous_granule: str | None
    previous_directory: Path | None
    next_granule: str | None
    next_directory: Path | None
    subset_active: bool
    subset_mode: str
    subset_start: int | float | None
    subset_end: int | float | None
    save_development_data: bool
    output_version: str
    output_product_type: str
    output_directory: Path
    maximum_altitude_km: int
    maximum_altitude_index: int | None


@dataclass
class SliceData:
    """Input and output arrays associated with one processing slice."""

    input: xr.Dataset
    masks: xr.Dataset = field(default_factory=xr.Dataset)
    development: xr.Dataset = field(default_factory=xr.Dataset)
    previous_context_count: int = 0
    next_context_count: int = 0


@dataclass
class ProcessingResult:
    """Arrays needed to write the final 2D-McDA product."""

    data: xr.Dataset
    development: xr.Dataset
    altitude: xr.DataArray
    longitude_min: float
    longitude_max: float


@dataclass
class GranulePreparation:
    """Everything computed once for a granule, before running the algorithm.

    Built by ``workflow.preparation.prepare_granule``: the planned slices,
    the neighboring-granule context profiles, and the empty whole-granule
    output datasets that the algorithm will fill slice by slice.
    """

    profile_starts: np.ndarray
    profile_ends: np.ndarray
    context_starts: np.ndarray
    context_ends: np.ndarray
    slice_count: int
    profile_count: int
    last_profile_in_file: int
    previous_profiles: xr.Dataset | None
    next_profiles: xr.Dataset | None
    altitude: xr.DataArray
    granule_detection_product: xr.Dataset
    granule_development_data: xr.Dataset
