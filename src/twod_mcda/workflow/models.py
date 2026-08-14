"""Data structures passed through the processing workflow."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


ArrayMapping = dict[str, np.ndarray]


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

    input: ArrayMapping
    masks: ArrayMapping = field(default_factory=dict)
    development: ArrayMapping = field(default_factory=dict)
    previous_context_count: int = 0
    next_context_count: int = 0


@dataclass
class ProcessingResult:
    """Arrays needed to write the final 2D-McDA product."""

    data: ArrayMapping
    development: ArrayMapping
    altitude: np.ndarray
    longitude_min: float
    longitude_max: float
