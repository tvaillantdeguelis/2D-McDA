"""Data structures used by the refactored processing orchestration."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


ArrayMapping = dict[str, np.ndarray]


@dataclass(frozen=True)
class ProcessingRequest:
    """Resolved inputs and options for one CALIOP granule."""

    granule_date: str
    caliop_version: str
    caliop_product_type: str
    current_directory: Path
    previous_granule: str | None
    previous_directory: Path | None
    next_granule: str | None
    next_directory: Path | None
    subset_mode: str
    subset_start: int | float | None
    subset_end: int | float | None
    save_development_data: bool
    output_version: str
    output_product_type: str
    output_directory: Path
    maximum_altitude_index: int | None

    @classmethod
    def from_mapping(cls, values):
        """Build a request from the mapping shared with the legacy runner."""

        def optional_path(value):
            return Path(value) if value is not None else None

        return cls(
            granule_date=values["granule_date"],
            caliop_version=values["cal_lid_l1_version"],
            caliop_product_type=values["cal_lid_l1_type"],
            current_directory=Path(values["folder_path"]),
            previous_granule=values["previous_granule"],
            previous_directory=optional_path(values["previous_folder_path"]),
            next_granule=values["next_granule"],
            next_directory=optional_path(values["next_folder_path"]),
            subset_mode=values["slice_type"],
            subset_start=values["slice_start"],
            subset_end=values["slice_end"],
            save_development_data=values["save_development_data"],
            output_version=values["version_2d_mcda"],
            output_product_type=values["type_2d_mcda"],
            output_directory=Path(values["out_folder"]),
            maximum_altitude_index=values["index30m_alt_max"],
        )


@dataclass
class SliceData:
    """Input and output arrays associated with one processing slice."""

    input: ArrayMapping
    masks: ArrayMapping = field(default_factory=dict)
    development: ArrayMapping = field(default_factory=dict)
    previous_profiles_used: bool = False
    next_profiles_used: bool = False


@dataclass
class ProcessingResult:
    """Arrays needed to write the final 2D-McDA product."""

    data: ArrayMapping
    development: ArrayMapping
    altitude: np.ndarray
    longitude_min: float
    longitude_max: float
