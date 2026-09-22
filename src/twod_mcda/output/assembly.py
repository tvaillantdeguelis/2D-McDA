"""Allocate the output datasets of one run and fill them from per-slice results."""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from twod_mcda.caliop.constants import FILL_VALUE_FLOAT

PROFILE_METADATA = (
    "Profile_ID",
    "Profile_Time",
    "Profile_UTC_Time",
    "Latitude",
    "Longitude",
)

DETECTION_MASKS = (
    "Parallel_Detection_Flags_532",
    "Perpendicular_Detection_Flags_532",
    "Detection_Flags_1064",
    "Composite_Detection_Flags",
)


@dataclass
class ProcessingResult:
    """Arrays needed to write the final 2D-McDA product."""

    data: xr.Dataset
    development: xr.Dataset
    altitude: xr.DataArray
    longitude_min: float
    longitude_max: float


@dataclass
class OutputDatasets:
    """The output arrays the algorithm fills slice by slice.

    They span only the profiles the run was asked to process, which may be a
    subset of the granule: their ``profile`` coordinate starts at the first
    requested profile. ``development`` holds the intermediate algorithm arrays
    and stays empty unless they are saved; ``store_development`` creates its
    variables on first use, since their step dimensions are known only once a
    slice has been processed.
    """

    detection: xr.Dataset
    development: xr.Dataset
    altitude: xr.DataArray


def empty_outputs(granule_reader):
    """Allocate the output datasets covering the profiles to process."""

    altitude = granule_reader.get_data("Lidar_Data_Altitudes")
    detection = _empty_detection_output(
        granule_reader.nb_profiles,
        altitude.values,
        granule_reader.prof_min,
    )
    return OutputDatasets(
        detection=detection,
        development=xr.Dataset(coords=detection.coords),
        altitude=altitude,
    )


def _empty_detection_output(nb_profiles, altitude, profile_start=0):
    """Allocate an xarray product dataset with named coordinates."""

    coords = {
        "profile": np.arange(profile_start, profile_start + nb_profiles),
        "altitude": altitude,
    }
    profile_dims = ("profile",)
    grid_dims = ("profile", "altitude")
    return xr.Dataset(
        {
            "Profile_ID": xr.DataArray(
                np.full(nb_profiles, int(FILL_VALUE_FLOAT), dtype=np.int32),
                dims=profile_dims,
            ),
            "Profile_Time": xr.DataArray(
                np.full(nb_profiles, FILL_VALUE_FLOAT, dtype=np.float64),
                dims=profile_dims,
            ),
            "Profile_UTC_Time": xr.DataArray(
                np.full(nb_profiles, FILL_VALUE_FLOAT, dtype=np.float64),
                dims=profile_dims,
            ),
            "Latitude": xr.DataArray(
                np.full(nb_profiles, FILL_VALUE_FLOAT, dtype=np.float32),
                dims=profile_dims,
            ),
            "Longitude": xr.DataArray(
                np.full(nb_profiles, FILL_VALUE_FLOAT, dtype=np.float32),
                dims=profile_dims,
            ),
            **{
                name: xr.DataArray(
                    np.zeros((nb_profiles, altitude.size), dtype=np.uint8),
                    dims=grid_dims,
                )
                for name in DETECTION_MASKS
            },
        },
        coords=coords,
    )


def store_slice(output, slice_data, bounds, granule_reader):
    """Store only the result interval, excluding its processing context."""

    output_min = granule_reader.prof_min
    output_max = output_min + output.sizes["profile"] - 1
    expected_profiles = np.arange(output_min, output_max + 1)
    if not np.array_equal(output.coords["profile"], expected_profiles):
        output.coords["profile"] = expected_profiles
    copy_min = max(bounds.profile_min, output_min)
    copy_max = min(bounds.profile_max, output_max)
    if copy_min > copy_max:
        return

    selected_profiles = slice(copy_min, copy_max)

    for name in PROFILE_METADATA:
        output[name].loc[{"profile": selected_profiles}] = slice_data.input[name].sel(
            profile=selected_profiles
        )

    for name in DETECTION_MASKS:
        output[name].loc[{"profile": selected_profiles}] = slice_data.masks[name].sel(
            profile=selected_profiles
        )


def store_development(output, slice_development, bounds, granule_reader):
    """Store development data without the processing context."""

    output_min = granule_reader.prof_min
    output_max = output_min + granule_reader.nb_profiles - 1
    copy_min = max(bounds.profile_min, output_min)
    copy_max = min(bounds.profile_max, output_max)
    if copy_min > copy_max:
        return

    if "profile" not in output.coords:
        output.coords["profile"] = np.arange(output_min, output_max + 1)

    for name, values in slice_development.items():
        if "profile" not in values.dims:
            raise ValueError(
                f"Unsupported development array shape for {name!r}: " f"{values.shape}."
            )

        selected = values.sel(profile=slice(copy_min, copy_max))
        if name not in output:
            fill_value = FILL_VALUE_FLOAT if values.dtype.kind == "f" else 0
            output[name] = values.reindex(
                profile=output.coords["profile"],
                fill_value=fill_value,
            )

        output[name].loc[{"profile": selected.coords["profile"]}] = selected


def assemble_results(outputs, granule_reader):
    """Build the complete product payload from assembled slice outputs."""

    return ProcessingResult(
        data=outputs.detection,
        development=outputs.development,
        altitude=outputs.altitude,
        longitude_min=granule_reader.lon_min,
        longitude_max=granule_reader.lon_max,
    )
