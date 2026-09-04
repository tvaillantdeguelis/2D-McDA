"""Assemble whole-granule output datasets from per-slice results."""

import numpy as np
import xarray as xr

from twod_mcda.caliop.constants import FILL_VALUE_FLOAT
from twod_mcda.workflow.models import ProcessingResult

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


def empty_output(profile_count, altitude, profile_start=0):
    """Allocate an xarray product dataset with named coordinates."""

    if np.isscalar(altitude):
        altitude_values = np.arange(int(altitude))
    else:
        altitude_values = np.asarray(altitude)
    coords = {
        "profile": np.arange(profile_start, profile_start + profile_count),
        "altitude": altitude_values,
    }
    profile_dims = ("profile",)
    grid_dims = ("profile", "altitude")
    return xr.Dataset(
        {
            "Profile_ID": xr.DataArray(
                np.full(profile_count, int(FILL_VALUE_FLOAT), dtype=np.int32),
                dims=profile_dims,
            ),
            "Profile_Time": xr.DataArray(
                np.full(profile_count, FILL_VALUE_FLOAT, dtype=np.float64),
                dims=profile_dims,
            ),
            "Profile_UTC_Time": xr.DataArray(
                np.full(profile_count, FILL_VALUE_FLOAT, dtype=np.float64),
                dims=profile_dims,
            ),
            "Latitude": xr.DataArray(
                np.full(profile_count, FILL_VALUE_FLOAT, dtype=np.float32),
                dims=profile_dims,
            ),
            "Longitude": xr.DataArray(
                np.full(profile_count, FILL_VALUE_FLOAT, dtype=np.float32),
                dims=profile_dims,
            ),
            **{
                name: xr.DataArray(
                    np.zeros((profile_count, altitude_values.size), dtype=np.uint8),
                    dims=grid_dims,
                )
                for name in DETECTION_MASKS
            },
        },
        coords=coords,
    )


def store_slice(
    output,
    slice_data,
    profile_min,
    profile_max,
    input_profile_min,
    file_min,
):
    """Store only the result interval, excluding its processing context."""

    file_max = file_min + output.sizes["profile"] - 1
    expected_profiles = np.arange(file_min, file_max + 1)
    if not np.array_equal(output.coords["profile"], expected_profiles):
        output.coords["profile"] = expected_profiles
    copy_min = max(profile_min, file_min)
    copy_max = min(profile_max, file_max)
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


def store_development(
    output,
    slice_development,
    profile_min,
    profile_max,
    input_profile_min,
    file_min,
    profile_count,
):
    """Store development data without the processing context."""

    file_max = file_min + profile_count - 1
    copy_min = max(profile_min, file_min)
    copy_max = min(profile_max, file_max)
    if copy_min > copy_max:
        return

    if "profile" not in output.coords:
        output.coords["profile"] = np.arange(
            file_min,
            file_min + profile_count,
        )

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


def assemble_results(output, development, altitude, granule):
    """Build the complete product payload from assembled slice outputs."""

    return ProcessingResult(
        data=output,
        development=development,
        altitude=altitude,
        longitude_min=granule.lon_min,
        longitude_max=granule.lon_max,
    )
