"""Plan profile slices and trim adjacent-granule context."""

import numpy as np
import xarray as xr


def plan_slices(
    profile_min,
    profile_max,
    slice_size,
    context_size,
):
    """
    Compute result slices and the context required to process each one.

    Context bounds are intentionally not clipped to a granule: negative
    indexes and indexes beyond the current granule identify data that must be
    read from adjacent granules.

    Parameters
    ----------
    profile_min : int
        First requested profile index.
    profile_max : int
        Last requested profile index.
    slice_size : int
        Maximum distance between the inclusive result bounds of one slice.
    context_size : int
        Number of input context profiles on each side of every result slice.

    Returns
    -------
    profile_starts, profile_ends : numpy.ndarray
        Inclusive bounds of the profiles retained from each slice.
    context_starts, context_ends : numpy.ndarray
        Inclusive input bounds used to process each slice.
    """

    profile_starts = np.arange(
        profile_min,
        profile_max,
        slice_size,
        dtype=int,
    )
    if profile_starts.size == 0:
        profile_starts = np.array([profile_min], dtype=int)

    profile_ends = np.minimum(profile_starts + slice_size, profile_max)
    context_starts = profile_starts - context_size
    context_ends = profile_ends + context_size

    return profile_starts, profile_ends, context_starts, context_ends


def trim_slice_context(slice_data):
    """Remove the neighboring-granule context added on either side of a slice."""

    context_by_side = (
        ("start", slice_data.previous_context_count),
        ("end", slice_data.next_context_count),
    )

    for side, profile_count in context_by_side:
        if profile_count == 0:
            continue
        print(f"\n\n*****Remove context from {side} adjacent file...*****")
        slice_data.input = trim_profiles(slice_data.input, profile_count, side)
        slice_data.masks = trim_profiles(slice_data.masks, profile_count, side)
        slice_data.development = trim_profiles(
            slice_data.development,
            profile_count,
            side,
        )


def trim_profiles(array, profile_count, side):
    """
    Remove profiles added from an adjacent granule.

    Parameters
    ----------
    array : xarray.DataArray or xarray.Dataset
        Labelled object containing a ``profile`` dimension.
    profile_count : int
        Number of profiles to remove.
    side : {"start", "end"}
        Side from which profiles are removed.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Trimmed view of the input object.
    """

    if side not in {"start", "end"}:
        raise ValueError(
            f"Invalid side {side!r}. Expected 'start' or 'end'."
        )

    if isinstance(array, (xr.DataArray, xr.Dataset)):
        if "profile" not in array.dims:
            return array
        indexer = (
            slice(profile_count, None)
            if side == "start"
            else slice(None, -profile_count)
        )
        return array.isel(profile=indexer)

    if array.ndim == 1:
        profile_axis = 0
    elif array.ndim == 2:
        profile_axis = 0
    elif array.ndim == 3:
        profile_axis = 1
    else:
        raise ValueError(
            f"Unsupported array dimension: {array.ndim}"
        )

    slices = [slice(None)] * array.ndim

    if side == "start":
        slices[profile_axis] = slice(profile_count, None)
    else:
        slices[profile_axis] = slice(None, -profile_count)

    return array[tuple(slices)]
