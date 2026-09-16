"""Plan profile slices, load them with adjacent-granule context, then trim it back off.

A granule is processed in slices of ``NB_PROF_SLICE`` profiles. Each slice is
read with ``NB_PROF_CONTEXT`` extra profiles on each side so that the algorithm
never sees an artificial edge; at the granule boundaries that context comes from
the neighboring granules. The context is removed again once the algorithm has
run, so only the requested profiles reach the output product.
"""

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from twod_mcda.reading.access import read_slice


@dataclass
class SliceData:
    """Input and output arrays associated with one processing slice."""

    input: xr.Dataset
    masks: xr.Dataset = field(default_factory=xr.Dataset)
    development: xr.Dataset = field(default_factory=xr.Dataset)
    previous_context_count: int = 0
    next_context_count: int = 0


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


def describe_slice(
    index, slice_count, profile_min, profile_max, context_min, context_max
):
    """Describe both the retained profiles and the full algorithm input."""

    return (
        f"Process slice {index:d}/{slice_count:d} "
        f"(profiles {profile_min:d} to {profile_max:d} using slice "
        f"{context_min:d} to {context_max:d})"
    )


def load_slice(profile_min, profile_max, granule_reader, previous, following):
    """Read one current-granule slice and add context at file edges."""

    data = read_slice(granule_reader, profile_min, profile_max)
    slice_data = SliceData(input=data)
    granule_last_profile = granule_reader.data_reader.nb_profiles - 1

    if profile_min == 0 and previous is not None:
        first_time = data["Profile_Time"].isel(profile=0).item()
        previous_time = previous["Profile_Time"].isel(profile=-1).item()
        time_gap = np.abs(first_time - previous_time)
        print(
            "\tTime between last profile of previous file and first profile "
            f"of current file = {time_gap:.2f} s"
        )
        if profiles_are_consecutive(previous_time, first_time):
            print("\tAppend previous granule")
            slice_data.input = append_adjacent_profiles(data, previous, "start")
            slice_data.previous_context_count = previous.sizes["profile"]
        else:
            print(
                "\tPrevious granule does not seem consecutive. "
                "No start context added."
            )

    if profile_max == granule_last_profile and following is not None:
        following_time = following["Profile_Time"].isel(profile=0).item()
        last_time = data["Profile_Time"].isel(profile=-1).item()
        time_gap = np.abs(following_time - last_time)
        print(
            "\tTime between last profile of current file and first profile "
            f"of next file = {time_gap:.2f} s"
        )
        if profiles_are_consecutive(last_time, following_time):
            print("\tAppend next granule")
            slice_data.input = append_adjacent_profiles(
                slice_data.input,
                following,
                "end",
            )
            slice_data.next_context_count = following.sizes["profile"]
        else:
            print("\tNext granule does not seem consecutive. No end context added.")

    return slice_data


def append_adjacent_profiles(current, adjacent, side):
    """Append all profile-dependent variables from an adjacent granule."""

    if side not in {"start", "end"}:
        raise ValueError(f"Invalid side: {side!r}")

    arrays = (adjacent, current) if side == "start" else (current, adjacent)
    return xr.concat(
        arrays,
        dim="profile",
        data_vars="all",
        coords="minimal",
        compat="override",
        join="override",
    )


def profiles_are_consecutive(first_time, second_time):
    """Return whether two boundary profiles are less than one second apart."""

    difference = abs(second_time - first_time)
    return bool(difference < 1)


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
        raise ValueError(f"Invalid side {side!r}. Expected 'start' or 'end'.")

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
        raise ValueError(f"Unsupported array dimension: {array.ndim}")

    slices = [slice(None)] * array.ndim

    if side == "start":
        slices[profile_axis] = slice(profile_count, None)
    else:
        slices[profile_axis] = slice(None, -profile_count)

    return array[tuple(slices)]
