"""Plan profile slices, load them with adjacent-granule context, then trim it back off.

A granule is processed in slices of ``NB_PROF_SLICE`` profiles. Each slice is
read with ``NB_PROF_CONTEXT`` extra profiles on each side so that the algorithm
never sees an artificial edge; at the granule boundaries that context comes from
the neighboring granules. The context is removed again once the algorithm has
run, so only the requested profiles reach the output product.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from twod_mcda.reading.access import read_adjacent_profiles, read_slice


@dataclass
class SliceData:
    """Input and output arrays associated with one processing slice."""

    input: xr.Dataset
    masks: xr.Dataset = field(default_factory=xr.Dataset)
    development: xr.Dataset = field(default_factory=xr.Dataset)
    nb_profiles_previous_context: int = 0
    nb_profiles_next_context: int = 0


@dataclass(frozen=True)
class SliceBounds:
    """Profile bounds of one processing slice.

    ``profile_min`` and ``profile_max`` delimit the profiles retained in the
    output. ``context_min`` and ``context_max`` widen them with the processing
    context, and are intentionally not clipped: a negative index or an index
    beyond the current granule identifies context that must come from an
    adjacent granule. ``first_profile_to_load`` and ``last_profile_to_load``
    are the same bounds clipped to the current granule file, so they are what
    ``load_slice`` reads from it.
    """

    profile_min: int
    profile_max: int
    context_min: int
    context_max: int
    first_profile_to_load: int
    last_profile_to_load: int


@dataclass(frozen=True)
class AdjacentContext:
    """Context profiles read from the granules adjacent to the current one.

    Each side is ``None`` when the slices need no context there, or when the
    adjacent granule was not found. The two counts are how many profiles were
    asked of each neighbor, which stays meaningful even when the granule is
    missing. The paths are kept only to report which files provided the context.
    """

    previous_profiles: xr.Dataset | None
    next_profiles: xr.Dataset | None
    previous_granule_path: Path | None
    next_granule_path: Path | None
    nb_profiles_previous_context: int
    nb_profiles_next_context: int


def plan_slices(granule_reader, slice_size, context_size):
    """
    Split the profiles to process into slices and resolve the context of each one.

    Parameters
    ----------
    granule_reader : reading.reader.CALIOPRegularGridReader
        Reader of the granule to process, holding the requested profile range.
    slice_size : int
        Maximum distance between the inclusive result bounds of one slice.
    context_size : int
        Number of input context profiles on each side of every result slice.

    Returns
    -------
    tuple of SliceBounds
        The slices to process, in processing order.
    """

    profile_starts = np.arange(
        granule_reader.prof_min,
        granule_reader.prof_max,
        slice_size,
        dtype=int,
    )
    profile_ends = np.minimum(profile_starts + slice_size, granule_reader.prof_max)
    context_starts = profile_starts - context_size
    context_ends = profile_ends + context_size

    last_profile_in_file = granule_reader.last_profile_in_file
    planned_slices = zip(profile_starts, profile_ends, context_starts, context_ends)
    return tuple(
        SliceBounds(
            profile_min=int(profile_min),
            profile_max=int(profile_max),
            context_min=int(context_min),
            context_max=int(context_max),
            first_profile_to_load=max(int(context_min), 0),
            last_profile_to_load=min(int(context_max), last_profile_in_file),
        )
        for profile_min, profile_max, context_min, context_max in planned_slices
    )


def _read_context_profiles(request, granule, directory, nb_profiles_context, side):
    """Read the context profiles one adjacent granule provides, if it exists."""

    if not nb_profiles_context or granule is None:
        return None, None

    # The previous granule provides its last profiles, the next one its first.
    profile_start = -nb_profiles_context if side == "start" else None
    profile_end = None if side == "start" else nb_profiles_context - 1
    return read_adjacent_profiles(
        request,
        granule,
        directory,
        profile_start,
        profile_end,
    )


def load_adjacent_context(request, slices):
    """Read from the adjacent granules the context profiles the slices expect.

    The context each neighbor must provide is the part of the first and last
    slice that was clipped away when their bounds were restricted to the
    current granule file.
    """

    nb_profiles_previous_context = (
        slices[0].first_profile_to_load - slices[0].context_min
    )
    nb_profiles_next_context = slices[-1].context_max - slices[-1].last_profile_to_load

    previous_profiles, previous_granule_path = _read_context_profiles(
        request,
        request.previous_granule,
        request.previous_granule_directory,
        nb_profiles_previous_context,
        "start",
    )
    next_profiles, next_granule_path = _read_context_profiles(
        request,
        request.next_granule,
        request.next_granule_directory,
        nb_profiles_next_context,
        "end",
    )

    return AdjacentContext(
        previous_profiles=previous_profiles,
        next_profiles=next_profiles,
        previous_granule_path=previous_granule_path,
        next_granule_path=next_granule_path,
        nb_profiles_previous_context=nb_profiles_previous_context,
        nb_profiles_next_context=nb_profiles_next_context,
    )


def load_slice(bounds, granule_reader, adjacent_context):
    """Read one current-granule slice and add context at file edges."""

    data = read_slice(
        granule_reader,
        bounds.first_profile_to_load,
        bounds.last_profile_to_load,
    )
    slice_data = SliceData(input=data)
    granule_last_profile = granule_reader.last_profile_in_file
    previous = adjacent_context.previous_profiles
    following = adjacent_context.next_profiles

    if bounds.first_profile_to_load == 0 and previous is not None:
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
            slice_data.nb_profiles_previous_context = previous.sizes["profile"]
        else:
            print(
                "\tPrevious granule does not seem consecutive. "
                "No start context added."
            )

    if bounds.last_profile_to_load == granule_last_profile and following is not None:
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
            slice_data.nb_profiles_next_context = following.sizes["profile"]
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
        ("start", slice_data.nb_profiles_previous_context),
        ("end", slice_data.nb_profiles_next_context),
    )

    for side, nb_profiles_context in context_by_side:
        if nb_profiles_context == 0:
            continue
        print(f"\n\n*****Remove context from {side} adjacent file...*****")
        slice_data.input = trim_profiles(slice_data.input, nb_profiles_context, side)
        slice_data.masks = trim_profiles(slice_data.masks, nb_profiles_context, side)
        slice_data.development = trim_profiles(
            slice_data.development,
            nb_profiles_context,
            side,
        )


def trim_profiles(array, nb_profiles_context, side):
    """
    Remove profiles added from an adjacent granule.

    Parameters
    ----------
    array : xarray.DataArray or xarray.Dataset
        Labelled object containing a ``profile`` dimension.
    nb_profiles_context : int
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
            slice(nb_profiles_context, None)
            if side == "start"
            else slice(None, -nb_profiles_context)
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
        slices[profile_axis] = slice(nb_profiles_context, None)
    else:
        slices[profile_axis] = slice(None, -nb_profiles_context)

    return array[tuple(slices)]
