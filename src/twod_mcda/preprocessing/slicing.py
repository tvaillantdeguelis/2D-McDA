"""
Utilities for splitting CALIOP profiles into overlapping processing slices.
"""

import numpy as np


def get_slice_bounds(
    profile_min,
    profile_max,
    slice_size,
    overlap_size,
):
    """
    Compute the start and end profile indexes of processing slices.

    Parameters
    ----------
    profile_min : int
        First profile index to process.
    profile_max : int
        Last profile index to process.
    slice_size : int
        Maximum number of profiles in one slice.
    overlap_size : int
        Number of overlapping profiles between consecutive slices.

    Returns
    -------
    start_indexes : numpy.ndarray
        Start profile index of each slice.
    end_indexes : numpy.ndarray
        End profile index of each slice.
    """

    profile_count = profile_max - profile_min + 1

    # Process the full interval as a single slice when it is small enough.
    if profile_count <= slice_size:
        return (
            np.array([profile_min]),
            np.array([profile_max]),
        )

    # Create overlapping slices. A very short final slice is merged into
    # the previous one by stopping the sequence before half a slice remains.
    start_indexes = np.arange(
        profile_min,
        profile_max - slice_size // 2 + 2,
        slice_size - overlap_size,
    )

    end_indexes = start_indexes + slice_size - 1
    end_indexes[-1] = profile_max

    return start_indexes, end_indexes


def trim_profiles(array, profile_count, side):
    """
    Remove profiles added from an adjacent granule.

    Parameters
    ----------
    array : numpy.ndarray
        Array containing a profile dimension.
    profile_count : int
        Number of profiles to remove.
    side : {"start", "end"}
        Side from which profiles are removed.

    Returns
    -------
    numpy.ndarray
        Trimmed view of the input array.
    """

    if side not in {"start", "end"}:
        raise ValueError(
            f"Invalid side {side!r}. Expected 'start' or 'end'."
        )

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