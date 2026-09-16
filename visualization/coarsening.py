"""Degrade CALIOP data to a coarser resolution, for side-by-side viewing.

The 2D-McDA pipeline always works at the native 333 m × 30 m resolution. Only
the viewer needs the coarser CALIOP resolutions, to show what a feature looks
like once averaged the way the operational Level 2 products are, which is why
this lives next to the notebook rather than in ``twod_mcda``.

The averaging replaces every bin by the mean of its block, so the array keeps
its 333 m × 30 m shape: the resolution changes, the grid does not.
"""

import numpy as np

#: Number of 333 m profiles averaged together per coarse horizontal resolution.
PROFILES_PER_HORIZONTAL_BIN = {"333m": 1, "1km": 3, "5km": 15}

#: Number of 30 m bins averaged together per coarse vertical resolution.
BINS_PER_VERTICAL_BIN = {"30m": 1, "60m": 2, "180m": 6}

#: Profiles per 5 km chunk in the CALIOP averaging scheme.
PROFILES_PER_CHUNK = 15


def first_profile_of_chunk(profile_index):
    """
    Return the offset of the first complete 5 km chunk at or after a profile.

    :param profile_index: granule-wide index of the first profile loaded
    :return: offset, within the loaded profiles, of the first chunk boundary
    """

    offset = -profile_index % PROFILES_PER_CHUNK

    return int(offset)


def _block_average(data, block_size, axis, start=0):
    """Replace each complete block along one axis by its mean."""

    if block_size == 1:
        return data

    length = data.shape[axis]
    averaged = np.ma.copy(data)
    for block_start in range(start, length - block_size + 1, block_size):
        block = slice(block_start, block_start + block_size)
        indexer = [slice(None), slice(None)]
        indexer[axis] = block
        mean = np.ma.mean(data[tuple(indexer)], axis=axis, keepdims=True)
        averaged[tuple(indexer)] = np.ma.repeat(mean, block_size, axis=axis)

    return averaged


def coarsen(data, vertical_resolution, horizontal_resolution, profile_min):
    """
    Average CALIOP data to a coarser resolution, keeping the 333 m × 30 m grid.

    Blocks that the array does not hold in full, at the top of the profile or
    past the last chunk boundary, are left at their native resolution.

    :param data: 2D masked array indexed (profile, altitude)
    :param vertical_resolution: '30m', '60m' or '180m'
    :param horizontal_resolution: '333m', '1km' or '5km'
    :param profile_min: granule-wide index of the first profile of ``data``,
                        used to align the averaging on the 5 km chunks
    :return: averaged array, with the shape of ``data``
    """

    if vertical_resolution not in BINS_PER_VERTICAL_BIN:
        raise ValueError(
            f"Unknown vertical resolution {vertical_resolution!r}; use one of "
            f"{sorted(BINS_PER_VERTICAL_BIN)}."
        )
    if horizontal_resolution not in PROFILES_PER_HORIZONTAL_BIN:
        raise ValueError(
            f"Unknown horizontal resolution {horizontal_resolution!r}; use one "
            f"of {sorted(PROFILES_PER_HORIZONTAL_BIN)}."
        )
    if data.ndim != 2:
        raise ValueError(f"Expected a (profile, altitude) array, got {data.shape}.")

    averaged = _block_average(
        data,
        BINS_PER_VERTICAL_BIN[vertical_resolution],
        axis=1,
    )

    return _block_average(
        averaged,
        PROFILES_PER_HORIZONTAL_BIN[horizontal_resolution],
        axis=0,
        start=first_profile_of_chunk(profile_min),
    )
