"""CALIOP vertical and horizontal grid transformations."""

import numpy as np

from twod_mcda.caliop.constants import (
    FILL_VALUE_FLOAT,
    LIDAR_ALTITUDE_DIMENSION,
    LAYER_ALTITUDE_R1_INDEX_RANGE,
    LAYER_ALTITUDE_R2_INDEX_RANGE,
    LAYER_ALTITUDE_R3_INDEX_RANGE,
    LAYER_ALTITUDE_R4_INDEX_RANGE,
    LAYER_ALTITUDE_R5_INDEX_RANGE,
    N_30M_BINS_PER_BIN_R1,
    N_30M_BINS_PER_BIN_R2,
    N_30M_BINS_PER_BIN_R3,
    N_30M_BINS_PER_BIN_R4,
    N_30M_BINS_PER_BIN_R5,
    N_BINS_R1,
    N_BINS_R2,
    N_BINS_R3,
    N_BINS_R4,
    N_BINS_R5,
    N_LASER_PULSES_PER_5km,
)


def alt_to_regular_30m_vertical_grid(alt, reverse_altitude=True):
    """
    Put alt in regular grid (30 m)
    reverse_altitude: if True, return array from bottom to top
    """

    # Initialization
    nb_vert_levels = get_nb_regular_30m_vertical_levels()
    reg_grid_alt = np.ones(nb_vert_levels) * FILL_VALUE_FLOAT
    reg_grid_r5_index_range = (0, N_BINS_R5 * N_30M_BINS_PER_BIN_R5)
    reg_grid_r4_index_range = (
        reg_grid_r5_index_range[1],
        reg_grid_r5_index_range[1] + N_BINS_R4 * N_30M_BINS_PER_BIN_R4,
    )
    reg_grid_r3_index_range = (
        reg_grid_r4_index_range[1],
        reg_grid_r4_index_range[1] + N_BINS_R3 * N_30M_BINS_PER_BIN_R3,
    )
    reg_grid_r2_index_range = (
        reg_grid_r3_index_range[1],
        reg_grid_r3_index_range[1] + N_BINS_R2 * N_30M_BINS_PER_BIN_R2,
    )
    reg_grid_r1_index_range = (
        reg_grid_r2_index_range[1],
        reg_grid_r2_index_range[1] + N_BINS_R1 * N_30M_BINS_PER_BIN_R1,
    )

    # Copy region R2 (30 m resolution)
    reg_grid_alt[reg_grid_r2_index_range[0] : reg_grid_r2_index_range[1]] = alt[
        LAYER_ALTITUDE_R2_INDEX_RANGE[0] : LAYER_ALTITUDE_R2_INDEX_RANGE[1] + 1
    ]
    step_30m = (
        alt[LAYER_ALTITUDE_R2_INDEX_RANGE[1] - 2]
        - alt[LAYER_ALTITUDE_R2_INDEX_RANGE[1] - 1]
    )

    # Complete R1 by decreasing by nb_bins×step_30m the lowest bin altitude of R2
    start_index = reg_grid_r1_index_range[0]
    end_index = reg_grid_r1_index_range[1]
    reg_grid_alt[start_index:end_index] = (
        reg_grid_alt[start_index - 1]
        - (np.arange(end_index - start_index) + 1) * step_30m
    )

    # Complete R3, R4, and R5 by increasing by nb_bins×step_30m the highest bin altitude of R2
    end_index = reg_grid_r3_index_range[1]
    reg_grid_alt[end_index - 1 :: -1] = (
        reg_grid_alt[end_index] + (np.arange(end_index) + 1) * step_30m
    )

    if reverse_altitude:
        reg_grid_alt = reg_grid_alt[::-1]

    return reg_grid_alt


def regular_30m_grid_native_bins(reverse_altitude=True):
    """
    Return, for each level of the regular 30 m grid, the index of the native CALIOP
    bin it copies.
    reverse_altitude: if True, list the levels from bottom to top
    """

    # Regions from top (R5) to bottom (R1), as the native bins are ordered
    regions = (
        (LAYER_ALTITUDE_R5_INDEX_RANGE, N_30M_BINS_PER_BIN_R5),
        (LAYER_ALTITUDE_R4_INDEX_RANGE, N_30M_BINS_PER_BIN_R4),
        (LAYER_ALTITUDE_R3_INDEX_RANGE, N_30M_BINS_PER_BIN_R3),
        (LAYER_ALTITUDE_R2_INDEX_RANGE, N_30M_BINS_PER_BIN_R2),
        (LAYER_ALTITUDE_R1_INDEX_RANGE, N_30M_BINS_PER_BIN_R1),
    )
    native_bins = np.concatenate(
        [
            np.repeat(np.arange(first, last + 1), nb_30m_bins_per_bin)
            for (first, last), nb_30m_bins_per_bin in regions
        ]
    )

    if reverse_altitude:
        native_bins = native_bins[::-1]

    return native_bins


def shape_to_regular_30m_vertical_grid(
    data, dim=LIDAR_ALTITUDE_DIMENSION, reverse_altitude=True
):
    """
    Duplicate CALIOP data along ``dim`` to get a regular 30 m vertical resolution grid.
    Each regular level copies the native bin it falls in, so missing values (NaN)
    stay missing.
    reverse_altitude: if True, return array from bottom to top
    """

    native_bins = regular_30m_grid_native_bins(reverse_altitude)
    # The algorithm has always run in double precision on the regular grid
    return data.isel({dim: native_bins}).astype(np.float64)


def get_nb_regular_30m_vertical_levels():

    nb_30m_vert_levels = (
        N_BINS_R1 * N_30M_BINS_PER_BIN_R1
        + N_BINS_R2 * N_30M_BINS_PER_BIN_R2
        + N_BINS_R3 * N_30M_BINS_PER_BIN_R3
        + N_BINS_R4 * N_30M_BINS_PER_BIN_R4
        + N_BINS_R5 * N_30M_BINS_PER_BIN_R5
    )

    return nb_30m_vert_levels


def get_single_shot_index_from_5km_index(i_5km):
    """
    Return min and max single shot indexes of a 5 km profile index
    """
    ss_min = i_5km * N_LASER_PULSES_PER_5km
    ss_max = (i_5km + 1) * N_LASER_PULSES_PER_5km - 1

    return ss_min, ss_max
