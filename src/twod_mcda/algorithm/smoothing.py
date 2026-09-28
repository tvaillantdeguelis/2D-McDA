"""Average the lidar signal, and adjust its noise threshold accordingly.

Each function here consumes the signal and returns it together with the updated
standard deviation, because averaging n samples divides the noise by sqrt(n).
"""

import numpy as np
from scipy.ndimage import convolve1d

from twod_mcda.caliop.constants import (
    N_30M_BINS_PER_BIN_R1,
    N_30M_BINS_PER_BIN_R2,
    N_30M_BINS_PER_BIN_R3,
    N_BINS_R1,
    N_BINS_R2,
    N_LASER_PULSES_PER_1km,
)
from twod_mcda.parameters import (
    FLAG_AFA,
    FLAG_FA,
    FLAG_LIKELY_ARTIFACT,
    FLAG_NOTHING,
    FLAG_SMALL_STRIPS,
)


def remove_detect_from_sr(sr, feature):
    """Remove detected pixel from the ATSR signal"""

    # NaN where not "nothing"
    return np.where(feature != FLAG_NOTHING, np.nan, sr)

def average_below_8_2(sr, sr_sigma, first_profile):
    """Average below 8.2 km as between 8.2 km and 20.2 km (60 m × 1 km)

    ``sr`` and ``sr_sigma`` are indexed (profile, altitude), altitude from
    bottom to top; ``first_profile`` is the index, in its granule, of the first
    profile, which aligns the averaging on the granule's 1 km profiles.
    """

    # Initialization
    new_sr = sr.copy()
    nb_prof = sr.shape[0]
    nb_bins_below_8_2km = (
        N_30M_BINS_PER_BIN_R1 * N_BINS_R1 + N_30M_BINS_PER_BIN_R2 * N_BINS_R2
    )
    block_h = N_LASER_PULSES_PER_1km  # 3 profiles of 333 m
    block_v = N_30M_BINS_PER_BIN_R3  # 2 bins of 30 m

    # A granule holds whole 5 km frames, so its 1 km profiles start at the
    # profile indexes that are multiples of 3
    offset_h = -first_profile % block_h

    # Average 60 m × 1 km; the profiles left over at both ends are not averaged
    nb_blocks_h = (nb_prof - offset_h) // block_h
    rows = slice(offset_h, offset_h + block_h * nb_blocks_h)
    nb_bins = min(nb_bins_below_8_2km, sr.shape[1])
    paired = block_v * (nb_bins // block_v)
    new_sr[rows, :paired] = _mean_by_block(sr[rows, :paired], block_h, block_v)
    if nb_bins > paired:
        # Profile truncated at an odd number of bins: last block 1 bin high
        new_sr[rows, paired:nb_bins] = _mean_by_block(
            sr[rows, paired:nb_bins], block_h, 1
        )

    # Keep missing where was already missing
    new_sr[np.isnan(sr)] = np.nan

    # Adapt SR threshold where averaged, below 8.2 km
    sr_sigma = sr_sigma.copy()
    sr_sigma[rows, :nb_bins] = sr_sigma[rows, :nb_bins] / np.sqrt(block_h * block_v)

    return new_sr, sr_sigma

def _mean_by_block(values, block_height, block_width):
    """Spread over each block the mean of its non-missing values.

    ``values`` is tiled by blocks of ``block_height`` rows × ``block_width``
    columns; a block without any valid value stays missing (NaN).
    """

    nb_rows = values.shape[0] // block_height
    nb_cols = values.shape[1] // block_width
    # Gather each block's pixels in row order along a last axis
    blocks = (
        values.reshape(nb_rows, block_height, nb_cols, block_width)
        .transpose(0, 2, 1, 3)
        .reshape(nb_rows, nb_cols, block_height * block_width)
    )
    valid = ~np.isnan(blocks)
    count = valid.sum(axis=-1)
    total = np.where(valid, blocks, 0.0).sum(axis=-1)
    mean = np.full(total.shape, np.nan)
    np.divide(total * 1.0, count, out=mean, where=count > 0)
    return np.repeat(np.repeat(mean, block_height, axis=0), block_width, axis=1)

def gaussian_2d_window(
    width_window,
    horizontal_gauss_sigma,
    ab_signal,
    feature,
    ab_sigma,
    height_window=7,
    vertical_gauss_sigma=3,
):
    """Apply a 2-D gaussian averaging window to the AB signal"""

    # Initialization
    ab2 = np.asarray(ab_signal, dtype=float)
    new_ab = np.full(ab_signal.shape, np.nan)

    # width_window should be odd numbers
    if width_window % 2 != 1:
        raise ValueError(f"width_window (= {width_window}) should be odd")

    # Apply gaussian 2-D averaging
    h_nside = np.int64((width_window - 1) / 2)
    x = np.arange(width_window) - h_nside
    v_nside = np.int64((height_window - 1) / 2)
    y = np.arange(height_window) - v_nside
    horizontal_gaussian = np.exp(-(x**2) / (2 * horizontal_gauss_sigma**2))
    vertical_gaussian = np.exp(-(y**2) / (2 * vertical_gauss_sigma**2))
    nb_prof_averaged = np.sum(np.outer(horizontal_gaussian, vertical_gaussian))

    # Normalize by the locally available Gaussian weights so missing samples do
    # not reduce the average. The 2-D Gaussian is separable, hence two 1-D
    # convolutions give the same result at a much lower cost.
    valid = ~np.isnan(ab2)
    weighted_signal = np.where(valid, ab2, 0.0)
    numerator = convolve1d(
        weighted_signal,
        horizontal_gaussian,
        axis=0,
        mode="constant",
        cval=0.0,
    )
    numerator = convolve1d(
        numerator,
        vertical_gaussian,
        axis=1,
        mode="constant",
        cval=0.0,
    )
    denominator = convolve1d(
        valid.astype(float),
        horizontal_gaussian,
        axis=0,
        mode="constant",
        cval=0.0,
    )
    denominator = convolve1d(
        denominator,
        vertical_gaussian,
        axis=1,
        mode="constant",
        cval=0.0,
    )

    special = (
        (feature == FLAG_FA)
        | (feature == FLAG_AFA)
        | (feature == FLAG_LIKELY_ARTIFACT)
        | (feature == FLAG_SMALL_STRIPS)
    )
    eligible = (valid | special) & (denominator != 0)
    if h_nside:
        eligible[:h_nside, :] = False
        eligible[-h_nside:, :] = False
    if v_nside:
        eligible[:, :v_nside] = False
        eligible[:, -v_nside:] = False
    # Missing (NaN) where not eligible
    np.divide(numerator, denominator, out=new_ab, where=eligible)

    # Adapt SR threshold
    ab_sigma = ab_sigma / np.sqrt(nb_prof_averaged)

    return new_ab, ab_sigma
