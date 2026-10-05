"""Average the lidar signal, and adjust its noise threshold accordingly.

Each function here consumes the signal and returns it together with the updated
standard deviation, because averaging n samples divides the noise by sqrt(n).
"""

import numpy as np
from scipy.ndimage import convolve1d

from twod_mcda.caliop.constants import (
    N_30M_BINS_PER_BIN_R4,
    N_LASER_PULSES_PER_5km,
)
from twod_mcda.caliop.grids import (
    first_regular_30m_level_of_region_4,
    regular_30m_grid_native_sample_sizes,
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

def average_to_5km_180m(sr, sr_sigma, first_profile, channel):
    """Average the whole profile to 5 km × 180 m, the coarsest CALIOP resolution.

    ``sr`` and ``sr_sigma`` are indexed (profile, altitude), altitude from
    bottom to top; ``first_profile`` is the index, in its granule, of the first
    profile, which aligns the averaging on the granule's 5 km frames.

    The horizontal blocks hold ``N_LASER_PULSES_PER_5km`` profiles and are
    aligned on those frames, so that a block never straddles two of the 5 km
    frames the Level 1 product is built from. The vertical blocks hold 180 m and
    are anchored on the native bins of region 4, which CALIOP already downlinks
    at 180 m: a block there covers exactly one native bin instead of mixing two.
    """

    # Initialization
    new_sr = sr.copy()
    nb_prof, nb_bins = sr.shape
    block_h = N_LASER_PULSES_PER_5km  # 15 profiles of 333 m = 5 km
    block_v = N_30M_BINS_PER_BIN_R4  # 6 bins of 30 m = 180 m

    # A granule holds whole 5 km frames, so its frames start at the profile
    # indexes that are multiples of 15
    offset_h = -first_profile % block_h

    # Average 180 m × 5 km; the profiles left over at both ends are not averaged
    nb_blocks_h = (nb_prof - offset_h) // block_h
    rows = slice(offset_h, offset_h + block_h * nb_blocks_h)
    for start, stop in _vertical_blocks(nb_bins, block_v):
        new_sr[rows, start:stop] = _mean_by_block(
            sr[rows, start:stop], block_h, stop - start
        )

    # Keep missing where was already missing
    new_sr[np.isnan(sr)] = np.nan

    # Adapt SR threshold where averaged; only the native samples of a block are
    # independent, and how many it holds depends on the altitude
    sr_sigma = sr_sigma.copy()
    nb_averaged = _nb_independent_samples(channel, block_h, block_v, nb_bins)
    sr_sigma[rows, :] = sr_sigma[rows, :] / np.sqrt(nb_averaged)

    return new_sr, sr_sigma

def _vertical_blocks(nb_bins, block_v):
    """Yield the (start, stop) bounds of every 180 m block of a profile.

    The blocks are anchored on the native 180 m bins of region 4, so the levels
    left over below the first and above the last whole block form a shorter
    block of their own rather than being dropped.
    """

    offset_v = min(first_regular_30m_level_of_region_4() % block_v, nb_bins)
    nb_blocks_v = max((nb_bins - offset_v) // block_v, 0)
    paired_end = offset_v + block_v * nb_blocks_v

    if offset_v:
        yield 0, offset_v
    for start in range(offset_v, paired_end, block_v):
        yield start, start + block_v
    if nb_bins > paired_end:
        yield paired_end, nb_bins

def _nb_independent_samples(channel, block_h, block_v, nb_bins):
    """Number of independent native samples inside one averaging block.

    Returns one value per altitude level, as a block holds as many native
    samples as the downlinked resolution allows at that altitude: 90 below
    8.2 km at 532 nm, 45 there at 1064 nm (downlinked at 60 m), 15 between
    8.2 km and 20.2 km, 3 between 20.2 km and 30.1 km, and 1 above, where one
    native sample already spans more than 5 km × 180 m.
    """

    wl = 1064 if channel == "1064" else 532
    native_v, native_h = regular_30m_grid_native_sample_sizes(wl)
    nb_bins_averaged = np.maximum(block_v // native_v[:nb_bins], 1)
    nb_shots_averaged = np.maximum(block_h // native_h[:nb_bins], 1)

    return nb_bins_averaged * nb_shots_averaged

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
    height_window,
    vertical_gauss_sigma,
    width_window,
    horizontal_gauss_sigma,
    ab_signal,
    feature,
    ab_sigma,
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
