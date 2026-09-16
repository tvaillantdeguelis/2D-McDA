"""Average the lidar signal, and adjust its noise threshold accordingly.

Each function here consumes the signal and returns it together with the updated
standard deviation, because averaging n samples divides the noise by sqrt(n).
"""

import numpy as np
from scipy.ndimage import convolve1d

from twod_mcda.caliop.constants import (
    FILL_VALUE_FLOAT,
    N_30M_BINS_PER_BIN_R1,
    N_30M_BINS_PER_BIN_R2,
    N_BINS_R1,
    N_BINS_R2,
)
from twod_mcda.parameters import (
    FLAG_AFA,
    FLAG_FA,
    FLAG_LIKELY_ARTIFACT,
    FLAG_NOTHING,
    FLAG_SMALL_STRIPS,
    FLAG_SURFACE,
)


def remove_detect_from_sr(sr, feature):
    """Remove detected pixel from the ATSR signal"""

    # Mask where not "nothing"
    new_sr = np.ma.masked_where(feature != FLAG_NOTHING, sr)

    return new_sr

def average_below_8_2(sr, sr_sigma):
    """Average below 8.2 km as between 8.2 km and 20.2 km (60 m × 1 km)"""

    # Initialization
    new_sr = np.ma.copy(sr)
    nb_prof = sr.shape[0]
    nb_bins_below_8_2km = (
        N_30M_BINS_PER_BIN_R1 * N_BINS_R1 + N_30M_BINS_PER_BIN_R2 * N_BINS_R2
    )

    # Look for horizontal offset if 1st profile not the start of a 1-km profile
    index_vertical_bin = 100  # random bin in the R2 region
    if sr[0, index_vertical_bin] == sr[1, index_vertical_bin]:
        if sr[1, index_vertical_bin] == sr[2, index_vertical_bin]:
            offset_h = 0
        else:
            offset_h = 2
    else:
        offset_h = 1

    # Average 60 m × 1 km (2 verticals × 3 horizontals)
    i_array = np.arange(offset_h, nb_prof - 2, 3)  # 3 horizontals
    j_array = np.arange(0, nb_bins_below_8_2km, 2)  # 2 verticals
    i_progress = 0
    for i in i_array:
        for j in j_array:
            new_sr[i : i + 3, j : j + 2] = np.ma.mean(sr[i : i + 3, j : j + 2])

    # Remask where was already masked
    new_sr.mask = np.copy(sr.mask)

    # Adapt SR threshold below 8.2 km
    sr_sigma[:nb_bins_below_8_2km] = sr_sigma[:nb_bins_below_8_2km] / np.sqrt(6)

    return new_sr, sr_sigma

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
    ab2 = np.ma.asarray(ab_signal).filled(FILL_VALUE_FLOAT).astype(float, copy=False)
    new_ab = np.full(ab_signal.shape, FILL_VALUE_FLOAT, dtype=float)
    copy_feature = np.ma.asarray(feature).filled(FLAG_SURFACE)

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

    # Normalize by the locally available Gaussian weights so masked samples do
    # not reduce the average. The 2-D Gaussian is separable, hence two 1-D
    # convolutions give the same result at a much lower cost.
    valid = ab2 != FILL_VALUE_FLOAT
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
        (copy_feature == FLAG_FA)
        | (copy_feature == FLAG_AFA)
        | (copy_feature == FLAG_LIKELY_ARTIFACT)
        | (copy_feature == FLAG_SMALL_STRIPS)
    )
    eligible = (valid | special) & (denominator != 0)
    if h_nside:
        eligible[:h_nside, :] = False
        eligible[-h_nside:, :] = False
    if v_nside:
        eligible[:, :v_nside] = False
        eligible[:, -v_nside:] = False
    np.divide(numerator, denominator, out=new_ab, where=eligible)

    # Mask where FILL_VALUE_FLOAT
    new_ab = np.ma.masked_where(new_ab == FILL_VALUE_FLOAT, new_ab)
    # Adapt SR threshold
    ab_sigma = ab_sigma / np.sqrt(nb_prof_averaged)

    return new_ab, ab_sigma
