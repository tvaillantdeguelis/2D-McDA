"""Write detection flags into the mask from direct, per-profile rules.

Every function takes the current mask and returns a new one with one more flag
painted on: the surface, the pixels above threshold, the fully attenuated
column below a layer, and so on. None of them look at pixel neighborhoods;
that is ``morphology``.
"""

import numpy as np

from twod_mcda.parameters import (
    FLAG_AFA,
    FLAG_FA,
    FLAG_LIKELY_ARTIFACT,
    FLAG_MAYBE,
    FLAG_NOTHING,
    FLAG_SMALL_STRIPS,
    FLAG_SURFACE,
)


def apply_surface_detection(feature, surf_alt_index):
    """Put FLAG_SURFACE where and below the surface was detected"""

    # Initialization
    nb_prof = feature.shape[0]
    new_feature = np.ma.copy(feature)

    # Loop on profiles
    for i in np.arange(nb_prof):
        # If surface detected
        if surf_alt_index[i] != 999:
            # Put flag from lowest bin to surface altitude
            new_feature[i, : surf_alt_index[i] + 1] = FLAG_SURFACE

    return new_feature

def apply_threshold(k, feature, sr, sr_sigma, where_FA=False):
    """Put FLAG_MAYBE where signal above threshold"""

    # Initialization
    new_feature = np.ma.copy(feature)

    # Define threshold
    sr_maybe = 1 + k * sr_sigma

    if where_FA:
        # Put flag where ATSR > threshold and where ATSR is not masked
        new_feature[np.ma.where(sr > sr_maybe)] = FLAG_MAYBE
    else:
        # Put flag where ATSR > threshold and where feature is still "nothing"
        new_feature[np.ma.where((sr > sr_maybe) & (new_feature == FLAG_NOTHING))] = (
            FLAG_MAYBE
        )

    return new_feature

def fill_likely_artifact(params, feature, FLAG_VERY_HIGH_ECHO):
    """Put flag "Likely artifact" below high signal points"""

    # Initialization
    nb_alt = feature.shape[1]
    nb_prof = feature.shape[0]
    new_feature = np.ma.copy(feature)

    # Loop on profiles
    for i in range(nb_prof):
        # From bottom to top
        for j in np.arange(nb_alt):
            # Look for FLAG_VERY_HIGH_ECHO
            if new_feature[i, j] == FLAG_VERY_HIGH_ECHO:
                # If FLAG_VERY_HIGH_ECHO at the very bottom
                if j == 0:
                    # Nothing to flag below
                    continue
                # If bin below is FLAG_VERY_HIGH_ECHO
                elif new_feature[i, j - 1] == FLAG_VERY_HIGH_ECHO:
                    # Same layer, already done
                    continue
                # Else, flag below on the params.nb_bins_PMT_artifact extent
                else:
                    # Go down
                    j2 = j - 1
                    while (
                        (j2 >= 0)
                        & (j - j2 <= params.nb_bins_PMT_artifact)
                        & (new_feature[i, j2] == FLAG_NOTHING)
                    ):
                        new_feature[i, j2] = FLAG_LIKELY_ARTIFACT
                        j2 -= 1

    return new_feature

def fill_fully_attenuated(feature):
    """Fill with flag 'Fully Attenuated' from lowest altitude to first feature"""

    # Initialization
    nb_prof = feature.shape[0]
    nb_alt = feature.shape[1]
    new_feature = np.ma.copy(feature)

    # Loop on profiles
    for i in np.arange(nb_prof):
        # If surface detected
        if new_feature[i, 0] == FLAG_SURFACE:
            # No 'Fully Attenuated' here
            continue
        # If surface not detected
        else:
            # From lowest altitude go up until reaching a layer
            j = 0
            # While layer not reached
            while (
                (new_feature[i, j] == FLAG_NOTHING)
                | (new_feature[i, j] == FLAG_LIKELY_ARTIFACT)
            ) & (j < nb_alt):
                # Flag 'Fully Attenuated'
                new_feature[i, j] = FLAG_FA
                j += 1
                # If reach top (30.1 km)
                if j >= nb_alt:
                    # Remove all FA in the profile
                    new_feature[i, :] = FLAG_NOTHING
                    # And stop
                    break

    return new_feature

def FLAG_WEAK_SIGNAL(params, feature, sr, sr_sigma):
    """Flag where, between detected layers, more than ratio_nb are below
    sr_thresold"""

    # Initialization
    sr_thresold = sr_sigma * params.weak_signal_ratio_threshold
    nb_prof = feature.shape[0]
    nb_alt = feature.shape[1]
    new_feature = np.ma.copy(feature)

    # Loop on profiles
    for i in np.arange(nb_prof):
        # From lowest altitude go up
        nb_below = 0  # nb ranges below threshold
        nb_tot = 0  # total nb ranges in the region between 2 layers
        j = 0
        # While top not reached
        while j < nb_alt:
            if new_feature[i, j] != FLAG_NOTHING:
                j += 1  # not yet in region with no detection
                continue
            cs_min_index = j  # min index of the "CS" region
            while new_feature[i, j] == FLAG_NOTHING:
                nb_tot += 1
                if sr[i, j] < sr_thresold[i, j]:
                    nb_below += 1
                j += 1
                if j >= nb_alt:  # if reach top of column
                    break
            cs_max_index = j - 1  # max index of the "CS" region
            # If fraction_nb_below_threshold below limit put flag in this region
            if nb_below / nb_tot > params.weak_signal_ratio:
                if cs_max_index < nb_alt - 1:  # not if no layer above
                    new_feature[i, cs_min_index : cs_max_index + 1] = FLAG_AFA
            nb_below = 0
            nb_tot = 0

    return new_feature

def reput_low_confidence_flags(feature, feature_before_av):
    """Reput all not confident flags where overwritten during averaging"""

    # Initialization
    new_feature = np.copy(feature)

    # Reput FA and AFA where they were
    new_feature[feature_before_av == FLAG_FA] = FLAG_FA
    new_feature[feature_before_av == FLAG_AFA] = FLAG_AFA
    new_feature[feature_before_av == FLAG_SMALL_STRIPS] = FLAG_SMALL_STRIPS
    new_feature[feature_before_av == FLAG_LIKELY_ARTIFACT] = FLAG_LIKELY_ARTIFACT

    return new_feature
