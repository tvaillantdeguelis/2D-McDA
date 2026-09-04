"""Merge the three channel-specific feature masks."""

#!/usr/bin/env python
# coding: utf8

from datetime import datetime
import numpy as np
import xarray as xr

from twod_mcda.caliop.xarray_utils import as_masked_array

# Global variables
flag_nothing = 0
# flag_detection_level = 1, 2,... for each level of detection
max_flag_detect = 5
flag_surface = 254
flag_likely_artifact = 253
flag_FA = 252
flag_AFA = 251
flag_small_strips = 250

flag_detect = 1  # will replace all flag_detection_level
flag_low_confidence = 249  # will replace flag_likely_artifact, flag_AFA, and
# flag_small_strips


def change_detection_values(mask):
    """Put all 'flag_detection_level' = 1, 2,... to 'flag_detect'
    Put 'flag_likely_artifact', 'flag_AFA', and 'flag_small_strips' to
    'flag_low_confidence'"""

    new_mask = np.ma.copy(mask)

    # Put 'flag_detect'
    new_mask[(new_mask >= 1) & (new_mask <= max_flag_detect)] = flag_detect

    # Put 'flag_low_confidence'
    new_mask[
        (new_mask == flag_likely_artifact)
        | (new_mask == flag_AFA)
        | (new_mask == flag_small_strips)
    ] = flag_low_confidence

    return new_mask


# **********************************************************************
# MAIN FUNCTION
# **********************************************************************
def merged_feature_masks(mask_532_par, mask_532_per, mask_1064):
    """Merged the feature masks from the 3 channels"""

    tic_function = datetime.now()
    template = mask_532_par if isinstance(mask_532_par, xr.DataArray) else None
    mask_532_par = as_masked_array(mask_532_par)
    mask_532_per = as_masked_array(mask_532_per)
    mask_1064 = as_masked_array(mask_1064)

    # Check if flag values not declared in global variables
    if not np.all(
        (mask_532_par <= max_flag_detect) | (mask_532_par >= flag_small_strips)
    ):
        raise ValueError("mask_532_par has values not declared in global variables")
    if not np.all(
        (mask_532_per <= max_flag_detect) | (mask_532_per >= flag_small_strips)
    ):
        raise ValueError("mask_532_per has values not declared in global variables")
    if not np.all((mask_1064 <= max_flag_detect) | (mask_1064 >= flag_small_strips)):
        raise ValueError("mask_1064 has values not declared in global variables")

    #################################
    #### Create a composite mask ####
    print("\t=> Create a composite mask from the 3 channel " "feature masks...")

    # Put 'flag_detect' and 'flag_low_confidence'
    flag_532_par = change_detection_values(mask_532_par)
    flag_532_per = change_detection_values(mask_532_per)
    flag_1064 = change_detection_values(mask_1064)

    # --------------------------------------------------------------------------
    # Bits 1–3: Classification
    # Note: the order in which to write the classification is important
    # e.g.: 'surface' will overwrite 'surface' above 'detect'

    # 0: invalid (bad or missing data) (initialization)
    merged_mask = np.zeros(mask_532_par.shape, dtype=np.uint8)

    # 3: low confidence
    merged_mask[
        (flag_532_par == flag_low_confidence)
        | (flag_532_per == flag_low_confidence)
        | (flag_1064 == flag_low_confidence)
    ] = 3

    # 1: "clear air"
    merged_mask[
        (flag_532_par == flag_nothing)
        | (flag_532_per == flag_nothing)
        | (flag_1064 == flag_nothing)
    ] = 1

    # 3: low confidence if "clear air" only from 532_per
    merged_mask[
        (
            (flag_532_par == flag_low_confidence)
            & (flag_532_per == flag_nothing)
            & (flag_1064 == flag_low_confidence)
        )
        | (
            (flag_532_par == flag_low_confidence)
            & (flag_532_per == flag_nothing)
            & (flag_1064 == flag_FA)
        )
        | (
            (flag_532_par == flag_FA)
            & (flag_532_per == flag_nothing)
            & (flag_1064 == flag_low_confidence)
        )
    ] = 3

    # 2: atmospheric feature
    merged_mask[
        (flag_532_par == flag_detect)
        | (flag_532_per == flag_detect)
        | (flag_1064 == flag_detect)
    ] = 2

    # 4: not used

    # 5: surface or subsurface
    merged_mask[
        (flag_532_par == flag_surface)
        | (flag_532_per == flag_surface)
        | (flag_1064 == flag_surface)
    ] = 5

    # 6: not used

    # 7: totally attenuated
    # if 532_par and 1064 FA and 532_per FA, nothing, or low_confidence
    merged_mask[
        (flag_532_par == flag_FA)
        & (flag_1064 == flag_FA)
        & (
            (flag_532_per == flag_FA)
            | (flag_532_per == flag_nothing)
            | (flag_532_per == flag_low_confidence)
        )
    ] = 7

    # --------------------------------------------------------------------------
    # Bits 4: 532 nm parallel channel detection status %1000 = 8
    merged_mask[(flag_532_par == flag_detect) | (flag_532_par == flag_surface)] += 8

    # --------------------------------------------------------------------------
    # Bits 5: 532 nm perpendicular channel detection status %10000 = 16
    merged_mask[(flag_532_per == flag_detect) | (flag_532_per == flag_surface)] += 16

    # --------------------------------------------------------------------------
    # Bits 6: 532 nm perpendicular channel detection status %100000 = 32
    merged_mask[(flag_1064 == flag_detect) | (flag_1064 == flag_surface)] += 32

    print(f"\t(Elapsed time: {datetime.now() - tic_function})")

    if template is None:
        return merged_mask
    return xr.DataArray(
        merged_mask,
        dims=template.dims,
        coords=template.coords,
        name="Composite_Detection_Flags",
    )
