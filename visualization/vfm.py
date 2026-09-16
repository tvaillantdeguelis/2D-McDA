"""Unfold the CALIOP Level 2 Vertical Feature Mask onto its single-shot grid.

Viewer-only: 2D-McDA processes the Level 1 product and never reads the VFM, so
this lives next to the notebook that displays the VFM alongside the 2D-McDA
masks rather than in the processing package.

The VFM packs one 5 km record into a single row, at three horizontal
resolutions depending on the altitude region: 5 km above 20.2 km (region 4),
1.667 km between 8.2 and 20.2 km (region 3), and 333 m below 8.2 km
(region 2). Unfolding repeats each value over the single-shot profiles it
covers, so that every region shares the same 333 m horizontal grid.
"""

import numpy as np

from twod_mcda.caliop.constants import (
    N_333M_BINS_PER_BIN_R3,
    N_333M_BINS_PER_BIN_R4,
    N_BINS_R2,
    N_BINS_R3,
    N_BINS_R4,
    N_LASER_PULSES_PER_5km,
)

#: Vertical bins of one VFM profile: regions R1 and R5 are not in the VFM.
NUMBER_OF_VFM_VERTICAL_BINS = N_BINS_R2 + N_BINS_R3 + N_BINS_R4

#: Profiles packed in one 5 km VFM record, per altitude region.
N_PROFILES_VFM_R4 = 3
N_PROFILES_VFM_R3 = 5
N_PROFILES_VFM_R2 = 15


def unfold_vfm(vfm):
    """
    Unfold the VFM onto the regular single-shot grid with 545 levels.

    :param vfm: folded VFM, one row per 5 km record
    :return: unfolded VFM, one row per 333 m profile
    """
    # Number of VFM masks in the file
    nb_vfm = vfm.shape[0]

    # Initialization
    vfm_unfolded = np.zeros(
        (nb_vfm * N_LASER_PULSES_PER_5km, NUMBER_OF_VFM_VERTICAL_BINS), dtype="uint16"
    )
    r4_index_range = (0, 0 + N_BINS_R4)  # Regions R1 and R5 are not in VFM
    r3_index_range = (r4_index_range[1], r4_index_range[1] + N_BINS_R3)
    r2_index_range = (r3_index_range[1], r3_index_range[1] + N_BINS_R2)

    # Loop on each VFM mask in the file
    i_vfm = 0
    while i_vfm < nb_vfm:
        # 20.2 to 30.1 km
        for i in np.arange(N_PROFILES_VFM_R4):
            start = i_vfm * N_LASER_PULSES_PER_5km + i * N_333M_BINS_PER_BIN_R4
            vfm_unfolded[
                start : start + N_333M_BINS_PER_BIN_R4,
                r4_index_range[0] : r4_index_range[1],
            ] = vfm[i_vfm, i * N_BINS_R4 : (i + 1) * N_BINS_R4]
        # 8.2 to 20.2 km
        for i in np.arange(N_PROFILES_VFM_R3):
            start = i_vfm * N_LASER_PULSES_PER_5km + i * N_333M_BINS_PER_BIN_R3
            index_first_bin_r3 = N_PROFILES_VFM_R4 * N_BINS_R4
            vfm_unfolded[
                start : start + N_333M_BINS_PER_BIN_R3,
                r3_index_range[0] : r3_index_range[1],
            ] = vfm[
                i_vfm,
                index_first_bin_r3
                + i * N_BINS_R3 : index_first_bin_r3
                + (i + 1) * N_BINS_R3,
            ]
        # -0.5 to 8.2 km
        for i in np.arange(N_PROFILES_VFM_R2):
            index_first_bin_r2 = (
                N_PROFILES_VFM_R4 * N_BINS_R4 + N_PROFILES_VFM_R3 * N_BINS_R3
            )
            vfm_unfolded[
                i_vfm * N_LASER_PULSES_PER_5km + i,
                r2_index_range[0] : r2_index_range[1],
            ] = vfm[
                i_vfm,
                index_first_bin_r2
                + i * N_BINS_R2 : index_first_bin_r2
                + (i + 1) * N_BINS_R2,
            ]
        i_vfm += 1

    return vfm_unfolded
