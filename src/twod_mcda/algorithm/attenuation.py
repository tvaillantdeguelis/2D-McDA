"""Attenuation correction applied between feature-detection levels."""

import numpy as np

from twod_mcda.algorithm.flags import FLAG_FA, FLAG_NOTHING, FLAG_SURFACE
from twod_mcda.caliop.constants import FILL_VALUE_FLOAT


def transmission_correction(sr, sr_init, b_mol, feature, temperature, params):
    """Correct sr signal below feature from transmittance"""

    # Initialization
    new_sr = np.ma.copy(sr)
    twoway_transmittance_array = np.ma.ones(feature.shape) * FILL_VALUE_FLOAT
    nb_alt = new_sr.shape[1]
    nb_prof = new_sr.shape[0]

    # Loop on profiles
    for i in range(nb_prof):

        # Initialization
        int_atb = 0
        twoway_transmittance = 1
        twoway_transmittance_current_layer = 1
        reenter_nothing = True

        # From highest altitude go down
        for j in np.arange(nb_alt - 1, -1, -1):

            if feature[i, j] == FLAG_NOTHING:

                # Add (multiply) transmittance of the layer to those already detected
                if reenter_nothing:
                    twoway_transmittance *= twoway_transmittance_current_layer
                    reenter_nothing = False

                # Reinitialize for next layer
                int_atb = 0

                # Save in an array for plot
                twoway_transmittance_array[i, j] = twoway_transmittance

                # Don't use twoway_transmittance below limit
                twoway_transmittance = max(
                    twoway_transmittance, params.twoway_transmittance_limit
                )

                # Correct from transmission of layers above
                new_sr[i, j] = (
                    new_sr[i, j] / twoway_transmittance
                )  # masked value not affected

            elif (feature[i, j] == FLAG_SURFACE) | (feature[i, j] == FLAG_FA):
                break

            else:
                int_atb += (
                    (sr_init[i, j] - 1) * b_mol[i, j] * 0.030
                )  # integrate (R' -1)*beta_m
                if temperature[i, j] < params.temp_ice_liquid:  # at cloud base
                    S = params.S_ice
                else:
                    S = params.S_liquid
                twoway_transmittance_current_layer = (
                    1 - 2 * params.mult_scatt * int_atb * S
                )
                reenter_nothing = True

    # Mask where feature
    twoway_transmittance_array = np.ma.masked_where(
        twoway_transmittance_array == FILL_VALUE_FLOAT, twoway_transmittance_array
    )

    return new_sr, twoway_transmittance_array
