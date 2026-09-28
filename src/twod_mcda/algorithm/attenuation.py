"""Attenuation correction applied between feature-detection levels."""

import numpy as np
from numba import jit

from twod_mcda.parameters import FLAG_FA, FLAG_NOTHING, FLAG_SURFACE


@jit(nopython=True)
def transmission_correction_jit(
    new_sr,
    sr_init,
    b_mol,
    feature,
    temperature,
    twoway_transmittance_array,
    twoway_transmittance_limit,
    temp_ice_liquid,
    S_ice,
    S_liquid,
    mult_scatt,
):
    """Part extracted from transmission_correction function for faster
    processing with @jit"""

    nb_prof = new_sr.shape[0]
    nb_alt = new_sr.shape[1]

    # Loop on profiles
    for i in range(nb_prof):

        # Initialization
        int_atb = 0.0
        twoway_transmittance = 1.0
        twoway_transmittance_current_layer = 1.0
        reenter_nothing = True

        # From highest altitude go down
        for j in range(nb_alt - 1, -1, -1):

            if feature[i, j] == FLAG_NOTHING:

                # Add (multiply) transmittance of the layer to those already detected
                if reenter_nothing:
                    twoway_transmittance *= twoway_transmittance_current_layer
                    reenter_nothing = False

                # Reinitialize for next layer
                int_atb = 0.0

                # Save in an array for plot
                twoway_transmittance_array[i, j] = twoway_transmittance

                # Don't use twoway_transmittance below limit (a missing
                # transmittance stays missing, as with the builtin max)
                if twoway_transmittance_limit > twoway_transmittance:
                    twoway_transmittance = twoway_transmittance_limit

                # Correct from transmission of layers above
                new_sr[i, j] = (
                    new_sr[i, j] / twoway_transmittance
                )  # missing value not affected

            elif (feature[i, j] == FLAG_SURFACE) or (feature[i, j] == FLAG_FA):
                break

            else:
                int_atb += (
                    (sr_init[i, j] - 1) * b_mol[i, j] * 0.030
                )  # integrate (R' -1)*beta_m
                if temperature[i, j] < temp_ice_liquid:  # at cloud base
                    S = S_ice
                else:
                    S = S_liquid
                twoway_transmittance_current_layer = (
                    1 - 2 * mult_scatt * int_atb * S
                )
                reenter_nothing = True

    return new_sr, twoway_transmittance_array


def transmission_correction(sr, sr_init, b_mol, feature, temperature, params):
    """Correct sr signal below feature from transmittance"""

    # Initialization; a missing value (NaN) met in a layer makes the
    # transmittance, hence the corrected signal below, missing too
    return transmission_correction_jit(
        sr.copy(),
        sr_init,
        b_mol,
        feature,
        temperature,
        np.full(feature.shape, np.nan),
        params.twoway_transmittance_limit,
        params.temp_ice_liquid,
        params.S_ice,
        params.S_liquid,
        params.mult_scatt,
    )
