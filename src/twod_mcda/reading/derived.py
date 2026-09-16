"""Derived CALIOP variables needed by the detection algorithm.

These are the variables CALIOP does not store: the parallel 532 nm backscatter,
the molecular model, and the noise terms. They are all computed from native
Level 1 variables read over one profile range, which is why this object is built
from a reader and its bounds rather than mixed into the reader itself.
"""

import numpy as np
from scipy.interpolate import interp1d

from twod_mcda.caliop.physics import (
    compute_ab_mol_and_b_mol,
    compute_backgroundnoise,
    compute_par_ab532,
    compute_shotnoise,
    get_caliop_correction_function,
    get_nb_pixels,
    nsf_from_V_domain_to_betap_domain,
    range_from_altitude,
    rms_from_P_domain_to_betap_domain,
)
from twod_mcda.caliop.constants import FILL_VALUE_FLOAT
from twod_mcda.utils.arrays import as_masked_array


class DerivedVariables:
    """Compute arrays derived from native CALIOP Level 1 variables.

    Bound to one profile range: ``molecular_profiles`` caches its result, so a
    new instance is needed whenever ``prof_min`` or ``prof_max`` changes.
    """

    def __init__(self, data_reader, prof_min, prof_max):
        self.data_reader = data_reader
        self.prof_min = prof_min
        self.prof_max = prof_max
        self._molecular_profiles = {
            "532": None,
            "532par": None,
            "532per": None,
            "1064": None,
        }

    def _native_values(self, key, do_fillvalue):
        data = self.data_reader.get_data(
            key,
            self.prof_min,
            self.prof_max,
            do_fillvalue,
        )
        return as_masked_array(data) if do_fillvalue else data.values

    def par_ab532(self, do_fillvalue):
        tot_ab_532 = self._native_values(
            "Total_Attenuated_Backscatter_532", do_fillvalue
        )
        per_ab_532 = self._native_values(
            "Perpendicular_Attenuated_Backscatter_532", do_fillvalue
        )
        par_ab_532 = compute_par_ab532(tot_ab_532, per_ab_532)
        return par_ab_532

    def molecular_profiles(self, wl, polar, do_fillvalue):
        mol_nd = self._native_values("Molecular_Number_Density", do_fillvalue)
        O3_nd = self._native_values("Ozone_Number_Density", do_fillvalue)
        alt = self._native_values("Lidar_Data_Altitudes", do_fillvalue)
        met_alt = self._native_values("Met_Data_Altitudes", do_fillvalue)
        if self._molecular_profiles[str(wl) + polar] is None:
            self._molecular_profiles[str(wl) + polar] = compute_ab_mol_and_b_mol(
                mol_nd, O3_nd, alt, met_alt, wl, polar
            )
        return self._molecular_profiles[str(wl) + polar]

    def nsf_in_ab_domain(self, wl, polar, do_fillvalue):
        sat_alt = self._native_values("Spacecraft_Altitude", do_fillvalue)[
            :, np.newaxis
        ]
        caliop_lidar_tilt = self._native_values("Off_Nadir_Angle", do_fillvalue)[
            :, np.newaxis
        ]
        data_alt = self._native_values("Lidar_Data_Altitudes", do_fillvalue)[
            np.newaxis, :
        ]
        range_alt = range_from_altitude(sat_alt, data_alt, caliop_lidar_tilt)
        pgr = np.array((1,))
        if wl == 532:
            calibration_cst = self._native_values(
                "Calibration_Constant_532", do_fillvalue
            )[:, np.newaxis]
            laser_energy = self._native_values("Laser_Energy_532", do_fillvalue)[
                :, np.newaxis
            ]
            if polar == "par":
                nsf = self._native_values(
                    "Noise_Scale_Factor_532_Parallel", do_fillvalue
                )[:, np.newaxis]
            if polar == "per":
                nsf = self._native_values(
                    "Noise_Scale_Factor_532_Perpendicular", do_fillvalue
                )[:, np.newaxis]
                pgr = self._native_values(
                    "Depolarization_Gain_Ratio_532", do_fillvalue
                )[:, np.newaxis]
        elif wl == 1064:
            calibration_cst = self._native_values(
                "Calibration_Constant_1064", do_fillvalue
            )[:, np.newaxis]
            laser_energy = self._native_values("Laser_Energy_1064", do_fillvalue)[
                :, np.newaxis
            ]
            nsf = self._native_values("Noise_Scale_Factor_1064", do_fillvalue)[
                :, np.newaxis
            ]
        else:
            raise Exception(
                f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n"
            )
        return nsf_from_V_domain_to_betap_domain(
            nsf, range_alt, laser_energy, calibration_cst, pgr
        )

    def rms_in_ab_domain(self, wl, polar, do_fillvalue):
        sat_alt = self._native_values("Spacecraft_Altitude", do_fillvalue)[
            :, np.newaxis
        ]
        caliop_lidar_tilt = self._native_values("Off_Nadir_Angle", do_fillvalue)[
            :, np.newaxis
        ]
        data_alt = self._native_values("Lidar_Data_Altitudes", do_fillvalue)[
            np.newaxis, :
        ]
        range_alt = range_from_altitude(sat_alt, data_alt, caliop_lidar_tilt)
        pgr = np.array((1,))
        if wl == 532:
            calibration_cst = self._native_values(
                "Calibration_Constant_532", do_fillvalue
            )[:, np.newaxis]
            laser_energy = self._native_values("Laser_Energy_532", do_fillvalue)[
                :, np.newaxis
            ]
            if polar == "par":
                rms = self._native_values(
                    "Parallel_RMS_Baseline_532", do_fillvalue
                )[:, np.newaxis]
                gain = self._native_values(
                    "Parallel_Amplifier_Gain_532", do_fillvalue
                )[:, np.newaxis]
            if polar == "per":
                rms = self._native_values(
                    "Perpendicular_RMS_Baseline_532", do_fillvalue
                )[:, np.newaxis]
                gain = self._native_values(
                    "Perpendicular_Amplifier_Gain_532", do_fillvalue
                )[:, np.newaxis]
                pgr = self._native_values(
                    "Depolarization_Gain_Ratio_532", do_fillvalue
                )[:, np.newaxis]
        elif wl == 1064:
            calibration_cst = self._native_values(
                "Calibration_Constant_1064", do_fillvalue
            )[:, np.newaxis]
            laser_energy = self._native_values("Laser_Energy_1064", do_fillvalue)[
                :, np.newaxis
            ]
            rms = self._native_values("RMS_Baseline_1064", do_fillvalue)[
                :, np.newaxis
            ]
            gain = self._native_values("Amplifier_Gain_1064", do_fillvalue)[
                :, np.newaxis
            ]
        else:
            raise Exception(
                f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n"
            )
        return rms_from_P_domain_to_betap_domain(
            rms, range_alt, laser_energy, gain, calibration_cst, pgr
        )

    def shotnoise(self, wl, polar, do_fillvalue):
        nb_bins_shift = self._native_values("Number_Bins_Shift", do_fillvalue)
        nb_bins_shift_abs = np.squeeze(np.abs(nb_bins_shift))
        nsf = self.nsf_in_ab_domain(wl, polar, do_fillvalue)
        mol_ab, _ = self.molecular_profiles(wl, polar, do_fillvalue)
        fcorr = get_caliop_correction_function(wl)
        nb_pixels = get_nb_pixels(wl)
        return compute_shotnoise(fcorr, nb_bins_shift_abs, nb_pixels, nsf, mol_ab)

    def backgroundnoise(self, wl, polar, do_fillvalue):
        nb_bins_shift = self._native_values("Number_Bins_Shift", do_fillvalue)
        nb_bins_shift_abs = np.squeeze(np.abs(nb_bins_shift))
        rms = self.rms_in_ab_domain(wl, polar, do_fillvalue)
        mol_ab, _ = self.molecular_profiles(wl, polar, do_fillvalue)
        fcorr = get_caliop_correction_function(wl)
        nb_pixels = get_nb_pixels(wl)
        return compute_backgroundnoise(fcorr, nb_bins_shift_abs, nb_pixels, rms, mol_ab)

    def interp_temperature(self, met_temp, do_fillvalue):
        met_alt = self._native_values("Met_Data_Altitudes", do_fillvalue)
        alt = np.ma.asarray(
            self._native_values("Lidar_Data_Altitudes", do_fillvalue)
        ).filled(FILL_VALUE_FLOAT)
        met_temp = as_masked_array(met_temp)
        # Replace last filled value (subsurface) with surface value (last not filled value)
        last_notmasked_index = np.ma.notmasked_edges(met_temp, axis=1)[1]
        for i in range(last_notmasked_index[0].size):
            met_temp[i, last_notmasked_index[1][i] + 1 :] = met_temp[
                i, last_notmasked_index[1][i]
            ]
        met_temp = met_temp.filled(
            0
        )  # Put 0 where still filled value (hopefully not) because it can't be mask array for interp
        # Interpolate to get temperature values for all lidar data alt
        f = interp1d(met_alt, met_temp)
        temp = f(alt)
        return temp
