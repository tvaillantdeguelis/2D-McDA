"""Derived CALIOP variables needed by the detection algorithm."""

import numpy as np
from scipy.interpolate import interp1d

from twod_mcda.caliop.physics import (
    compute_ab_mol_and_b_mol,
    compute_backgroundnoise,
    compute_par_ab532,
    compute_shotnoise,
    nsf_from_V_domain_to_betap_domain,
    range_from_altitude,
    rms_from_P_domain_to_betap_domain,
)
from twod_mcda.caliop.constants import (
    FILL_VALUE_FLOAT,
    get_caliop_correction_function,
    get_nb_pixels,
)


class CALIOPDerivedVariablesMixin:
    """Compute arrays derived from native CALIOP Level 1 variables."""

    def _get_par_ab532(self, do_fillvalue):
        tot_ab_532 = self.data_reader.get_data("Total_Attenuated_Backscatter_532", self.prof_min,
                                               self.prof_max, 'profindex', do_fillvalue)
        per_ab_532 = self.data_reader.get_data("Perpendicular_Attenuated_Backscatter_532",
                                                self.prof_min, self.prof_max, 'profindex',
                                                do_fillvalue)
        par_ab_532 = compute_par_ab532(tot_ab_532, per_ab_532)
        return par_ab_532
    
    def _get_molecular_profiles(self, wl, polar, do_fillvalue):
        mol_nd = self.data_reader.get_data("Molecular_Number_Density", self.prof_min,
                                           self.prof_max, 'profindex', do_fillvalue)
        O3_nd = self.data_reader.get_data("Ozone_Number_Density", self.prof_min,
                                          self.prof_max, 'profindex', do_fillvalue)
        alt = self.data_reader.get_data("Lidar_Data_Altitudes", self.prof_min,
                                        self.prof_max, 'profindex', do_fillvalue)
        met_alt = self.data_reader.get_data("Met_Data_Altitudes", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)
        if self._molecular_profiles[str(wl)+polar] is None:
            self._molecular_profiles[str(wl)+polar] = compute_ab_mol_and_b_mol(mol_nd, O3_nd, alt,
                                                                               met_alt, wl, polar)
        return self._molecular_profiles[str(wl)+polar]
    
    def _get_nsf_in_ab_domain(self, wl, polar, do_fillvalue):
        sat_alt = self.data_reader.get_data("Spacecraft_Altitude", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        caliop_lidar_tilt = self.data_reader.get_data("Off_Nadir_Angle", self.prof_min,
                                                      self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        data_alt = self.data_reader.get_data("Lidar_Data_Altitudes", self.prof_min,
                                             self.prof_max, 'profindex', do_fillvalue)[np.newaxis,:]
        range_alt = range_from_altitude(sat_alt, data_alt, caliop_lidar_tilt)
        pgr = np.array((1,))
        if wl == 532:
            calibration_cst = self.data_reader.get_data("Calibration_Constant_532", self.prof_min,
                                                        self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            laser_energy = self.data_reader.get_data("Laser_Energy_532", self.prof_min,
                                                     self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            if polar == 'par':
                nsf = self.data_reader.get_data("Noise_Scale_Factor_532_Parallel", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            if polar == 'per':
                nsf = self.data_reader.get_data("Noise_Scale_Factor_532_Perpendicular", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
                pgr = self.data_reader.get_data("Depolarization_Gain_Ratio_532", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        elif wl == 1064:
            calibration_cst = self.data_reader.get_data("Calibration_Constant_1064", self.prof_min,
                                                        self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            laser_energy = self.data_reader.get_data("Laser_Energy_1064", self.prof_min,
                                                     self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            nsf = self.data_reader.get_data("Noise_Scale_Factor_1064", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        else:
            raise Exception(f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n")
        return nsf_from_V_domain_to_betap_domain(nsf, range_alt, laser_energy, calibration_cst, pgr)
    
    def _get_rms_in_ab_domain(self, wl, polar, do_fillvalue):
        sat_alt = self.data_reader.get_data("Spacecraft_Altitude", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        caliop_lidar_tilt = self.data_reader.get_data("Off_Nadir_Angle", self.prof_min,
                                         self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        data_alt = self.data_reader.get_data("Lidar_Data_Altitudes", self.prof_min,
                                             self.prof_max, 'profindex', do_fillvalue)[np.newaxis,:]
        range_alt = range_from_altitude(sat_alt, data_alt, caliop_lidar_tilt)
        pgr = np.array((1,))
        if wl == 532:
            calibration_cst = self.data_reader.get_data("Calibration_Constant_532", self.prof_min,
                                                        self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            laser_energy = self.data_reader.get_data("Laser_Energy_532", self.prof_min,
                                                     self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            if polar == 'par':
                rms = self.data_reader.get_data("Parallel_RMS_Baseline_532", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
                gain = self.data_reader.get_data("Parallel_Amplifier_Gain_532", self.prof_min,
                                                 self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            if polar == 'per':
                rms = self.data_reader.get_data("Perpendicular_RMS_Baseline_532", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
                gain = self.data_reader.get_data("Perpendicular_Amplifier_Gain_532", self.prof_min,
                                                 self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
                pgr = self.data_reader.get_data("Depolarization_Gain_Ratio_532", self.prof_min,
                                                self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        elif wl == 1064:
            calibration_cst = self.data_reader.get_data("Calibration_Constant_1064", self.prof_min,
                                                        self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            laser_energy = self.data_reader.get_data("Laser_Energy_1064", self.prof_min,
                                                     self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            rms = self.data_reader.get_data("RMS_Baseline_1064", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
            gain = self.data_reader.get_data("Amplifier_Gain_1064", self.prof_min,
                                             self.prof_max, 'profindex', do_fillvalue)[:, np.newaxis]
        else:
            raise Exception(f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n")
        return rms_from_P_domain_to_betap_domain(rms, range_alt, laser_energy, gain, calibration_cst, pgr)
    
    def _get_shotnoise(self, wl, polar, do_fillvalue):
        nb_bins_shift = self.data_reader.get_data("Number_Bins_Shift", self.prof_min,
                                                  self.prof_max, 'profindex', do_fillvalue)
        nb_bins_shift_abs = np.squeeze(np.abs(nb_bins_shift))
        nsf = self._get_nsf_in_ab_domain(wl, polar, do_fillvalue)
        mol_ab, _ = self._get_molecular_profiles(wl, polar, do_fillvalue)
        fcorr = get_caliop_correction_function(wl)
        nb_pixels = get_nb_pixels(wl)
        return compute_shotnoise(fcorr, nb_bins_shift_abs, nb_pixels, nsf, mol_ab)
    
    def _get_backgroundnoise(self, wl, polar, do_fillvalue):
        nb_bins_shift = self.data_reader.get_data("Number_Bins_Shift", self.prof_min,
                                                  self.prof_max, 'profindex', do_fillvalue)
        nb_bins_shift_abs = np.squeeze(np.abs(nb_bins_shift))
        rms = self._get_rms_in_ab_domain(wl, polar, do_fillvalue)
        mol_ab, _ = self._get_molecular_profiles(wl, polar, do_fillvalue)
        fcorr = get_caliop_correction_function(wl)
        nb_pixels = get_nb_pixels(wl)
        return compute_backgroundnoise(fcorr, nb_bins_shift_abs, nb_pixels, rms, mol_ab)
    
    def _interp_temperature(self, met_temp, do_fillvalue):
        met_alt = self.data_reader.get_data("Met_Data_Altitudes", self.prof_min,
                                            self.prof_max, 'profindex', do_fillvalue)
        alt = self.data_reader.get_data("Lidar_Data_Altitudes", self.prof_min, self.prof_max,
                                        'profindex', do_fillvalue).filled(FILL_VALUE_FLOAT)
        # Replace last filled value (subsurface) with surface value (last not filled value)
        last_notmasked_index = np.ma.notmasked_edges(met_temp, axis=1)[1]
        for i in range(last_notmasked_index[0].size):
            met_temp[i, last_notmasked_index[1][i] + 1 :] = met_temp[i, last_notmasked_index[1][i]]
        met_temp = met_temp.filled(0) # Put 0 where still filled value (hopefully not) because it can't be mask array for interp
        # Interpolate to get temperature values for all lidar data alt
        f = interp1d(met_alt, met_temp)
        temp = f(alt)
        return temp
