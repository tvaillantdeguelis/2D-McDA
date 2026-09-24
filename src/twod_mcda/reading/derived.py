"""Derived CALIOP variables needed by the detection algorithm.

These are the variables CALIOP does not store: the parallel 532 nm backscatter,
the molecular model, and the noise terms. They are all computed from native
Level 1 variables read over one profile range, which is why this object is built
from a reader and its bounds rather than mixed into the reader itself.

Every variable is returned as a labelled DataArray, with NaN where missing.
"""

import numpy as np
import xarray as xr
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
from twod_mcda.caliop.constants import FILL_VALUE_FLOAT, MET_ALTITUDE_DIMENSION
from twod_mcda.utils.arrays import mask_invalid


class DerivedVariables:
    """Compute arrays derived from native CALIOP Level 1 variables.

    Bound to one profile range: ``molecular_profiles`` caches its result, so a
    new instance is needed whenever ``prof_min`` or ``prof_max`` changes.
    """

    def __init__(self, granule_file, prof_min, prof_max):
        self.granule_file = granule_file
        self.prof_min = prof_min
        self.prof_max = prof_max
        self._molecular_profiles = {
            "532": None,
            "532par": None,
            "532per": None,
            "1064": None,
        }

    def _native_data(self, key, do_fillvalue):
        """Return a native variable as a labelled DataArray, NaN where missing."""

        data = self.granule_file.get_data(
            key,
            self.prof_min,
            self.prof_max,
            do_fillvalue,
        )
        return mask_invalid(data) if do_fillvalue else data

    def par_ab532(self, do_fillvalue):
        tot_ab_532 = self._native_data("Total_Attenuated_Backscatter_532", do_fillvalue)
        per_ab_532 = self._native_data(
            "Perpendicular_Attenuated_Backscatter_532", do_fillvalue
        )
        # xarray arithmetic keeps the attributes of its first operand, which
        # describe the total backscatter, not this derived variable
        return compute_par_ab532(tot_ab_532, per_ab_532).drop_attrs()

    def molecular_profiles(self, wl, polar, do_fillvalue):
        channel = str(wl) + polar
        if self._molecular_profiles[channel] is None:
            self._molecular_profiles[channel] = compute_ab_mol_and_b_mol(
                self._native_data("Molecular_Number_Density", do_fillvalue),
                self._native_data("Ozone_Number_Density", do_fillvalue),
                self._native_data("Lidar_Data_Altitudes", do_fillvalue),
                self._native_data("Met_Data_Altitudes", do_fillvalue),
                wl,
                polar,
            )
        return self._molecular_profiles[channel]

    def _range(self, do_fillvalue):
        """Return the range from the spacecraft to every lidar altitude bin."""

        # Named dimensions broadcast (profile) against (lidar_altitude) into
        # (profile, lidar_altitude)
        return range_from_altitude(
            self._native_data("Spacecraft_Altitude", do_fillvalue),
            self._native_data("Lidar_Data_Altitudes", do_fillvalue),
            self._native_data("Off_Nadir_Angle", do_fillvalue),
        )

    def nsf_in_ab_domain(self, wl, polar, do_fillvalue):
        range_alt = self._range(do_fillvalue)
        pgr = np.array((1,))
        if wl == 532:
            calibration_cst = self._native_data("Calibration_Constant_532", do_fillvalue)
            laser_energy = self._native_data("Laser_Energy_532", do_fillvalue)
            if polar == "par":
                nsf = self._native_data("Noise_Scale_Factor_532_Parallel", do_fillvalue)
            if polar == "per":
                nsf = self._native_data(
                    "Noise_Scale_Factor_532_Perpendicular", do_fillvalue
                )
                pgr = self._native_data("Depolarization_Gain_Ratio_532", do_fillvalue)
        elif wl == 1064:
            calibration_cst = self._native_data(
                "Calibration_Constant_1064", do_fillvalue
            )
            laser_energy = self._native_data("Laser_Energy_1064", do_fillvalue)
            nsf = self._native_data("Noise_Scale_Factor_1064", do_fillvalue)
        else:
            raise Exception(
                f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n"
            )
        return nsf_from_V_domain_to_betap_domain(
            nsf, range_alt, laser_energy, calibration_cst, pgr
        ).drop_attrs()

    def rms_in_ab_domain(self, wl, polar, do_fillvalue):
        range_alt = self._range(do_fillvalue)
        pgr = np.array((1,))
        if wl == 532:
            calibration_cst = self._native_data("Calibration_Constant_532", do_fillvalue)
            laser_energy = self._native_data("Laser_Energy_532", do_fillvalue)
            if polar == "par":
                rms = self._native_data("Parallel_RMS_Baseline_532", do_fillvalue)
                gain = self._native_data("Parallel_Amplifier_Gain_532", do_fillvalue)
            if polar == "per":
                rms = self._native_data("Perpendicular_RMS_Baseline_532", do_fillvalue)
                gain = self._native_data(
                    "Perpendicular_Amplifier_Gain_532", do_fillvalue
                )
                pgr = self._native_data("Depolarization_Gain_Ratio_532", do_fillvalue)
        elif wl == 1064:
            calibration_cst = self._native_data(
                "Calibration_Constant_1064", do_fillvalue
            )
            laser_energy = self._native_data("Laser_Energy_1064", do_fillvalue)
            rms = self._native_data("RMS_Baseline_1064", do_fillvalue)
            gain = self._native_data("Amplifier_Gain_1064", do_fillvalue)
        else:
            raise Exception(
                f"Error: Unrecognized wavelength: {wl}; use 532 or 1064 instead\n\n"
            )
        return rms_from_P_domain_to_betap_domain(
            rms, range_alt, laser_energy, gain, calibration_cst, pgr
        ).drop_attrs()

    def shotnoise(self, wl, polar, do_fillvalue):
        nb_bins_shift = self._native_data("Number_Bins_Shift", do_fillvalue)
        nsf = self.nsf_in_ab_domain(wl, polar, do_fillvalue)
        mol_ab, _ = self.molecular_profiles(wl, polar, do_fillvalue)
        fcorr = get_caliop_correction_function(wl)
        nb_pixels = get_nb_pixels(wl)
        return compute_shotnoise(fcorr, abs(nb_bins_shift), nb_pixels, nsf, mol_ab)

    def backgroundnoise(self, wl, polar, do_fillvalue):
        nb_bins_shift = self._native_data("Number_Bins_Shift", do_fillvalue)
        rms = self.rms_in_ab_domain(wl, polar, do_fillvalue)
        mol_ab, _ = self.molecular_profiles(wl, polar, do_fillvalue)
        fcorr = get_caliop_correction_function(wl)
        nb_pixels = get_nb_pixels(wl)
        return compute_backgroundnoise(fcorr, abs(nb_bins_shift), nb_pixels, rms, mol_ab)

    def interp_temperature(self, do_fillvalue):
        """Return the temperature interpolated on the lidar data altitudes."""

        # The interpolation cannot handle missing values, so they are resolved
        # here whatever do_fillvalue
        met_temp = mask_invalid(self._native_data("Temperature", do_fillvalue))
        met_alt = self._native_data("Met_Data_Altitudes", do_fillvalue)
        alt = self._native_data("Lidar_Data_Altitudes", do_fillvalue)
        met_temp = _extend_surface_value(met_temp, MET_ALTITUDE_DIMENSION)
        # Put 0 where still missing (hopefully not): interp1d needs finite values
        met_temp = met_temp.fillna(0)
        # Interpolate to get temperature values for all lidar data alt
        f = interp1d(met_alt.values, met_temp.values)
        temp = f(alt.fillna(FILL_VALUE_FLOAT).values)
        profile_dim = met_temp.dims[0]
        return xr.DataArray(
            temp,
            dims=(profile_dim, *alt.dims),
            coords={profile_dim: met_temp.coords[profile_dim]},
        )


def _extend_surface_value(met_data, dim):
    """Replace the missing values below the surface with the surface value.

    The surface is the last valid level of each profile along ``dim``; missing
    values above it are left missing. Profiles without any valid value stay
    missing.
    """

    valid = met_data.notnull()
    nb_valid_so_far = valid.cumsum(dim)
    # From the surface downward, every valid value of the profile has been seen
    at_or_below_surface = nb_valid_so_far == valid.sum(dim)
    # argmax returns the first level reaching the total: the surface itself
    surface = met_data.isel({dim: nb_valid_so_far.argmax(dim)})
    return met_data.where(valid | ~at_or_below_surface, surface)
