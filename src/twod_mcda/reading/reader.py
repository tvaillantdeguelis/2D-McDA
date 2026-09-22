"""Read CALIOP L1 products on the regular grid expected by 2D-McDA.

``CALIOPRegularGridReader`` sits on top of ``native.CALIOPGranuleFile``: it puts
every array on the regular 333 m × 30 m grid over a range of profiles, and
delegates every variable CALIOP does not store to ``derived.DerivedVariables``.

The reader knows nothing about how a granule file is named or about how a
requested subset is expressed: ``access.open_granule`` resolves both and hands
this class an open file and two profile indexes.
"""

import numpy as np
import xarray as xr

from twod_mcda.caliop.constants import (
    NUMBER_OF_VERTICAL_BINS,
    NUMBER_OF_VERTICAL_BINS_MET,
)
from twod_mcda.caliop.grids import (
    alt_to_regular_30m_vertical_grid,
    shape_to_regular_30m_vertical_grid,
)
from twod_mcda.reading.derived import DerivedVariables
from twod_mcda.utils.arrays import as_masked_array


def _molecular_attenuated_backscatter(derived, wavelength, polarization, do_fillvalue):
    """Return the molecular model in the attenuated backscatter domain."""

    attenuated_backscatter, _ = derived.molecular_profiles(
        wavelength,
        polarization,
        do_fillvalue,
    )
    return attenuated_backscatter


def _molecular_backscatter(derived, wavelength, polarization, do_fillvalue):
    """Return the molecular model in the backscatter domain."""

    _, backscatter = derived.molecular_profiles(
        wavelength,
        polarization,
        do_fillvalue,
    )
    return backscatter


def _scattering_ratio_uncertainty(derived, wavelength, polarization, do_fillvalue):
    """Combine the range-dependent and range-independent noise of one channel."""

    shot_noise = derived.shotnoise(wavelength, polarization, do_fillvalue)
    background_noise = derived.backgroundnoise(wavelength, polarization, do_fillvalue)
    return np.sqrt(shot_noise**2 + background_noise**2)


#: Variables CALIOP does not store, computed by ``derived.DerivedVariables``
#: from the native Level 1 variables of one profile range.
DERIVED_VARIABLES = {
    "Parallel_Attenuated_Backscatter_532": (
        lambda derived, do_fillvalue: derived.par_ab532(do_fillvalue)
    ),
    #### Molecular model, in the attenuated backscatter domain ####
    "Molecular_Total_Attenuated_Backscatter_532": (
        lambda derived, do_fillvalue: _molecular_attenuated_backscatter(
            derived, 532, "", do_fillvalue
        )
    ),
    "Molecular_Parallel_Attenuated_Backscatter_532": (
        lambda derived, do_fillvalue: _molecular_attenuated_backscatter(
            derived, 532, "par", do_fillvalue
        )
    ),
    "Molecular_Perpendicular_Attenuated_Backscatter_532": (
        lambda derived, do_fillvalue: _molecular_attenuated_backscatter(
            derived, 532, "per", do_fillvalue
        )
    ),
    "Molecular_Attenuated_Backscatter_1064": (
        lambda derived, do_fillvalue: _molecular_attenuated_backscatter(
            derived, 1064, "", do_fillvalue
        )
    ),
    #### Molecular model, in the backscatter domain ####
    "Molecular_Total_Backscatter_532": (
        lambda derived, do_fillvalue: _molecular_backscatter(
            derived, 532, "", do_fillvalue
        )
    ),
    "Molecular_Parallel_Backscatter_532": (
        lambda derived, do_fillvalue: _molecular_backscatter(
            derived, 532, "par", do_fillvalue
        )
    ),
    "Molecular_Perpendicular_Backscatter_532": (
        lambda derived, do_fillvalue: _molecular_backscatter(
            derived, 532, "per", do_fillvalue
        )
    ),
    "Molecular_Backscatter_1064": (
        lambda derived, do_fillvalue: _molecular_backscatter(
            derived, 1064, "", do_fillvalue
        )
    ),
    #### NSF in beta' domain (NSF in level 1B data is in V = P / Ga domain) ####
    "Noise_Scale_Factor_532_Parallel_AB_domain": (
        lambda derived, do_fillvalue: derived.nsf_in_ab_domain(532, "par", do_fillvalue)
    ),
    "Noise_Scale_Factor_532_Perpendicular_AB_domain": (
        lambda derived, do_fillvalue: derived.nsf_in_ab_domain(532, "per", do_fillvalue)
    ),
    "Noise_Scale_Factor_1064_AB_domain": (
        lambda derived, do_fillvalue: derived.nsf_in_ab_domain(1064, "", do_fillvalue)
    ),
    #### RMS in beta' domain (RMS in level 1B data is in P domain) ####
    "Parallel_RMS_Baseline_532_AB_domain": (
        lambda derived, do_fillvalue: derived.rms_in_ab_domain(532, "par", do_fillvalue)
    ),
    "Perpendicular_RMS_Baseline_532_AB_domain": (
        lambda derived, do_fillvalue: derived.rms_in_ab_domain(532, "per", do_fillvalue)
    ),
    "RMS_Baseline_1064_AB_domain": (
        lambda derived, do_fillvalue: derived.rms_in_ab_domain(1064, "", do_fillvalue)
    ),
    #### Range-dependent uncertainty based on the molecular model ####
    "Shot_Noise_532_Parallel": (
        lambda derived, do_fillvalue: derived.shotnoise(532, "par", do_fillvalue)
    ),
    "Shot_Noise_532_Perpendicular": (
        lambda derived, do_fillvalue: derived.shotnoise(532, "per", do_fillvalue)
    ),
    "Shot_Noise_1064": (
        lambda derived, do_fillvalue: derived.shotnoise(1064, "", do_fillvalue)
    ),
    #### Range-independent uncertainty from the background RMS ####
    "Background_Noise_532_Parallel": (
        lambda derived, do_fillvalue: derived.backgroundnoise(532, "par", do_fillvalue)
    ),
    "Background_Noise_532_Perpendicular": (
        lambda derived, do_fillvalue: derived.backgroundnoise(532, "per", do_fillvalue)
    ),
    "Background_Noise_1064": (
        lambda derived, do_fillvalue: derived.backgroundnoise(1064, "", do_fillvalue)
    ),
    #### Scattering ratio uncertainty standard deviation ####
    "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel": (
        lambda derived, do_fillvalue: _scattering_ratio_uncertainty(
            derived, 532, "par", do_fillvalue
        )
    ),
    "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular": (
        lambda derived, do_fillvalue: _scattering_ratio_uncertainty(
            derived, 532, "per", do_fillvalue
        )
    ),
    "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064": (
        lambda derived, do_fillvalue: _scattering_ratio_uncertainty(
            derived, 1064, "", do_fillvalue
        )
    ),
}


class CALIOPRegularGridReader:
    def __init__(
        self,
        granule_file,
        profile_start,
        profile_end,
        max_altitude_index=None,
    ):
        """
        Get original and derived CALIOP parameters on the regular 333 m × 30 m grid.

        :param granule_file: open ``native.CALIOPGranuleFile`` to read from. The
                             reader closes it, so one reader owns one file.
        :param profile_start: first profile to read (at 333 m resolution)
        :param profile_end: last profile to read (included) (at 333 m resolution)
        :param max_altitude_index: (optional) number of bins to keep at the bottom of the
                                   regular 30 m grid, which spans -1.95 to 39.93 km in 1400
                                   bins. Resolves 'processing.max_altitude_km', so that the
                                   algorithm and the output product stop at the altitude of
                                   interest instead of carrying the whole profile.
                                   default: the whole profile
        """
        self.granule_file = granule_file
        self.max_altitude_index = max_altitude_index

        self.prof_min = int(profile_start)
        self.prof_max = int(profile_end)
        self.nb_profiles = self.prof_max - self.prof_min + 1

        longitude = self.granule_file.get_data(
            "Longitude",
            self.prof_min,
            self.prof_max,
        )
        self.lon_min = longitude.isel(profile=0).item()
        self.lon_max = longitude.isel(profile=-1).item()

        # The derived variables are bound to a profile range, so they are built
        # on demand and rebuilt whenever a different range is read.
        self._derived = None
        self._derived_bounds = None

    @property
    def filepath(self):
        """Path of the granule file being read."""

        return self.granule_file.filepath

    @property
    def last_profile_in_file(self):
        """Index of the last profile of the granule, subset or not."""

        return self.granule_file.nb_profiles - 1

    def close(self):
        """Close the underlying CALIOP file."""

        self.granule_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _derived_variables(self, profile_start, profile_end):
        """Return the derived variables of a profile range, rebuilt when it changes."""

        bounds = (profile_start, profile_end)
        if bounds != self._derived_bounds:
            self._derived = DerivedVariables(
                self.granule_file,
                profile_start,
                profile_end,
            )
            self._derived_bounds = bounds
        return self._derived

    def get_data(self, key, profile_start=None, profile_end=None, do_fillvalue=True):
        """
        Get data on regular grid.

        :param key: a CALIPSO parameter
        :param profile_start: (optional) first profile to read (at 333 m resolution)
                              default: the first profile of the reader
        :param profile_end: (optional) last profile to read (included)
                            default: the last profile of the reader
        :param do_fillvalue: mask where fillvalue
        :return: labelled xarray data array
        """
        profile_start = self.prof_min if profile_start is None else int(profile_start)
        profile_end = self.prof_max if profile_end is None else int(profile_end)

        if key in self.granule_file.get_cal_keys():
            var_of_profiles = self.granule_file.is_profile_variable(key)
            data = self.granule_file.get_data(
                key,
                profile_start if var_of_profiles else None,
                profile_end if var_of_profiles else None,
                do_fillvalue,
            )
        elif key in DERIVED_VARIABLES:
            data = DERIVED_VARIABLES[key](
                self._derived_variables(profile_start, profile_end),
                do_fillvalue,
            )
        else:
            raise KeyError(f"Error: unknown key = {key}.\n")

        attributes = data.attrs.copy() if isinstance(data, xr.DataArray) else {}
        if isinstance(data, xr.DataArray):
            data = as_masked_array(data) if do_fillvalue else data.values

        # Put on the regular 30 m vertical grid, truncated at the requested altitude
        vertical_dimension = None
        if key == "Lidar_Data_Altitudes":
            if data.size == NUMBER_OF_VERTICAL_BINS:
                data = alt_to_regular_30m_vertical_grid(data)
                data = data[: self.max_altitude_index]
        elif data.ndim == 1:
            # No vertical averaging for 1D data
            pass
        elif data.ndim == 2:
            if data.shape[1] == NUMBER_OF_VERTICAL_BINS:
                data = shape_to_regular_30m_vertical_grid(data)
                data = data[:, : self.max_altitude_index]
                vertical_dimension = "altitude"
            elif data.shape[1] == NUMBER_OF_VERTICAL_BINS_MET:
                vertical_dimension = "met_altitude"
                if key == "Temperature":
                    derived = self._derived_variables(profile_start, profile_end)
                    data = derived.interp_temperature(data, do_fillvalue)
                    data = shape_to_regular_30m_vertical_grid(data)
                    data = data[:, : self.max_altitude_index]
                    vertical_dimension = "altitude"
        else:
            raise ValueError(f"Error: key = {key} with ndim ≥ 3 not implemented yet.\n")

        return self._as_dataarray(
            key,
            data,
            vertical_dimension,
            profile_start,
            attributes,
        )

    def _as_dataarray(
        self,
        key,
        data,
        vertical_dimension,
        profile_start,
        attributes=None,
    ):
        """Return regular-grid data with stable dimensions and coordinates."""

        if isinstance(data, xr.DataArray):
            data = data.values
        if data.ndim == 0:
            dims = ()
            coords = {}
        elif key == "Lidar_Data_Altitudes":
            dims = ("altitude",)
            coords = {"altitude": np.arange(data.shape[0])}
        elif key == "Met_Data_Altitudes":
            dims = ("met_altitude",)
            coords = {"met_altitude": np.arange(data.shape[0])}
        elif data.ndim == 1:
            dims = ("profile",)
            coords = {"profile": _profile_coordinate(profile_start, data.shape[0])}
        elif data.ndim == 2:
            dims = ("profile", vertical_dimension)
            coords = {
                "profile": _profile_coordinate(profile_start, data.shape[0]),
                vertical_dimension: np.arange(data.shape[1]),
            }
        else:
            dims = tuple(f"{key.lower()}_dim_{axis}" for axis in range(data.ndim))
            coords = {}
        return xr.DataArray(
            data,
            dims=dims,
            coords=coords,
            name=key,
            attrs=attributes or {},
        )


def _profile_coordinate(profile_start, size):
    """Index the profiles of a slice with their granule-wide indexes."""

    return np.arange(profile_start, profile_start + size, dtype=int)
