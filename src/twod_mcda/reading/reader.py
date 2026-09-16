"""Read CALIOP L1 products on the regular grid expected by 2D-McDA.

``CALIOPRegularGridReader`` sits on top of ``native.CALIPSOReader``: it resolves
the granule file, selects a profile range (by index or by longitude), puts every
array on the regular 30 m vertical grid, and delegates every variable CALIOP does
not store to ``derived.DerivedVariables``.
"""

from copy import copy
import os

import numpy as np
import xarray as xr

from twod_mcda.caliop.constants import (
    CAL_LID_FILENAME_FMT,
    NUMBER_OF_VERTICAL_BINS,
    NUMBER_OF_VERTICAL_BINS_MET,
)
from twod_mcda.caliop.geography import get_prof_min_max_indexes_from_lon
from twod_mcda.caliop.grids import (
    alt_to_regular_30m_vertical_grid,
    from_30mx333m_to_new_resolution,
    shape_to_regular_30m_vertical_grid,
)
from twod_mcda.reading.derived import DerivedVariables
from twod_mcda.reading.native import CALIPSOReader
from twod_mcda.utils.arrays import as_masked_array


class CALIOPRegularGridReader:
    def __init__(
        self,
        product,
        version,
        data_type,
        granule,
        folderpath,
        grid="333mx30m",
        slice_start=None,
        slice_end=None,
        slice_start_end_type="profindex",
        index30m_alt_max=None,
    ):
        """
        Get original and derived CALIOP parameters on a regular grid.

        :param product: CALIOP data product ('L1', 'L2_VFM', ...)
        :param version: CALIOP version product, without the 'V' prefix (ex: '4.10')
        :param data_type: CALIOP data type (ex: 'Standard')
        :param granule: 'YYYY-MM-DDThh-mm-ssZx'
        :param folderpath: directory holding the granule file
        :param grid: (optional) regular grid on which to put the data: '333m×30m', '1kmx60m', '5kmx60m', or '5kmx180m'
                     default: '333mx30m' (CALIOP full resolution)
        :param slice_start: (optional) start profile of the slice to load (at 333 m resolution)
                            default: the first profile
        :param slice_end: (optional) end profile of the slice to load (included) (at 333 m resolution)
                          default: the end of the data
        :param slice_start_end_type: 'profindex' if profile indexes provided or 'longitude' if
                                     longitudes provided (longitudes because increases/decreases
                                     monotonously on one granule unlike latitudes)
                                     default: 'profindex'
        """
        self.product = product
        self.version = version
        self.data_type = data_type
        self.hgrid = grid.split("x")[0]
        self.vgrid = grid.split("x")[1]
        self.granule = granule
        self.index30m_alt_max = index30m_alt_max
        self.filename = CAL_LID_FILENAME_FMT % (
            product,
            data_type,
            f"V{version}".replace(".", "-"),
            granule,
        )
        self.folderpath = folderpath
        self.filepath = os.path.join(self.folderpath, self.filename)

        self.data_reader = CALIPSOReader(self.filepath)

        # Get lat and lon on the whole granule (2D-McDA always reads the L1 product at its
        # native 333 m resolution, so the granule-wide and native-resolution lat/lon are the same)
        self.lon_granule = self.data_reader.get_data("Longitude").copy()
        self.lat_granule = self.data_reader.get_data("Latitude").copy()
        self._lon_granule_l1 = self.lon_granule.copy()
        self._lat_granule_l1 = self.lat_granule.copy()

        if slice_start_end_type == "profindex":
            if slice_start is not None:
                if slice_start >= 0:
                    self.prof_min = int(slice_start)
                else:
                    self.prof_min = self._lon_granule_l1.size + int(slice_start)
            else:
                self.prof_min = 0
            if slice_end is not None:
                self.prof_max = int(slice_end)
            else:
                self.prof_max = self._lat_granule_l1.size - 1
        elif slice_start_end_type == "longitude":
            self.prof_min, self.prof_max = get_prof_min_max_indexes_from_lon(
                self._lon_granule_l1, slice_start, slice_end
            )
        else:
            raise Exception(
                f"Error: slice_start_end_type = '{slice_start_end_type}' is not "
                "defined. Please use 'profindex' or 'longitude'\n"
            )
        self.lon_min = self._lon_granule_l1.isel(profile=self.prof_min).item()
        self.lat_min = self._lat_granule_l1.isel(profile=self.prof_min).item()
        self.lon_max = self._lon_granule_l1.isel(profile=self.prof_max).item()
        self.lat_max = self._lat_granule_l1.isel(profile=self.prof_max).item()
        self.nb_profiles = self.prof_max - self.prof_min + 1
        self._derived = DerivedVariables(
            self.data_reader,
            self.prof_min,
            self.prof_max,
        )

    def close(self):
        """Close the underlying CALIOP file."""

        self.data_reader.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def select_profiles(self, profile_start, profile_end):
        """Return a lightweight slice view sharing the open HDF4 file."""

        selected = copy(self)
        selected.prof_min = int(profile_start)
        selected.prof_max = int(profile_end)
        selected.lon_min = selected._lon_granule_l1.isel(
            profile=selected.prof_min
        ).item()
        selected.lat_min = selected._lat_granule_l1.isel(
            profile=selected.prof_min
        ).item()
        selected.lon_max = selected._lon_granule_l1.isel(
            profile=selected.prof_max
        ).item()
        selected.lat_max = selected._lat_granule_l1.isel(
            profile=selected.prof_max
        ).item()
        selected.nb_profiles = selected.prof_max - selected.prof_min + 1
        # The derived variables are bound to a profile range, so the slice needs
        # its own instance rather than the one copied from the whole granule.
        selected._derived = DerivedVariables(
            selected.data_reader,
            selected.prof_min,
            selected.prof_max,
        )
        return selected

    def print_lat_lon_min_max(self):
        print(
            f"\tFrom min profile index {self.prof_min:d} "
            f"(lat = {self.lat_min:.2f} / "
            f"lon = {self.lon_min:.2f}) "
            f"to max profile index {self.prof_max:d} "
            f"(lat = {self.lat_max:.2f} / "
            f"lon = {self.lon_max:.2f})"
        )

    def get_data(self, key, do_fillvalue=True):
        """
        Get data on regular grid.

        :param key: a CALIPSO parameter
        :param do_fillvalue: mask where fillvalue
        :return: labelled xarray data array
        """
        if key in self.data_reader.get_cal_keys():
            var_of_profiles = self.data_reader.is_profile_variable(key)
            data = self.data_reader.get_data(
                key,
                self.prof_min if var_of_profiles else None,
                self.prof_max if var_of_profiles else None,
                "profindex",
                do_fillvalue,
            )

        elif key == "Parallel_Attenuated_Backscatter_532":
            data = self._derived.par_ab532(do_fillvalue)

        elif key == "Molecular_Total_Attenuated_Backscatter_532":
            data, _ = self._derived.molecular_profiles(532, "", do_fillvalue)
        elif key == "Molecular_Parallel_Attenuated_Backscatter_532":
            data, _ = self._derived.molecular_profiles(532, "par", do_fillvalue)
        elif key == "Molecular_Perpendicular_Attenuated_Backscatter_532":
            data, _ = self._derived.molecular_profiles(532, "per", do_fillvalue)
        elif key == "Molecular_Attenuated_Backscatter_1064":
            data, _ = self._derived.molecular_profiles(1064, "", do_fillvalue)

        elif key == "Molecular_Total_Backscatter_532":
            _, data = self._derived.molecular_profiles(532, "", do_fillvalue)
        elif key == "Molecular_Parallel_Backscatter_532":
            _, data = self._derived.molecular_profiles(532, "par", do_fillvalue)
        elif key == "Molecular_Perpendicular_Backscatter_532":
            _, data = self._derived.molecular_profiles(532, "per", do_fillvalue)
        elif key == "Molecular_Backscatter_1064":
            _, data = self._derived.molecular_profiles(1064, "", do_fillvalue)

        #### Compute NSF in beta' domain (NSF in level 1B data is in V = P / Ga domain)
        elif key == "Noise_Scale_Factor_532_Parallel_AB_domain":
            data = self._derived.nsf_in_ab_domain(532, "par", do_fillvalue)
        elif key == "Noise_Scale_Factor_532_Perpendicular_AB_domain":
            data = self._derived.nsf_in_ab_domain(532, "per", do_fillvalue)
        elif key == "Noise_Scale_Factor_1064_AB_domain":
            data = self._derived.nsf_in_ab_domain(1064, "", do_fillvalue)

        #### Compute RMS in beta' domain (RMS in level 1B data is in P domain)
        elif key == "Parallel_RMS_Baseline_532_AB_domain":
            data = self._derived.rms_in_ab_domain(532, "par", do_fillvalue)
        elif key == "Perpendicular_RMS_Baseline_532_AB_domain":
            data = self._derived.rms_in_ab_domain(532, "per", do_fillvalue)
        elif key == "RMS_Baseline_1064_AB_domain":
            data = self._derived.rms_in_ab_domain(1064, "", do_fillvalue)

        #### Compute range-dependent uncertainty based on molecular model
        elif key == "Shot_Noise_532_Parallel":
            data = self._derived.shotnoise(532, "par", do_fillvalue)
        elif key == "Shot_Noise_532_Perpendicular":
            data = self._derived.shotnoise(532, "per", do_fillvalue)
        elif key == "Shot_Noise_1064":
            data = self._derived.shotnoise(1064, "", do_fillvalue)

        #### Compute range-independent uncertainty from background RMS ####
        elif key == "Background_Noise_532_Parallel":
            data = self._derived.backgroundnoise(532, "par", do_fillvalue)
        elif key == "Background_Noise_532_Perpendicular":
            data = self._derived.backgroundnoise(532, "per", do_fillvalue)
        elif key == "Background_Noise_1064":
            data = self._derived.backgroundnoise(1064, "", do_fillvalue)

        #### Compute scattering ratio uncertainty standard deviation ####
        elif (
            key
            == "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel"
        ):
            shotnoise_532_par = self._derived.shotnoise(532, "par", do_fillvalue)
            bkgnoise_532_par = self._derived.backgroundnoise(532, "par", do_fillvalue)
            data = np.sqrt(shotnoise_532_par**2 + bkgnoise_532_par**2)
        elif (
            key
            == "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular"
        ):
            shotnoise_532_per = self._derived.shotnoise(532, "per", do_fillvalue)
            bkgnoise_532_per = self._derived.backgroundnoise(532, "per", do_fillvalue)
            data = np.sqrt(shotnoise_532_per**2 + bkgnoise_532_per**2)
        elif key == "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064":
            shotnoise_1064 = self._derived.shotnoise(1064, "", do_fillvalue)
            bkgnoise_1064 = self._derived.backgroundnoise(1064, "", do_fillvalue)
            data = np.sqrt(shotnoise_1064**2 + bkgnoise_1064**2)
        else:
            raise Exception(f"Error: unknown key = {key}.\n")

        attributes = data.attrs.copy() if isinstance(data, xr.DataArray) else {}
        if isinstance(data, xr.DataArray):
            data = as_masked_array(data) if do_fillvalue else data.values

        # Put on regular grid (new resolution but grid stays 300m×30m)
        if key == "Lidar_Data_Altitudes":
            if data.size == NUMBER_OF_VERTICAL_BINS:
                data = alt_to_regular_30m_vertical_grid(data)
                data = data[: self.index30m_alt_max]
        elif data.ndim == 1:
            # No vertical averaging for 1D data
            pass
        elif data.ndim == 2:
            if data.shape[1] == NUMBER_OF_VERTICAL_BINS:
                data = shape_to_regular_30m_vertical_grid(data)
                data = data[:, : self.index30m_alt_max]
                data = from_30mx333m_to_new_resolution(
                    data, self.vgrid, self.hgrid, self.prof_min
                )
            elif data.shape[1] == NUMBER_OF_VERTICAL_BINS_MET:
                if key == "Temperature":
                    data = self._derived.interp_temperature(data, do_fillvalue)
                    data = shape_to_regular_30m_vertical_grid(data)
                    data = data[:, : self.index30m_alt_max]
                    data = from_30mx333m_to_new_resolution(
                        data, self.vgrid, self.hgrid, self.prof_min
                    )
        else:
            raise Exception(f"Error: key = {key} with ndim ≥ 3 not implemented yet.\n")

        return self._as_dataarray(key, data, attributes)

    def _as_dataarray(self, key, data, attributes=None):
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
            coords = {
                "profile": np.arange(
                    self.prof_min,
                    self.prof_min + data.shape[0],
                    dtype=int,
                )
            }
        elif data.ndim == 2:
            vertical_dimension = (
                "met_altitude"
                if data.shape[1] == NUMBER_OF_VERTICAL_BINS_MET
                else "altitude"
            )
            dims = ("profile", vertical_dimension)
            coords = {
                "profile": np.arange(
                    self.prof_min,
                    self.prof_min + data.shape[0],
                    dtype=int,
                ),
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
