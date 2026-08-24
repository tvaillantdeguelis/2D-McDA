"""Read CALIOP products on the regular grid expected by 2D-McDA."""

from copy import copy
import os

import numpy as np
import xarray as xr
from twod_mcda.caliop.derived import CALIOPDerivedVariablesMixin
from twod_mcda.caliop.hdf import HDF4Reader
from twod_mcda.caliop.paths import automatic_path_detection
from twod_mcda.caliop.constants import (
    CAL_LID_FILENAME_FMT,
    FILL_VALUE_FLOAT,
    LIDAR_DATA_ALTITUDES,
    NUMBER_OF_VERTICAL_BINS,
    NUMBER_OF_VERTICAL_BINS_MET,
    N_LAYERS_01KM_LAYER_PRODUCT,
    N_LAYERS_05KM_LAYER_PRODUCT,
    N_LAYERS_333M_LAYER_PRODUCT,
    PRODUCT_H_RESOLUTION,
)
from twod_mcda.caliop.geography import get_prof_min_max_indexes_from_lon
from twod_mcda.caliop.variables import CALIOP_L1_VARIABLE_DIMS
from twod_mcda.caliop.xarray_utils import as_masked_array
from twod_mcda.caliop.grids import (
    add_zeros_where_missing_profiles,
    alt_to_regular_30m_vertical_grid,
    duplicate_1km_to_333m,
    duplicate_5km_to_333m,
    from_30mx333m_to_new_resolution,
    shape_to_regular_30m_vertical_grid,
    unfold_vfm,
)
MESSAGE_EXECPTION_SLICE_START_END_TYPE = '''f"Error: slice_start_end_type = '{slice_start_end_type}'" \
                                          "is not defined. Please use 'profindex' or 'longitude'\n"'''

class CALIPSOReader:
    """Lazy reader that keeps one HDF4 file open and caches one profile slice."""

    def __init__(self, filepath):
        self.filepath = filepath
        self._reader = HDF4Reader(filepath).__enter__()
        self._sds = self._reader.get_sds_keys()
        self._metadata = {
            key: self._reader.get_metadata(key)
            for key in self._reader.get_metadata_keys()
        }
        self._fill_values = {}
        self._active_profile_bounds = None
        self._slice_cache = {}
        self._static_cache = {}
        self.nb_profiles = self._sds["Latitude"][1][0]

    def close(self):
        """Close the underlying HDF4 handles."""

        if self._reader is not None:
            self._reader.__exit__(None, None, None)
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_cal_keys(self):
        return self._sds.keys() | self._metadata.keys()

    def get_fillvalue(self, key):
        if key in self._metadata:
            return None
        if key not in self._fill_values:
            fill_value = self._reader.get_fillvalue(key)
            if fill_value is None:
                fill_value = FILL_VALUE_FLOAT
            self._fill_values[key] = fill_value
        return self._fill_values[key]

    def _profile_axis(self, key):
        shape = self._sds[key][1]
        if shape[0] == self.nb_profiles:
            return 0
        if len(shape) > 1 and shape[1] == self.nb_profiles:
            return 1
        return None

    def is_profile_variable(self, key):
        """Return whether an SDS contains the granule profile dimension."""

        return key in self._sds and self._profile_axis(key) is not None

    @staticmethod
    def _squeeze_non_profile_axes(data, original_shape, profile_axis):
        axes = tuple(
            axis
            for axis, size in enumerate(original_shape)
            if size == 1 and axis != profile_axis
        )
        if axes:
            return data.squeeze(
                dim=tuple(data.dims[axis] for axis in axes),
                drop=True,
            )
        return data

    def _dimension_names(
        self,
        key,
        data,
        profile_start=0,
        profile_axis=None,
    ):
        """Attach stable semantic dimensions to one CALIOP variable."""

        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(
                data,
                dims=tuple(f"hdf_dim_{axis}" for axis in range(data.ndim)),
                name=key,
            )
        declared = CALIOP_L1_VARIABLE_DIMS.get(key)
        if declared is not None and len(declared) == data.ndim:
            dims = declared
        else:
            dims = []
            used = set()
            for axis, size in enumerate(data.shape):
                if axis == profile_axis or (
                    profile_axis is None
                    and size == self.nb_profiles
                    and "profile" not in used
                ):
                    dim = "profile"
                elif size == NUMBER_OF_VERTICAL_BINS and "lidar_altitude" not in used:
                    dim = "lidar_altitude"
                elif size == NUMBER_OF_VERTICAL_BINS_MET and "met_altitude" not in used:
                    dim = "met_altitude"
                else:
                    dim = f"{key.lower()}_dim_{axis}"
                dims.append(dim)
                used.add(dim)
            dims = tuple(dims)

        coords = {}
        if "profile" in dims:
            profile_size = data.shape[dims.index("profile")]
            coords["profile"] = np.arange(
                profile_start,
                profile_start + profile_size,
                dtype=int,
            )
        return xr.DataArray(
            data.data,
            dims=dims,
            coords=coords,
            name=key,
            attrs=data.attrs,
        )

    def _read_sds(self, key, profile_min, profile_max):
        shape = self._sds[key][1]
        profile_axis = self._profile_axis(key)

        if profile_axis is None:
            if key not in self._static_cache:
                data = self._reader.get_data(key, do_squeeze=False)
                if not isinstance(data, xr.DataArray):
                    data = xr.DataArray(
                        data,
                        dims=tuple(
                            f"hdf_dim_{axis}" for axis in range(data.ndim)
                        ),
                        name=key,
                    )
                self._static_cache[key] = self._squeeze_non_profile_axes(
                    data,
                    shape,
                    None,
                )
                self._static_cache[key] = self._dimension_names(
                    key,
                    self._static_cache[key],
                )
            return self._static_cache[key]

        bounds = (profile_min, profile_max)
        if bounds != self._active_profile_bounds:
            self._active_profile_bounds = bounds
            self._slice_cache.clear()
        if key in self._slice_cache:
            return self._slice_cache[key]

        start_index = 0 if profile_min is None else profile_min
        end_index = self.nb_profiles - 1 if profile_max is None else profile_max
        start = [0] * len(shape)
        count = list(shape)
        start[profile_axis] = start_index
        count[profile_axis] = end_index - start_index + 1
        data = self._reader.get_data(
            key,
            start=start,
            count=count,
            do_squeeze=False,
        )
        if not isinstance(data, xr.DataArray):
            data = xr.DataArray(
                data,
                dims=tuple(f"hdf_dim_{axis}" for axis in range(data.ndim)),
                name=key,
            )
        data = self._squeeze_non_profile_axes(data, shape, profile_axis)
        squeezed_before_profile = sum(
            size == 1
            for axis, size in enumerate(shape)
            if axis < profile_axis
        )
        effective_profile_axis = profile_axis - squeezed_before_profile
        data = self._dimension_names(
            key,
            data,
            start_index,
            effective_profile_axis,
        )
        self._slice_cache[key] = data
        return data

    def get_data(self, key, slice_start=None, slice_end=None, slice_start_end_type='profindex',
                 do_fillvalue=True):
        """
        Get data for the key parameter from slice_start to slice_end.
        
        :param key: a CALIPSO parameter
        :param slice_start: (optional) start profile of the slice to load
                            default: the first profile
        :param slice_end: (optional) end profile of the slice to load (included)
                          default: the end of the data
        :param slice_start_end_type: 'profindex' if profile indexes provided or 'longitude' if
                                     longitudes provided (longitudes because increases/decreases
                                     monotonously on one granule unlike latitudes)
                                     default: 'profindex'
        :param do_fillvalue: mask where fillvalue
        :return: labelled xarray data array
        """
        
        if key in self._metadata:
            values = np.asanyarray(self._metadata[key]).squeeze()
            raw = xr.DataArray(
                values,
                dims=tuple(f"metadata_dim_{axis}" for axis in range(values.ndim)),
                name=key,
            )
            return self._dimension_names(key, raw)
        if key not in self._sds:
            raise Exception(f"Error: key = '{key}' not found.\n")

        if slice_start_end_type == "profindex":
            prof_min, prof_max = slice_start, slice_end
        elif slice_start_end_type == "longitude":
            longitude = self.get_data("Longitude", do_fillvalue=False)
            prof_min, prof_max = get_prof_min_max_indexes_from_lon(
                longitude,
                slice_start,
                slice_end,
            )
        else:
            raise Exception(MESSAGE_EXECPTION_SLICE_START_END_TYPE)

        data = self._read_sds(key, prof_min, prof_max)
        if do_fillvalue:
            fill_value = self.get_fillvalue(key)
            data = data.where(data != fill_value)
            data.attrs["_FillValue"] = fill_value
        return data


class CALIOPReader():
    def __init__(self, product, version, data_type, granule_date, slice_start=None, slice_end=None,
                 slice_start_end_type='profindex', folderpath=None):
        """
        Get original and derived CALIOP parameters on a regular grid.
        
        :param product: CALIOP data product ('L1', 'L2_VFM', ...)
        :param version: CALIOP version product (ex: 'V4.10')
        :param data_type: CALIOP data type (ex: 'Standard')
        :param granule_date: 'YYYY-MM-DDThh-mm-ssZx'
        :param folderpath: (optional) data path, if not given then try automatic path detection
        :param slice_start: (optional) start profile of the slice to load
                            default: the first profile
        :param slice_end: (optional) end profile of the slice to load (included)
                          default: the end of the data
        :param slice_start_end_type: 'profindex' if profile indexes provided or 'longitude' if
                                     longitudes provided (longitudes because increases/decreases
                                     monotonously on one granule unlike latitudes)
                                     default: 'profindex'
        """
        self.product = product
        self.version = version
        self.data_type = data_type
        self.granule_date = granule_date
        if folderpath:
            self.folderpath = folderpath
        else: # automatic path detection
            self.folderpath = automatic_path_detection(product, version, data_type, granule_date)
        
        # Get filepath
        self.filename = CAL_LID_FILENAME_FMT % (product, data_type, version.replace(".", "-"),
                                                granule_date)
        self.filepath = os.path.join(self.folderpath, self.filename)
        
        # Load file
        self.data_reader = CALIPSOReader(self.filepath)
        
        # Get prof_min and prof_max
        lon = self.data_reader.get_data('Longitude')
        lat = self.data_reader.get_data('Latitude')
        if lon.ndim != 1:
            if lon.shape[1] == 3:
                lon = lon[:, 1]
                lat = lat[:, 1]
            else:
                raise Exception(f"TO DO")
        if slice_start_end_type == 'profindex':
            self.prof_min, self.prof_max = slice_start, slice_end
        elif slice_start_end_type == 'longitude':
            self.prof_min, self.prof_max = get_prof_min_max_indexes_from_lon(lon, slice_start, slice_end)
        else:
            raise Exception(f"Error: slice_start_end_type = '{slice_start_end_type}' is not defined. "
                     "Please use 'profindex' or 'longitude'\n")
        self.lon_min = lon.isel(profile=self.prof_min).item()
        self.lat_min = lat.isel(profile=self.prof_min).item()
        self.lon_max = lon.isel(profile=self.prof_max).item()
        self.lat_max = lat.isel(profile=self.prof_max).item()
        self.nb_profiles = self.prof_max - self.prof_min + 1
        

    def get_data(self, key, do_fillvalue=True):
        """
        Get data on regular grid.
        
        :param key: a CALIPSO parameter
        :param do_fillvalue: mask where fillvalue
        :return: labelled xarray data array
        """
        return self.data_reader.get_data(key, self.prof_min, self.prof_max, 'profindex', do_fillvalue)

        
class CALIOPRegularGridReader(CALIOPDerivedVariablesMixin):
    def __init__(self, product, version, data_type, granule_date, grid='333mx30m',
                 slice_start=None, slice_end=None, slice_start_end_type='profindex',
                 index30m_alt_max=None, folderpath=None, version_l1=None, data_type_l1=None,
                 deconvolution=False):
        """
        Get original and derived CALIOP parameters on a regular grid.
        
        :param product: CALIOP data product ('L1', 'L2_VFM', ...)
        :param version: CALIOP version product (ex: 'V4.10')
        :param data_type: CALIOP data type (ex: 'Standard')
        :param granule_date: 'YYYY-MM-DDThh-mm-ssZx'
        :param grid: (optional) regular grid on which to put the data: '333m×30m', '1kmx60m', '5kmx60m', or '5kmx180m'
                     default: '333mx30m' (CALIOP full resolution)
        :param folderpath: (optional) data path, if not given then try automatic path detection
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
        self.hgrid = grid.split('x')[0]
        self.vgrid = grid.split('x')[1]
        self.granule_date = granule_date
        self.index30m_alt_max = index30m_alt_max
        self.filename = CAL_LID_FILENAME_FMT % (product, data_type, version.replace(".", "-"),
                                                granule_date)
        if folderpath:
            self.folderpath = folderpath
        else: # automatic path detection
            self.folderpath = automatic_path_detection(product, version, data_type, granule_date)
        self.filepath = os.path.join(self.folderpath, self.filename)
        self.deconvolution = deconvolution
        self._molecular_profiles = {"532": None, "532par": None, "532per": None, "1064": None}
        
        self.data_reader = CALIPSOReader(self.filepath)
        
        # Get lat and lon on the whole granule
        self.lon_granule = self.data_reader.get_data('Longitude').copy()
        self.lat_granule = self.data_reader.get_data('Latitude').copy()
        
        # Get prof_min and prof_max from 333 m resolution lat/lon
        if self.product not in PRODUCT_H_RESOLUTION['333m']:
            try:
                self._lon_granule_l1 = self.data_reader.get_data('ssLongitude')
                self._lat_granule_l1 = self.data_reader.get_data('ssLatitude')
            except:
                if (version_l1 is None) or (data_type_l1 is None):
                    raise Exception("No latitude/longitude at 333 m resolution in the file. Please "
                                    "specify version_l1 and data_type_l1 when generating the "
                                    "CALIOPRegularGridReader object.")
                # Load lat/lon from corresponding L1 file
                cal_l1 = CALIOPRegularGridReader(product='L1',
                                                 version=version_l1,
                                                 data_type=data_type_l1,
                                                 granule_date=self.granule_date,
                                                 slice_start=slice_start,
                                                 slice_end=slice_end,
                                                 slice_start_end_type=slice_start_end_type)
                self._lon_granule_l1 = cal_l1.data_reader.get_data('Longitude')
                self._lat_granule_l1 = cal_l1.data_reader.get_data('Latitude')
        else:
            self._lon_granule_l1 = self.lon_granule.copy()
            self._lat_granule_l1 = self.lat_granule.copy()

        if slice_start_end_type=='profindex':
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
        elif slice_start_end_type=='longitude':
            self.prof_min, self.prof_max = get_prof_min_max_indexes_from_lon(self._lon_granule_l1, slice_start,
                                                                             slice_end)
        else:
            raise Exception(MESSAGE_EXECPTION_SLICE_START_END_TYPE)
        self.lon_min = self._lon_granule_l1.isel(profile=self.prof_min).item()
        self.lat_min = self._lat_granule_l1.isel(profile=self.prof_min).item()
        self.lon_max = self._lon_granule_l1.isel(profile=self.prof_max).item()
        self.lat_max = self._lat_granule_l1.isel(profile=self.prof_max).item()
        self.nb_profiles = self.prof_max - self.prof_min + 1

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
        selected.lon_min = selected._lon_granule_l1.isel(profile=selected.prof_min).item()
        selected.lat_min = selected._lat_granule_l1.isel(profile=selected.prof_min).item()
        selected.lon_max = selected._lon_granule_l1.isel(profile=selected.prof_max).item()
        selected.lat_max = selected._lat_granule_l1.isel(profile=selected.prof_max).item()
        selected.nb_profiles = selected.prof_max - selected.prof_min + 1
        selected._molecular_profiles = {
            "532": None,
            "532par": None,
            "532per": None,
            "1064": None,
        }
        return selected

    def print_lat_lon_min_max(self):
        print(f"\tFrom min profile index {self.prof_min:d} "
              f"(lat = {self.lat_min:.2f} / "
              f"lon = {self.lon_min:.2f}) "
              f"to max profile index {self.prof_max:d} "
              f"(lat = {self.lat_max:.2f} / "
              f"lon = {self.lon_max:.2f})")
    
    def get_data(self, key, do_fillvalue=True):
        """
        Get data on regular grid.
        
        :param key: a CALIPSO parameter
        :param do_fillvalue: mask where fillvalue
        :return: labelled xarray data array
        """
        if (key == 'Latitude') and (self.product not in PRODUCT_H_RESOLUTION['333m']):
            data = self._lat_granule_l1.copy()
        elif (key == 'Longitude') and (self.product not in PRODUCT_H_RESOLUTION['333m']):
            data = self._lon_granule_l1.copy()
        elif key in self.data_reader.get_cal_keys():
            var_of_profiles = self.data_reader.is_profile_variable(key)
            reads_native_slice = (
                var_of_profiles
                and self.product in PRODUCT_H_RESOLUTION['333m']
            )
            data = self.data_reader.get_data(
                key,
                self.prof_min if reads_native_slice else None,
                self.prof_max if reads_native_slice else None,
                'profindex',
                do_fillvalue,
            )
            if self.product in PRODUCT_H_RESOLUTION['333m']:
                pass
            elif self.product in PRODUCT_H_RESOLUTION['1km']:
                if (data.ndim >= 1) and var_of_profiles:
                    data = duplicate_1km_to_333m(data)
            elif (self.product in PRODUCT_H_RESOLUTION['5km']) or (self.product == "L2_VFM"):
                if key == "Feature_Classification_Flags":
                    data = unfold_vfm(data, put_in_all_alt_grid=True)
                elif key[:2] != "ss":
                    if (data.ndim >= 1) and var_of_profiles:
                        data = duplicate_5km_to_333m(data)
            else:
                raise Exception("Error: Product unknown or horizontal resolution of product unknown.")
            
            # Add zeros when nb profiles is different from L1 (at the end of the granule)
            if var_of_profiles and not reads_native_slice:
                nb_missing_prof = self._lat_granule_l1.shape[0] - data.shape[0] # doesn't work for VFM because _lat_granule_l1 is taken from VFM SS variables
                if nb_missing_prof > 0:
                    data = add_zeros_where_missing_profiles(data, nb_missing_prof)

            if var_of_profiles and not reads_native_slice:
                data = data[self.prof_min:self.prof_max + 1]

            # If layer variable then convert to grid
            try:
                key_is_layer_var = ((self.product == 'L2_05kmALay') or
                                    (self.product == 'L2_05kmCLay') or
                                    (self.product == 'L2_05kmMLay')) and \
                                   (self.data_reader.get_data(key).shape[1] == N_LAYERS_05KM_LAYER_PRODUCT) or \
                                   ((self.product == 'L2_01kmCLay')) and \
                                   (self.data_reader.get_data(key).shape[1] == N_LAYERS_01KM_LAYER_PRODUCT) or \
                                   ((self.product == 'L2_333mMLay')) and \
                                   (self.data_reader.get_data(key).shape[1] == N_LAYERS_333M_LAYER_PRODUCT)
            except:
                key_is_layer_var = False
            if key_is_layer_var:
                alt_base = as_masked_array(self.data_reader.get_data("Layer_Base_Altitude", None, None, 'profindex', do_fillvalue))
                alt_top = as_masked_array(self.data_reader.get_data("Layer_Top_Altitude", None, None, 'profindex', do_fillvalue))
                data = as_masked_array(data)
                if self.product in PRODUCT_H_RESOLUTION['333m']:
                    alt_base = alt_base[self.prof_min:self.prof_max+1, :]
                    alt_top = alt_top[self.prof_min:self.prof_max+1, :]
                    data = self.layer2grid(data, alt_base, alt_top)
                elif self.product in PRODUCT_H_RESOLUTION['1km']:
                    alt_base = duplicate_1km_to_333m(alt_base)
                    alt_top = duplicate_1km_to_333m(alt_top)
                    alt_base = alt_base[self.prof_min:self.prof_max+1, :]
                    alt_top = alt_top[self.prof_min:self.prof_max+1, :]
                    data = self.layer2grid(data, alt_base, alt_top)
                elif self.product in PRODUCT_H_RESOLUTION['5km']:
                    alt_base = duplicate_5km_to_333m(alt_base)
                    alt_top = duplicate_5km_to_333m(alt_top)
                    alt_base = alt_base[self.prof_min:self.prof_max+1, :]
                    alt_top = alt_top[self.prof_min:self.prof_max+1, :]
                    data = self.layer2grid(data, alt_base, alt_top)
            
        elif key == "Lidar_Data_Altitudes_init":
            data = self.data_reader.get_data("Lidar_Data_Altitudes", self.prof_min, self.prof_max,
                                             'profindex', do_fillvalue)
        
        elif key == "Parallel_Attenuated_Backscatter_532":
            data = self._get_par_ab532(do_fillvalue)
        
        elif key == "Molecular_Total_Attenuated_Backscatter_532":
            data, _ = self._get_molecular_profiles(532, '', do_fillvalue)
        elif key == "Molecular_Parallel_Attenuated_Backscatter_532":
            data, _ = self._get_molecular_profiles(532, 'par', do_fillvalue)
        elif key == "Molecular_Perpendicular_Attenuated_Backscatter_532":
            data, _ = self._get_molecular_profiles(532, 'per', do_fillvalue)
        elif key == "Molecular_Attenuated_Backscatter_1064":
            data, _ = self._get_molecular_profiles(1064, '', do_fillvalue)
         
        elif key == "Molecular_Total_Backscatter_532":
            _, data = self._get_molecular_profiles(532, '', do_fillvalue)
        elif key == "Molecular_Parallel_Backscatter_532":
            _, data = self._get_molecular_profiles(532, 'par', do_fillvalue)
        elif key == "Molecular_Perpendicular_Backscatter_532":
            _, data = self._get_molecular_profiles(532, 'per', do_fillvalue)
        elif key == "Molecular_Backscatter_1064":
            _, data = self._get_molecular_profiles(1064, '', do_fillvalue)
            
        #### Compute NSF in beta' domain (NSF in level 1B data is in V = P / Ga domain)
        elif key == "Noise_Scale_Factor_532_Parallel_AB_domain":
            data = self._get_nsf_in_ab_domain(532, 'par', do_fillvalue)
        elif key == "Noise_Scale_Factor_532_Perpendicular_AB_domain":
            data = self._get_nsf_in_ab_domain(532, 'per', do_fillvalue)
        elif key == "Noise_Scale_Factor_1064_AB_domain":
            data = self._get_nsf_in_ab_domain(1064, '', do_fillvalue)
            
        #### Compute RMS in beta' domain (RMS in level 1B data is in P domain)
        elif key == "Parallel_RMS_Baseline_532_AB_domain":
            data = self._get_rms_in_ab_domain(532, 'par', do_fillvalue)
        elif key == "Perpendicular_RMS_Baseline_532_AB_domain":
            data = self._get_rms_in_ab_domain(532, 'per', do_fillvalue)
        elif key == "RMS_Baseline_1064_AB_domain":
            data = self._get_rms_in_ab_domain(1064, '', do_fillvalue)
            
        #### Compute range-dependent uncertainty based on molecular model
        elif key == "Shot_Noise_532_Parallel":
            data = self._get_shotnoise(532, 'par', do_fillvalue)
        elif key == "Shot_Noise_532_Perpendicular":
            data = self._get_shotnoise(532, 'per', do_fillvalue)
        elif key == "Shot_Noise_1064":
            data = self._get_shotnoise(1064, '', do_fillvalue)

        #### Compute range-independent uncertainty from background RMS ####
        elif key == "Background_Noise_532_Parallel":
            data = self._get_backgroundnoise(532, 'par', do_fillvalue)
        elif key == "Background_Noise_532_Perpendicular":
            data = self._get_backgroundnoise(532, 'per', do_fillvalue)
        elif key == "Background_Noise_1064":
            data = self._get_backgroundnoise(1064, '', do_fillvalue)
        
        #### Compute scattering ratio uncertainty standard deviation ####
        elif key == "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Parallel":
            shotnoise_532_par = self._get_shotnoise(532, 'par', do_fillvalue)
            bkgnoise_532_par = self._get_backgroundnoise(532, 'par', do_fillvalue)
            data = np.sqrt(shotnoise_532_par**2 + bkgnoise_532_par**2)
        elif key == "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_532_Perpendicular":
            shotnoise_532_per = self._get_shotnoise(532, 'per', do_fillvalue)
            bkgnoise_532_per = self._get_backgroundnoise(532, 'per', do_fillvalue)
            data = np.sqrt(shotnoise_532_per**2 + bkgnoise_532_per**2)
        elif key == "Attenuated_Scattering_Ratio_Uncertainty_Standard_Deviation_1064":
            shotnoise_1064 = self._get_shotnoise(1064, '', do_fillvalue)
            bkgnoise_1064 = self._get_backgroundnoise(1064, '', do_fillvalue)
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
                data = data[:self.index30m_alt_max]
            # elif data.size == NUMBER_OF_VERTICAL_BINS_05KMPRO:
            #     data = alt05kmpro_to_regular_30m_vertical_grid(data)
        elif data.ndim == 1:
            # No vertical averaging for 1D data
            pass
        elif data.ndim == 2:
            if data.shape[1] == NUMBER_OF_VERTICAL_BINS:
                data = shape_to_regular_30m_vertical_grid(data)
                if self.deconvolution:
                    if key == "Parallel_Attenuated_Backscatter_532":
                        npzfile = np.load(os.path.split(os.path.join(os.path.dirname(__file__)))[0] +
                                          '/deconvolution_matrices.npz')
                        inv_mat_par = npzfile['inv_mat_par']
                        for i in np.arange(data.shape[0]):
                            data[i, :] = np.matmul(inv_mat_par, data[i, ::-1].filled(0))[::-1]
                            # [::-1] because profile top to bottom <=> bottom to top / fill mask values with 0 which should work with the deconvolution
                    elif key == "Perpendicular_Attenuated_Backscatter_532":
                        npzfile = np.load(os.path.split(os.path.join(os.path.dirname(__file__)))[0] +
                                          '/deconvolution_matrices.npz')
                        inv_mat_per = npzfile['inv_mat_per']
                        for i in np.arange(data.shape[0]):
                            data[i, :] = np.matmul(inv_mat_per, data[i, ::-1].filled(0))[::-1]
                            # [::-1] because profile top to bottom <=> bottom to top / fill mask values with 0 which should work with the deconvolution
                    elif key == "Total_Attenuated_Backscatter_532": # sum deconvoluted par and per
                        par_ab_532 = self.get_data("Parallel_Attenuated_Backscatter_532", do_fillvalue)
                        per_ab_532 = self.get_data("Perpendicular_Attenuated_Backscatter_532", do_fillvalue)
                        data = par_ab_532 + per_ab_532
                data = data[:, :self.index30m_alt_max]
                data = from_30mx333m_to_new_resolution(data, self.vgrid, self.hgrid, self.prof_min)
            elif data.shape[1] == NUMBER_OF_VERTICAL_BINS_MET:
                if key == "Temperature":
                    data = self._interp_temperature(data, do_fillvalue)
                    data = shape_to_regular_30m_vertical_grid(data)
                    data = data[:, :self.index30m_alt_max]
                    data = from_30mx333m_to_new_resolution(data, self.vgrid, self.hgrid, self.prof_min)
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
        elif key in {"Lidar_Data_Altitudes", "Lidar_Data_Altitudes_init"}:
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
    
    def layer2grid(self, data, alt_base, alt_top):
        """Replace layer properties in a grid using base and top altitudes"""
        data = as_masked_array(data)
        alt_base = as_masked_array(alt_base)
        alt_top = as_masked_array(alt_top)
        lidar_data_altitudes = LIDAR_DATA_ALTITUDES
        if lidar_data_altitudes is None:
            try:
                lidar_data_altitudes = self.data_reader.get_data(
                    "Lidar_Data_Altitudes"
                )
            except Exception as exc:
                raise RuntimeError(
                    "Lidar_Data_Altitudes is required to regrid layer data."
                ) from exc

        lidar_data_altitudes = as_masked_array(lidar_data_altitudes)
        data_grid = np.ma.ones((alt_base.shape[0], NUMBER_OF_VERTICAL_BINS))*FILL_VALUE_FLOAT
        for i_prof in np.arange(alt_base.shape[0]):
            for i_layer in np.arange(alt_base.shape[1]):
                if not alt_top.mask[i_prof, i_layer]: # if not masked (i.e. if layer exists)
                    alt_top_i = np.abs(
                        alt_top[i_prof, i_layer] - lidar_data_altitudes
                    ).argmin()
                    alt_base_i = np.abs(
                        alt_base[i_prof, i_layer] - lidar_data_altitudes
                    ).argmin()
                    data_grid[i_prof, alt_top_i:alt_base_i+1] = data[i_prof, i_layer]
        data_grid = np.ma.masked_where(data_grid == FILL_VALUE_FLOAT, data_grid)
        return data_grid
