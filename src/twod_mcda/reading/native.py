"""Read CALIOP products on their native grid, straight from the HDF4 file.

``CALIPSOReader`` is the lowest labelled layer above ``hdf.HDF4Reader``: it keeps
one file open, caches one profile slice, resolves fill values and gives every
array its CALIOP dimension names. It knows nothing about the regular 30 m grid
or about derived variables; that is ``reader.CALIOPRegularGridReader``.
"""

import numpy as np
import xarray as xr

from twod_mcda.caliop.constants import (
    FILL_VALUE_FLOAT,
    NUMBER_OF_VERTICAL_BINS,
    NUMBER_OF_VERTICAL_BINS_MET,
)
from twod_mcda.caliop.geography import get_prof_min_max_indexes_from_lon
from twod_mcda.reading.hdf import HDF4Reader
from twod_mcda.reading.variables import CALIOP_L1_VARIABLE_DIMS


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
                        dims=tuple(f"hdf_dim_{axis}" for axis in range(data.ndim)),
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
            size == 1 for axis, size in enumerate(shape) if axis < profile_axis
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

    def get_data(
        self,
        key,
        slice_start=None,
        slice_end=None,
        slice_start_end_type="profindex",
        do_fillvalue=True,
    ):
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
            raise Exception(
                f"Error: slice_start_end_type = '{slice_start_end_type}' is not "
                "defined. Please use 'profindex' or 'longitude'\n"
            )

        data = self._read_sds(key, prof_min, prof_max)
        if do_fillvalue:
            fill_value = self.get_fillvalue(key)
            data = data.where(data != fill_value)
            data.attrs["_FillValue"] = fill_value
        return data

