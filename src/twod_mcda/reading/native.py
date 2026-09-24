"""Read CALIOP products on their native grid, straight from the granule file.

``CALIOPGranuleFile`` is the lowest layer of the reading stack: it keeps one file
open, caches the profile slice being processed, resolves fill values and gives
every array its 2D-McDA dimension names. It knows nothing about the regular
30 m grid or about derived variables; that is ``reader.CALIOPRegularGridReader``.

CALIOP Level 1 granules are HDF4 files, read here through the netCDF4 engine:
``libnetcdf`` is built with HDF4 support on conda-forge, so every SDS is
readable as an ordinary netCDF variable, with its own dimension names and
attributes. Only SDSs are reachable this way, not the HDF4 Vdata tables. The
two vertical grids live in both, with identical values, so nothing is lost.
"""

from pathlib import Path

import numpy as np
import xarray as xr

from twod_mcda.caliop.constants import (
    FILL_VALUE_FLOAT,
    LIDAR_ALTITUDE_DIMENSION,
    MET_ALTITUDE_DIMENSION,
)

#: Dimension carrying the profiles of a granule, as named in the file.
PROFILE_DIMENSION = "Record_Number"

#: File dimensions renamed to the names used throughout 2D-McDA. Any other
#: dimension keeps its own name, lowercased.
DIMENSION_NAMES = {
    PROFILE_DIMENSION: "profile",
    "Lidar_Data_Altitudes": LIDAR_ALTITUDE_DIMENSION,
    "Met_Data_Altitudes": MET_ALTITUDE_DIMENSION,
}

#: Vertical grids read in double precision: the regular 30 m grid is
#: extrapolated from them, and the file stores them as float32.
VERTICAL_GRID_VARIABLES = ("Lidar_Data_Altitudes", "Met_Data_Altitudes")


class CALIOPGranuleFile:
    """Lazy reader that keeps one CALIOP file open and caches one profile slice."""

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self._dataset = xr.open_dataset(
            filepath,
            engine="netcdf4",
            # Fill values are applied by ``get_data`` instead: several CALIOP
            # variables carry -9999 without declaring ``_FillValue``, and
            # xarray's own masking would leave those values in the data.
            mask_and_scale=False,
        )
        self.nb_profiles = self._dataset.sizes[PROFILE_DIMENSION]
        self._active_profile_bounds = None
        self._slice_cache = {}
        self._static_cache = {}

    def close(self):
        """Close the underlying granule file."""

        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_cal_keys(self):
        """Return every variable the granule file holds."""

        return self._dataset.variables.keys()

    def is_profile_variable(self, key):
        """Return whether a variable carries the granule profile dimension."""

        return (
            key in self._dataset.variables
            and PROFILE_DIMENSION in self._dataset[key].dims
        )

    def get_fillvalue(self, key):
        """Return the fill value of a variable.

        CALIOP declares ``_FillValue`` on most variables but not on all of
        them: the meteorological profiles carry -9999 without declaring it,
        hence the fallback.
        """

        return self._dataset[key].attrs.get("_FillValue", FILL_VALUE_FLOAT)

    def _label(self, key, array, profile_start):
        """Rename the file dimensions and index the profiles of the slice."""

        dims = tuple(DIMENSION_NAMES.get(dim, dim.lower()) for dim in array.dims)
        coords = {}
        if PROFILE_DIMENSION in array.dims:
            profile_size = array.sizes[PROFILE_DIMENSION]
            coords["profile"] = np.arange(
                profile_start,
                profile_start + profile_size,
                dtype=int,
            )
        return xr.DataArray(
            array.data,
            dims=dims,
            coords=coords,
            name=key,
            attrs=array.attrs,
        )

    def _read(self, key, profile_min, profile_max):
        """Read one variable, over the requested profiles when it has any."""

        if not self.is_profile_variable(key):
            if key not in self._static_cache:
                self._static_cache[key] = self._load(key)
            return self._static_cache[key]

        bounds = (profile_min, profile_max)
        if bounds != self._active_profile_bounds:
            self._active_profile_bounds = bounds
            self._slice_cache.clear()
        if key not in self._slice_cache:
            self._slice_cache[key] = self._load(key, profile_min, profile_max)
        return self._slice_cache[key]

    def _load(self, key, profile_min=None, profile_max=None):
        """Read one variable off disk and give it its 2D-McDA labels."""

        array = self._dataset[key]

        profile_start = 0
        if PROFILE_DIMENSION in array.dims:
            profile_start = 0 if profile_min is None else int(profile_min)
            profile_stop = (
                self.nb_profiles if profile_max is None else int(profile_max) + 1
            )
            array = array.isel(
                {PROFILE_DIMENSION: slice(profile_start, profile_stop)}
            )

        # CALIOP stores its per-profile scalars with a trailing dimension of
        # length one. The profile dimension is never squeezed, so that a
        # single-profile slice keeps its shape.
        squeezable = [
            dim
            for dim in array.dims
            if dim != PROFILE_DIMENSION and array.sizes[dim] == 1
        ]
        if squeezable:
            array = array.squeeze(dim=squeezable, drop=True)

        array = array.load()
        if key in VERTICAL_GRID_VARIABLES:
            array = array.astype("float64")

        return self._label(key, array, profile_start)

    def get_data(self, key, slice_start=None, slice_end=None, do_fillvalue=True):
        """
        Get data for the key parameter from slice_start to slice_end.

        :param key: a CALIPSO parameter
        :param slice_start: (optional) start profile of the slice to load
                            default: the first profile
        :param slice_end: (optional) end profile of the slice to load (included)
                          default: the end of the data
        :param do_fillvalue: mask where fillvalue
        :return: labelled xarray data array
        """

        if key not in self._dataset.variables:
            raise KeyError(f"Error: key = '{key}' not found in {self.filepath}.")

        data = self._read(key, slice_start, slice_end)
        if do_fillvalue:
            fill_value = self.get_fillvalue(key)
            data = data.where(data != fill_value)
            data.attrs["_FillValue"] = fill_value
        return data
