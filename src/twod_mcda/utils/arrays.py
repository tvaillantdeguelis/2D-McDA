"""Helpers at the boundary between xarray and legacy numerical kernels."""

import numpy as np
import xarray as xr

from twod_mcda.caliop.constants import FILL_VALUE_FLOAT


def as_masked_array(data, fill_value=FILL_VALUE_FLOAT):
    """Return an xarray object as a masked array for a legacy kernel."""

    values = data.values if isinstance(data, xr.DataArray) else data
    values = np.ma.masked_invalid(np.ma.asarray(values))
    if fill_value is not None and _is_representable(fill_value, values.dtype):
        values = np.ma.masked_equal(
            values,
            np.asarray(fill_value, dtype=values.dtype).item(),
        )
    return values


def mask_invalid(data, fill_value=FILL_VALUE_FLOAT):
    """Return a DataArray with NaN wherever ``as_masked_array`` would mask it."""

    valid = np.isfinite(data)
    if fill_value is not None and _is_representable(fill_value, data.dtype):
        valid &= data != fill_value
    return data.where(valid)


def _is_representable(value, dtype):
    """Return whether a scalar round-trips through a NumPy dtype."""

    try:
        converted = np.asarray(value, dtype=dtype).item()
    except (OverflowError, TypeError, ValueError):
        return False
    return converted == value
