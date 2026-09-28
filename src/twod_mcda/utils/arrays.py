"""Helpers for the missing values of xarray objects."""

import numpy as np

from twod_mcda.caliop.constants import FILL_VALUE_FLOAT


def mask_invalid(data, fill_value=FILL_VALUE_FLOAT):
    """Return a DataArray with NaN where not finite or equal to the fill value."""

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
