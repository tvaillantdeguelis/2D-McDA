from pathlib import Path

import numpy as np
import xarray as xr

from twod_mcda.io.hdf_reader import HDF4Reader
from twod_mcda.io.caliop_l1_variables import CALIOP_L1_VARIABLE_DIMS


def read_caliop_l1(
    file_path,
    variables,
):
    """
    Read selected CALIOP Level 1 variables into an xarray Dataset.

    Data are returned on their native CALIOP grids. This function does
    not perform interpolation, regridding, averaging, or computation of
    derived variables.

    Parameters
    ----------
    file_path : str or Path
        Path to the CALIOP Level 1 HDF file.
    variables : iterable of str
        Names of the scientific data sets to read.

    Returns
    -------
    xarray.Dataset
        Dataset containing the requested CALIOP Level 1 variables.

    Raises
    ------
    FileNotFoundError
        If the input file does not exist.
    KeyError
        If a requested variable is absent from the file or has no
        registered dimension definition.
    ValueError
        If a variable shape does not match its registered dimensions.
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"CALIOP Level 1 file not found: {file_path}"
        )

    variables = tuple(variables)

    dataset = xr.Dataset(
        attrs={
            "source_file": str(file_path),
            "product": "CALIOP Level 1",
        }
    )

    with HDF4Reader(file_path) as reader:
        available_variables = set(reader.get_sds_keys())

        for variable in variables:
            if variable not in available_variables:
                raise KeyError(
                    f"Variable '{variable}' is not present in "
                    f"{file_path.name}."
                )

            if variable not in CALIOP_L1_VARIABLE_DIMS:
                raise KeyError(
                    f"No dimensions are registered for variable "
                    f"'{variable}'."
                )

            data = reader.get_data(variable)
            fill_value = _get_fill_value(
                reader,
                variable,
            )

            data = _replace_fill_values(
                data,
                fill_value,
            )

            dims = CALIOP_L1_VARIABLE_DIMS[variable]

            _validate_dimensions(
                variable,
                data,
                dims,
            )

            dataset[variable] = xr.DataArray(
                data=data,
                dims=dims,
                attrs=_build_variable_attrs(
                    reader,
                    variable,
                    fill_value,
                ),
            )

    return dataset


def _get_fill_value(
    reader,
    variable,
):
    """
    Return the fill value associated with one HDF variable.
    """

    fill_value = reader.get_fillvalue(variable)

    if fill_value is not None:
        return fill_value

    try:
        return reader._sd_interface.select(variable).fillvalue
    except AttributeError:
        return None
    

def _replace_fill_values(
    data,
    fill_value,
):
    """
    Replace floating-point fill values with NaN.

    Integer and categorical arrays keep their original fill value to
    avoid an implicit conversion to floating point.
    """

    if fill_value is None:
        return data

    if np.issubdtype(data.dtype, np.floating):
        data = np.array(
            data,
            copy=True,
        )

        data[data == fill_value] = np.nan

    return data


def _validate_dimensions(
    variable,
    data,
    dims,
):
    """
    Check that the number of dimensions matches the dimension names.
    """

    if data.ndim != len(dims):
        raise ValueError(
            f"Variable '{variable}' has shape {data.shape}, but its "
            f"registered dimensions are {dims}."
        )
    

def _build_variable_attrs(
    reader,
    variable,
    fill_value,
):
    """
    Build xarray attributes for one CALIOP variable.
    """

    attrs = {}

    if fill_value is not None:
        attrs["original_fill_value"] = fill_value

    return attrs
