"""Schema-driven primitives for writing self-describing netCDF-4 files."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from uuid import uuid4

from netCDF4 import Dataset
import numpy as np
import xarray as xr


@dataclass
class NetCDFVariable:
    """Description of one variable in a netCDF product."""

    name: str
    data: xr.DataArray | np.ndarray
    dimensions: tuple[str, ...] = ()
    fill_value: int | float | None = None
    attributes: dict[str, object] = field(default_factory=dict)
    compress: bool = True

    @property
    def key(self):
        """Backward-compatible alias for the variable name."""

        return self.name


def _dimensions(variables):
    """Infer dimension sizes and reject inconsistent variable shapes."""

    dimensions = {}
    for variable in variables:
        data = np.asanyarray(variable.data)
        if data.ndim != len(variable.dimensions):
            raise ValueError(
                f"Variable {variable.name!r} has {data.ndim} dimensions, "
                f"but {len(variable.dimensions)} names were provided."
            )

        for name, size in zip(variable.dimensions, data.shape):
            if name in dimensions and dimensions[name] != size:
                raise ValueError(
                    f"Dimension {name!r} has conflicting sizes "
                    f"{dimensions[name]} and {size}."
                )
            dimensions[name] = size

    return dimensions


def _attributes(variable, dtype):
    """Return attributes with numeric ranges encoded in the variable dtype."""

    attributes = {
        key: value
        for key, value in variable.attributes.items()
        if value is not None
    }
    for name in ("valid_range", "flag_values", "flag_masks"):
        if name in attributes:
            attributes[name] = np.asarray(attributes[name], dtype=dtype)
    return attributes


def _storage_data(specification):
    """Convert xarray missing values to the variable's netCDF fill value."""

    data = np.asanyarray(specification.data)
    if specification.fill_value is not None:
        data = np.ma.masked_equal(data, specification.fill_value)
        if data.dtype.kind in {"f", "c"}:
            data = np.ma.masked_invalid(data)
    return data


def write_netcdf(filename, variables, global_attributes=None):
    """Write variables atomically in netCDF-4 format.

    The resulting file uses the netCDF-4 enhanced data model with lossless
    DEFLATE compression and the shuffle filter for non-scalar arrays.
    """

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    variables = tuple(variables)
    dimensions = _dimensions(variables)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid4().hex}.tmp"
    )

    try:
        with Dataset(temporary_path, mode="w", format="NETCDF4") as dataset:
            dataset.setncatts(global_attributes or {})

            for name, size in dimensions.items():
                dataset.createDimension(name, size)

            for specification in variables:
                data = _storage_data(specification)
                options = {"fill_value": specification.fill_value}
                if specification.compress and data.ndim and data.size > 1:
                    options.update(
                        compression="zlib",
                        complevel=4,
                        shuffle=True,
                    )

                variable = dataset.createVariable(
                    specification.name,
                    data.dtype,
                    specification.dimensions,
                    **options,
                )
                variable.setncatts(_attributes(specification, data.dtype))
                if data.ndim:
                    variable[:] = data
                else:
                    variable.assignValue(data.item())

        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"{output_path} created.")
