"""Small HDF4 writer kept for the legacy processing path."""

from pathlib import Path

import numpy as np


class SDSData:
    """Description of one Scientific Data Set written to an HDF4 file."""

    def __init__(self, key, data, fillvalue=None):
        self.key = key
        self.data = data
        self.fillvalue = fillvalue
        self.description = None
        self.units = None
        self.valid_range = None
        self.dim_labels = None


def _hdf_type(dtype, sdc):
    """Return the pyhdf type matching a NumPy dtype."""

    dtype = np.dtype(dtype)
    types = {
        ("i", 1): sdc.INT8,
        ("u", 1): sdc.UINT8,
        ("i", 2): sdc.INT16,
        ("u", 2): sdc.UINT16,
        ("i", 4): sdc.INT32,
        ("u", 4): sdc.UINT32,
        ("f", 4): sdc.FLOAT32,
        ("f", 8): sdc.FLOAT64,
    }

    try:
        return types[(dtype.kind, dtype.itemsize)]
    except KeyError as exc:
        raise TypeError(f"Unsupported HDF4 dtype: {dtype}") from exc


def write_hdf(filename, params):
    """Write the legacy ``SDSData`` mapping to an HDF4 file."""

    try:
        from pyhdf.SD import SD, SDC
    except ImportError as exc:
        raise RuntimeError(
            "pyhdf is required to write the legacy HDF4 output. "
            "Install the project environment from environment.yml."
        ) from exc

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hdf = SD(str(output_path), SDC.WRITE | SDC.CREATE | SDC.TRUNC)

    try:
        for param in params.values():
            data = np.asanyarray(param.data)
            hdf_type = _hdf_type(data.dtype, SDC)
            sds = hdf.create(param.key, hdf_type, data.shape)

            if param.fillvalue is not None:
                sds.setfillvalue(param.fillvalue)

            if np.ma.isMaskedArray(data):
                fillvalue = param.fillvalue
                if fillvalue is None:
                    fillvalue = np.nan if data.dtype.kind == "f" else 0
                data = data.filled(fillvalue)

            sds[:] = data

            if param.description is not None:
                sds.description = param.description
            if param.units is not None:
                sds.units = param.units
            if param.valid_range is not None:
                sds.setrange(*param.valid_range)
            if param.dim_labels is not None:
                for index, label in enumerate(param.dim_labels):
                    sds.dim(index).setname(label)

            sds.endaccess()
    finally:
        hdf.end()

    print(f"{output_path} created.")

