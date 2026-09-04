"""Load CALIOP variables into labelled xarray datasets."""

import xarray as xr

from twod_mcda.caliop.reader import CALIOPRegularGridReader
from twod_mcda.caliop.constants import CALIOP_L1_PRODUCT_TYPE
from twod_mcda.caliop.variables import CALIOP_L1_PROCESSING_VARIABLES


def open_granule(
    request,
    granule_date,
    directory,
    profile_start=None,
    profile_end=None,
    subset_mode="profindex",
):
    """Open one CALIOP granule without loading its scientific arrays."""

    return CALIOPRegularGridReader(
        product="L1",
        version=request.caliop_version,
        data_type=CALIOP_L1_PRODUCT_TYPE,
        granule_date=granule_date,
        grid="333mx30m",
        slice_start=profile_start,
        slice_end=profile_end,
        slice_start_end_type=subset_mode,
        folderpath=str(directory),
        index30m_alt_max=request.maximum_altitude_index,
    )


def read_slice(granule, profile_start, profile_end):
    """Read and derive the detector inputs for one profile slice."""

    reader = granule.select_profiles(profile_start, profile_end)
    arrays = {
        variable: reader.get_data(variable)
        for variable in CALIOP_L1_PROCESSING_VARIABLES
    }
    altitude = arrays["Lidar_Data_Altitudes"]
    altitude_values = altitude.values
    arrays["Lidar_Data_Altitudes"] = altitude.assign_coords(altitude=altitude_values)
    for name, array in arrays.items():
        if "altitude" in array.dims:
            arrays[name] = array.assign_coords(altitude=altitude_values)

    dataset = xr.Dataset(arrays)
    return dataset.set_coords(["Latitude", "Longitude", "Lidar_Data_Altitudes"])
