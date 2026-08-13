"""Load the CALIOP arrays required by the unchanged 2D-McDA algorithm."""

from twod_mcda.io.calipso_reader import CALIOPRegularGridReader
from twod_mcda.io.caliop_l1_variables import CALIOP_L1_PROCESSING_VARIABLES


CALIOP_L1_PRODUCT_TYPE = "Standard"


def open_caliop_reader(
    request,
    granule_date,
    directory,
    profile_start=None,
    profile_end=None,
    subset_mode="profindex",
):
    """Create the native-grid reader used by the current algorithm."""

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


def load_processing_variables(reader):
    """Read native and derived arrays required by the detectors."""

    return {
        variable: reader.get_data(variable)
        for variable in CALIOP_L1_PROCESSING_VARIABLES
    }
