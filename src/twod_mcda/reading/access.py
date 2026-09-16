"""Open CALIOP granules and read their variables into labelled xarray datasets."""

from pathlib import Path

import xarray as xr

from twod_mcda.caliop.geography import get_prof_min_max_indexes_from_lon
from twod_mcda.reading.discovery import caliop_l1_filename
from twod_mcda.reading.native import CALIOPGranuleFile
from twod_mcda.reading.reader import CALIOPRegularGridReader
from twod_mcda.reading.variables import CALIOP_L1_PROCESSING_VARIABLES


def open_granule(
    request,
    granule,
    directory,
    profile_start=None,
    profile_end=None,
    subset_mode="profindex",
):
    """Open one CALIOP granule without loading its scientific arrays.

    :param request: resolved ``config.ProcessingRequest``
    :param granule: 'YYYY-MM-DDThh-mm-ssZx'
    :param directory: directory holding the granule file
    :param profile_start: (optional) start of the subset to read, as a profile index
                          or a longitude depending on ``subset_mode``
                          default: the first profile
    :param profile_end: (optional) end of the subset to read (included)
                        default: the end of the data
    :param subset_mode: 'profindex' if profile indexes are provided or 'longitude' if
                        longitudes are (longitudes because longitude increases or
                        decreases monotonously over one granule, unlike latitude)
                        default: 'profindex'
    """

    filepath = Path(directory) / caliop_l1_filename(granule, request.caliop_version)
    granule_file = CALIOPGranuleFile(filepath)

    try:
        prof_min, prof_max = _resolve_profile_bounds(
            granule_file,
            profile_start,
            profile_end,
            subset_mode,
        )
    except Exception:
        granule_file.close()
        raise

    return CALIOPRegularGridReader(
        granule_file,
        prof_min,
        prof_max,
        max_altitude_index=request.maximum_altitude_index,
    )


def _resolve_profile_bounds(granule_file, profile_start, profile_end, subset_mode):
    """Turn a requested subset into the first and last profile indexes to read."""

    if subset_mode == "longitude":
        longitude = granule_file.get_data("Longitude")
        return get_prof_min_max_indexes_from_lon(
            longitude,
            profile_start,
            profile_end,
        )

    if subset_mode != "profindex":
        raise ValueError(
            f"Error: subset_mode = '{subset_mode}' is not defined. "
            "Please use 'profindex' or 'longitude'\n"
        )

    if profile_start is None:
        prof_min = 0
    else:
        prof_min = int(profile_start)
        if prof_min < 0:
            prof_min += granule_file.nb_profiles

    if profile_end is None:
        prof_max = granule_file.nb_profiles - 1
    else:
        prof_max = int(profile_end)

    return prof_min, prof_max


def read_slice(granule_reader, profile_start, profile_end):
    """Read and derive the detector inputs for one profile slice."""

    arrays = {
        variable: granule_reader.get_data(variable, profile_start, profile_end)
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


def read_adjacent_profiles(request, granule, directory, profile_start, profile_end):
    """Load context profiles from one adjacent granule, then close its file."""

    with open_granule(
        request,
        granule,
        directory,
        profile_start,
        profile_end,
    ) as adjacent_granule_reader:
        adjacent_profiles = read_slice(
            adjacent_granule_reader,
            adjacent_granule_reader.prof_min,
            adjacent_granule_reader.prof_max,
        )
        adjacent_granule_path = adjacent_granule_reader.filepath

    return adjacent_profiles, adjacent_granule_path
