"""Open CALIOP granules and read their variables into labelled xarray datasets."""

from pathlib import Path

import xarray as xr

from twod_mcda.caliop.geography import get_prof_min_max_indexes_from_lon
from twod_mcda.reading.discovery import caliop_l1_filename
from twod_mcda.reading.native import CALIOPGranuleFile
from twod_mcda.reading.reader import CALIOPRegularGridReader
from twod_mcda.reading.variables import CALIOP_L1_PROCESSING_VARIABLES


def open_granule(request):
    """Open the primary granule described by a processing request.

    Convenience wrapper around ``_open_granule_file`` for the common case:
    opening the granule ``request`` itself points to, with its own subset
    bounds. Adjacent (previous/next) granules are opened by
    ``read_adjacent_profiles``, which calls ``_open_granule_file`` directly
    since it needs a different granule, directory, and subset bounds.
    """

    return _open_granule_file(
        request.current_granule_directory,
        request.granule,
        request.caliop_version,
        request.maximum_altitude_index,
        request.subset_start,
        request.subset_end,
        request.subset_mode,
    )


def _open_granule_file(
    directory,
    granule,
    caliop_version,
    max_altitude_index,
    profile_start=None,
    profile_end=None,
    subset_mode="profindex",
):
    """Open one CALIOP granule file without loading its scientific arrays.

    :param directory: directory holding the granule file
    :param granule: 'YYYY-MM-DDThh-mm-ssZx'
    :param caliop_version: CALIOP data version, used to build the file name
    :param max_altitude_index: highest altitude index kept when reading arrays
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

    filepath = Path(directory) / caliop_l1_filename(granule, caliop_version)
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
        max_altitude_index=max_altitude_index,
    )


def _resolve_profile_bounds(granule_file, profile_start, profile_end, subset_mode):
    """Turn a requested subset into the first and last profile indexes to read."""

    if subset_mode == "longitude":
        longitude = granule_file.get_data("Longitude")
        prof_min, prof_max = get_prof_min_max_indexes_from_lon(
            longitude,
            profile_start,
            profile_end,
        )
    elif subset_mode == "profindex":
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
            if prof_max < 0:
                prof_max += granule_file.nb_profiles
    else:
        raise ValueError(
            f"Error: subset_mode = '{subset_mode}' is not defined. "
            "Please use 'profindex' or 'longitude'\n"
        )

    if prof_max <= prof_min:
        raise ValueError(
            f"prof_max (= {prof_max}) <= prof_min (= {prof_min}); "
            "please check profile_start and profile_end"
        )

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

    with _open_granule_file(
        directory,
        granule,
        request.caliop_version,
        request.maximum_altitude_index,
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
