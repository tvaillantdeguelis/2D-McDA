"""
Functions to find CALIOP granule files and neighboring granules.
"""

from datetime import datetime, timedelta
from pathlib import Path
import re


GRANULE_PATTERN = re.compile(
    r"CAL_LID_L1-[^-]+-V(\d+)-(\d+)\."
    r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})Z[DN]\.hdf"
)


def extract_granule_time(filename):
    """
    Extract observation datetime from a CALIOP L1 filename.

    Example:
        CAL_LID_L1-Standard-V5-00.2013-01-11T03-25-54ZD.hdf

    Returns
    -------
    datetime
        Observation start time extracted from filename.
    """

    match = GRANULE_PATTERN.match(filename.name)

    if match is None:
        raise ValueError(
            f"Invalid CALIOP filename format: {filename.name}"
        )

    return datetime.strptime(
        match.group(3),
        "%Y-%m-%dT%H-%M-%S",
    )


def get_caliop_folder(cfg, date):
    """
    Build the CALIOP directory corresponding to a given date.

    Parameters
    ----------
    cfg : dict
        Processing configuration.

    date : datetime
        Date used to build the directory path.

    Returns
    -------
    Path
        CALIOP directory path.
    """

    cal_cfg = cfg["cal_lid_l1"]

    folder = Path(cal_cfg["folder"])

    relative_path = cal_cfg["path_format"].format(
        version=cal_cfg["version"],
        year=date.year,
        month=date.month,
        day=date.day,
    )

    return folder / relative_path


def find_granule_file(cfg, granule_time):
    """
    Find the CALIOP file corresponding to a granule timestamp.

    Parameters
    ----------
    cfg : dict
        Processing configuration.

    granule_time : datetime
        Granule observation start time.

    Returns
    -------
    Path
        CALIOP granule file.
    """

    folder = get_caliop_folder(cfg, granule_time)

    if not folder.exists():
        raise FileNotFoundError(
            f"CALIOP directory not found: {folder}"
        )

    for file in folder.glob("CAL_LID_L1-*.hdf"):

        if extract_granule_time(file) == granule_time:
            return file

    raise FileNotFoundError(
        f"CALIOP granule not found for time: {granule_time}"
    )


def find_granules_between_dates(cfg, start_date, end_date):
    """
    Find all CALIOP granules between two dates.

    Parameters
    ----------
    cfg : dict
        Processing configuration.

    start_date : datetime
        Start of the search period.

    end_date : datetime
        End of the search period.

    Returns
    -------
    list of str
        CALIOP granule identifiers sorted chronologically.
    """

    files = []

    # Search all directories between start_date and end_date.
    current_date = start_date

    while current_date.date() <= end_date.date():

        folder = get_caliop_folder(
            cfg,
            current_date,
        )

        if folder.exists():

            files.extend(
                folder.glob("CAL_LID_L1-*.hdf")
            )

        current_date += timedelta(days=1)


    # Sort files according to their observation time
    files = sorted(
        set(files),
        key=extract_granule_time,
    )


    # Keep only granules inside the requested period
    granule_files = [
        file
        for file in files
        if start_date <= extract_granule_time(file) <= end_date
    ]


    return [
        extract_granule_time(file).strftime("%Y-%m-%dT%H-%M-%SZN")
        for file in granule_files
    ]


def find_neighbor_granules(cfg, current_file):
    """
    Find previous and next CALIOP granules.

    The search is based on timestamps extracted from filenames.
    Temporal continuity is checked later after reading the data.

    Parameters
    ----------
    cfg : dict
        Processing configuration.

    current_file : Path
        Current CALIOP granule file.

    Returns
    -------
    previous_file : Path or None
        Previous granule file.

    next_file : Path or None
        Next granule file.
    """

    current_time = extract_granule_time(current_file)


    # Search one day before, current day, and one day after.
    # This handles granules crossing midnight.
    search_dates = [
        current_time - timedelta(days=1),
        current_time,
        current_time + timedelta(days=1),
    ]


    files = []

    for date in search_dates:

        folder = get_caliop_folder(cfg, date)

        if folder.exists():

            files.extend(
                folder.glob("CAL_LID_L1-*.hdf")
            )


    # Sort granules chronologically
    files = sorted(
        set(files),
        key=extract_granule_time,
    )


    previous_file = None
    next_file = None


    for index, file in enumerate(files):

        if file == current_file:

            if index > 0:
                previous_file = files[index - 1]

            if index < len(files) - 1:
                next_file = files[index + 1]

            break


    return previous_file, next_file