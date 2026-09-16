"""Discover CALIOP granules and their neighbors in configured archives."""

from datetime import datetime, timedelta
from pathlib import Path

from twod_mcda.caliop.constants import (
    CAL_LID_FILENAME_FMT,
    CAL_LID_L1_FILENAME_PATTERN,
    CALIOP_L1_PRODUCT_TYPE,
    GRANULE_TIME_FMT,
)


def parse_granule_time(granule):
    """
    Parse a granule into its observation datetime.

    Parameters
    ----------
    granule : str
        Granule, e.g. "2013-01-11T03-25-54ZD".

    Returns
    -------
    datetime
        Observation start time.
    """

    return datetime.strptime(
        granule[:-2],  # Remove the trailing 'ZD' or 'ZN'
        GRANULE_TIME_FMT,
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

    match = CAL_LID_L1_FILENAME_PATTERN.match(filename.name)

    if match is None:
        raise ValueError(f"Invalid CALIOP filename format: {filename.name}")

    return datetime.strptime(
        match.group(1),
        GRANULE_TIME_FMT,
    )


def extract_granule(filename):
    """
    Extract the full granule, including the day/night flag,
    from a CALIOP L1 filename.

    Example:
        CAL_LID_L1-Standard-V5-00.2013-01-11T03-25-54ZD.hdf
        -> "2013-01-11T03-25-54ZD"
    """

    match = CAL_LID_L1_FILENAME_PATTERN.match(filename.name)

    if match is None:
        raise ValueError(f"Invalid CALIOP filename format: {filename.name}")

    return match.group(1) + match.group(2)


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

    root_directory = Path(cal_cfg["root_directory"])

    relative_path = cal_cfg["path_format"].format(
        version=cal_cfg["version"],
        year=date.year,
        month=date.month,
        day=date.day,
    )

    return root_directory / relative_path


def find_granule_file(cfg):
    """
    Build the CALIOP file path corresponding to the configured granule.

    Parameters
    ----------
    cfg : dict
        Processing configuration, including the "granule",
        e.g. "2013-01-11T03-25-54ZD".

    Returns
    -------
    Path
        CALIOP granule file.
    """

    granule = cfg["granule"]
    folder = get_caliop_folder(cfg, parse_granule_time(granule))

    cal_cfg = cfg["cal_lid_l1"]
    version = f"V{cal_cfg['version']}".replace(".", "-")
    filename = CAL_LID_FILENAME_FMT % (
        "L1",
        CALIOP_L1_PRODUCT_TYPE,
        version,
        granule,
    )
    file = folder / filename

    if not file.exists():
        raise FileNotFoundError(f"CALIOP granule not found: {file}")

    return file


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
        CALIOP granules sorted chronologically.
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

            files.extend(folder.glob("CAL_LID_L1-*.hdf"))

        current_date += timedelta(days=1)

    # Sort files according to their observation time
    files = sorted(
        set(files),
        key=extract_granule_time,
    )

    # Keep only granules inside the requested period
    granule_files = [
        file for file in files if start_date <= extract_granule_time(file) <= end_date
    ]

    return [extract_granule(file) for file in granule_files]


def find_neighbor_granules(cfg):
    """
    Find previous and next CALIOP granules.

    The search is based on timestamps extracted from filenames.
    Temporal continuity is checked later after reading the data.

    Parameters
    ----------
    cfg : dict
        Processing configuration, including the "granule",
        e.g. "2013-01-11T03-25-54ZD".

    Returns
    -------
    previous_file : Path or None
        Previous granule file.

    next_file : Path or None
        Next granule file.
    """

    current_time = parse_granule_time(cfg["granule"])

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

            files.extend(folder.glob("CAL_LID_L1-*.hdf"))

    previous_file = None
    previous_time = None
    next_file = None
    next_time = None

    for file in files:

        file_time = extract_granule_time(file)

        if file_time < current_time:
            if previous_time is None or file_time > previous_time:
                previous_file = file
                previous_time = file_time
        elif file_time > current_time:
            if next_time is None or file_time < next_time:
                next_file = file
                next_time = file_time

    return previous_file, next_file
