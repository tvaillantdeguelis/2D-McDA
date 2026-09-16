"""Parse the CALIOP granule string used throughout the configuration and workflow."""

from datetime import datetime

from twod_mcda.caliop.constants import GRANULE_TIME_FMT


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
