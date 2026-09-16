"""Legacy automatic CALIOP path conventions for known hosts."""

import os
import socket

from twod_mcda.caliop.granule import parse_granule_time

# Get machine name
hostname = socket.gethostname()

# Initialization
CALIOP_DATA_HEAD_PATH = None
IIR_DATA_HEAD_PATH = None
CALIOP_DATA_TAIL_PATH_FMT = {}
IIR_DATA_TAIL_PATH_FMT = {}

if hostname[:5] == "icare":
    # Head paths
    CALIOP_DATA_HEAD_PATH = "/DATA/LIENS/CALIOP/"
    IIR_DATA_HEAD_PATH = "/DATA/LIENS/CALIOP/"
    # Tail paths format
    CALIOP_DATA_TAIL_PATH_FMT["L1"] = (
        "CAL_LID_L1.{version}/{year:d}/{year:d}_{month:02d}_{day:02d}/"
    )
    CALIOP_DATA_TAIL_PATH_FMT["L2_VFM"] = (
        "VFM.{version}/{year:d}/{year:d}_{month:02d}_{day:02d}/"
    )
    CALIOP_DATA_TAIL_PATH_FMT["L2_05kmMLay"] = (
        "05kmMLay.{version}/{year:d}/{year:d}_{month:02d}_{day:02d}/"
    )
    CALIOP_DATA_TAIL_PATH_FMT["L2_01kmCLay"] = (
        "01kmCLay.{version}/{year:d}/{year:d}_{month:02d}_{day:02d}/"
    )
    CALIOP_DATA_TAIL_PATH_FMT["L2_333mMLay"] = (
        "333mMLay.{version}/{year:d}/{year:d}_{month:02d}_{day:02d}/"
    )
    CALIOP_DATA_TAIL_PATH_FMT["L2_05kmALay"] = (
        "05kmALay.{version}/{year:d}/{year:d}_{month:02d}_{day:02d}/"
    )
    CALIOP_DATA_TAIL_PATH_FMT["L2_05kmAPro"] = (
        "05kmAPro.{version}/{year:d}/{year:d}_{month:02d}_{day:02d}/"
    )
    CALIOP_DATA_TAIL_PATH_FMT["L2_05kmCPro"] = (
        "05kmCPro.{version}/{year:d}/{year:d}_{month:02d}_{day:02d}/"
    )
    IIR_DATA_TAIL_PATH_FMT["L1"] = (
        "CAL_IIR_L1.{version}/{year:d}/{year:d}_{month:02d}_{day:02d}/"
    )
    IIR_DATA_TAIL_PATH_FMT["L2"] = (
        "CAL_IIR_L2.{version}/{year:d}/{year:d}_{month:02d}_{day:02d}/"
    )
elif hostname == "komputilo":
    # Head paths
    CALIOP_DATA_HEAD_PATH = "/home/ticjo/Documents/Pro/Recherche/codes/DATA/CALIPSO/"
elif hostname[:4] == "argo":
    # Head paths
    CALIOP_DATA_HEAD_PATH = "/SCF10/Data_Archive/CALIPSO/"
    # Tail paths format
    CALIOP_DATA_TAIL_PATH_FMT["L1"] = (
        "LID_L1.-{data_type}-{version}/{year:d}/{month:02d}/"
    )
else:
    # Explicit paths supplied by the pipeline remain usable on any machine.
    # Automatic path detection will raise a focused error only when requested.
    pass


def get_caliop_data_tail_path(product, version, data_type, granule):
    """
    Return tail path where the data product file is stored, according to the machine on which the
    script is runned.

    :param product: CALIOP data product ('L1', 'L2_VFM', ...)
    :param version: CALIOP version product (ex: 'V4.10')
    :param granule: 'YYYY-MM-DDThh-mm-ssZx'
    :return: tail path where the data product file is stored
    """

    granule_time = parse_granule_time(granule)
    if hostname[:5] == "icare":
        caliop_data_tail_path = CALIOP_DATA_TAIL_PATH_FMT[product].format(
            version=version.lower(),
            year=granule_time.year,
            month=granule_time.month,
            day=granule_time.day,
        )
    elif hostname == "komputilo":
        caliop_data_tail_path = ""
    elif hostname[:4] == "argo":
        caliop_data_tail_path = CALIOP_DATA_TAIL_PATH_FMT[product].format(
            data_type=data_type,
            version=version.replace(".", "-"),
            year=granule_time.year,
            month=granule_time.month,
        )
    return caliop_data_tail_path


def automatic_path_detection(product, version, data_type, granule):
    """Resolve a product directory using legacy host-specific conventions."""

    if not CALIOP_DATA_HEAD_PATH:
        raise RuntimeError(
            "CALIOP data paths are not defined for this machine; provide an "
            "explicit folder path instead."
        )

    tail_path = get_caliop_data_tail_path(
        product,
        version,
        data_type,
        granule,
    )
    return os.path.join(CALIOP_DATA_HEAD_PATH, tail_path)
