from datetime import datetime
import time

from .io.granule_finder import (
    find_granule_file,
    find_neighbor_granules,
)
from .io.caliop_l1_reader import read_caliop_l1
from .io.caliop_l1_variables import CALIOP_L1_NATIVE_VARIABLES
from .version import get_full_version


def process_granule(cfg):

    start_time = datetime.now().astimezone()
    start_tic = time.perf_counter()
    print(f"Start time: {start_time}")

    
    # Get algorithm version (from git)
    VERSION_2D_MCDA = get_full_version()
    print(f"2D-McDA version: {VERSION_2D_MCDA}")


    # Finding neighbor granules
    granule_time = datetime.strptime(cfg["granule"], "%Y-%m-%dT%H-%M-%SZN")
    current_file = find_granule_file(cfg, granule_time)
    previous_file, next_file = find_neighbor_granules(cfg, current_file)
    print(f"Previous granule: {previous_file}")
    print(f"Current granule : {current_file}")
    print(f"Next granule    : {next_file}")


    raw_data = read_caliop_l1(
        current_file,
        variables=CALIOP_L1_NATIVE_VARIABLES,
    )

    # regular_data = preprocess_caliop_l1(
    #     raw_data,
    #     target_grid="333mx30m",
    # )

    # slice_bounds = get_slice_bounds(
    #     reader.prof_min,
    #     reader.prof_max,
    #     NB_PROF_SLICE,
    #     NB_PROF_OVERLAP,
    # )

    # for profile_start, profile_end in zip(*slice_bounds):
    #     slice_result = process_slice(
    #         cfg=cfg,
    #         current_file=current_file,
    #         previous_file=previous_file,
    #         next_file=next_file,
    #         profile_start=profile_start,
    #         profile_end=profile_end,
    #     )

    #     store_slice_result(
    #         output_data,
    #         slice_result,
    #     )

    # write_output(
    #     cfg,
    #     output_data,
    # )

    end_time = datetime.now().astimezone()
    total_time = time.perf_counter() - start_tic

    print(f"End time: {end_time}")
    print(f"Total runtime: {total_time:.1f} s")

