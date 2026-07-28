from datetime import datetime
import time

from . import __version__
from .models.granule import Granule
from .version import get_full_version
from .io.granule_finder import find_granule_file, find_neighbor_granules
# from io.caliop_l1_reader import read_caliop_l1


def process_granule(cfg):

    start_time = datetime.now().astimezone()
    start_tic = time.perf_counter()
    print(f"Start time: {start_time}")

    
    # Get algorithm version (from git)
    VERSION_2D_MCDA = get_full_version()
    print(f"2D-McDA version: {VERSION_2D_MCDA}")


    # Finding neighbor granules
    granule_time = datetime.strptime(cfg["granule"]["granule"], "%Y-%m-%dT%H-%M-%SZN")
    current_file = find_granule_file(cfg, granule_time)
    previous_file, next_file = find_neighbor_granules(cfg, current_file)
    print(f"Previous granule: {previous_file}")
    print(f"Current granule : {current_file}")
    print(f"Next granule    : {next_file}")


    # # Reading CALIOP L1 data
    # section_start = time.perf_counter()

    # data = read_caliop_l1(cfg)

    # print(f"Reading CALIOP L1 data: {time.perf_counter() - section_start:.1f} s")


    # data = detect_surface(data)

    # data = detect_features(data)

    # data = merge_feature_masks(data)

    # if config.processing.make_classification:
    #     data = classify_features(data)

    # write_outputs(data, config)

    end_time = datetime.now().astimezone()
    total_time = time.perf_counter() - start_tic

    print(f"End time: {end_time}")
    print(f"Total runtime: {total_time:.1f} s")
