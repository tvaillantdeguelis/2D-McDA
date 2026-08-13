"""Refactored orchestration of the unchanged 2D-McDA algorithm."""

import numpy as np

from twod_mcda.calipso_constants import FILL_VALUE_FLOAT
from twod_mcda.config import NB_PROF_EDGE, NB_PROF_OVERLAP, NB_PROF_SLICE
from twod_mcda.io.caliop_input import (
    load_processing_variables,
    open_caliop_reader,
)
from twod_mcda.io.product_writer import write_product
from twod_mcda.merge.merge_masks import merged_feature_masks
from twod_mcda.preprocessing.neighbors import (
    append_adjacent_profiles,
    profiles_are_consecutive,
)
from twod_mcda.preprocessing.slicing import get_slice_bounds, trim_profiles
from twod_mcda.processing.channels import (
    detect_channel_features,
    detect_surfaces,
)
from twod_mcda.processing.models import ProcessingRequest, ProcessingResult, SliceData
from twod_mcda.timing import timer


PROFILE_METADATA = (
    "Profile_ID",
    "Profile_Time",
    "Profile_UTC_Time",
    "Latitude",
    "Longitude",
)

DETECTION_MASKS = (
    "Parallel_Detection_Flags_532",
    "Perpendicular_Detection_Flags_532",
    "Detection_Flags_1064",
    "Composite_Detection_Flags",
)


def _print_request(request, reader, previous_reader, next_reader):
    """Print only the input and processing settings useful to the user."""

    print("\n*****2D-McDA configuration*****")
    print(f"2D-McDA version        : {request.output_version}")
    print(f"CALIOP L1 version      : {request.caliop_version}")
    print(f"Save development data : {request.save_development_data}")
    print(f"Maximum altitude       : {request.maximum_altitude_km} km")
    print(f"Subset mode            : {request.subset_mode}")
    print(f"Subset limits          : {request.subset_start} -> {request.subset_end}")
    print("Granules:")
    if previous_reader is not None:
        print(f"  Previous : {previous_reader.filepath}")
    print(f"  Current  : {reader.filepath}")
    if next_reader is not None:
        print(f"  Next     : {next_reader.filepath}")


def _load_initial_data(request):
    """Open the main granule and load optional neighbor overlaps."""

    reader = open_caliop_reader(
        request,
        request.granule_date,
        request.current_directory,
        request.subset_start,
        request.subset_end,
        request.subset_mode,
    )
    starts, ends = get_slice_bounds(
        reader.prof_min,
        reader.prof_max,
        NB_PROF_SLICE,
        NB_PROF_OVERLAP,
    )
    granule_last_profile = reader.data_reader.nb_profiles - 1
    needs_previous = starts[0] == 0
    needs_next = ends[-1] == granule_last_profile

    previous_reader = None
    previous = None
    if needs_previous and request.previous_granule is not None:
        previous_reader = open_caliop_reader(
            request,
            request.previous_granule,
            request.previous_directory,
            -NB_PROF_OVERLAP,
            None,
        )
        previous = load_processing_variables(previous_reader)

    next_reader = None
    following = None
    if needs_next and request.next_granule is not None:
        next_reader = open_caliop_reader(
            request,
            request.next_granule,
            request.next_directory,
            None,
            NB_PROF_OVERLAP,
        )
        following = load_processing_variables(next_reader)

    _print_request(request, reader, previous_reader, next_reader)

    return reader, previous, following, starts, ends


def _empty_output(profile_count, altitude_count):
    """Allocate arrays with the product dtypes and fill values."""

    return {
        "Profile_ID": np.ones(profile_count, dtype=np.int32) * FILL_VALUE_FLOAT,
        "Profile_Time": np.ones(profile_count, dtype=np.float64) * FILL_VALUE_FLOAT,
        "Profile_UTC_Time": np.ones(profile_count, dtype=np.float64) * FILL_VALUE_FLOAT,
        "Latitude": np.ma.ones(profile_count, dtype=np.float32) * FILL_VALUE_FLOAT,
        "Longitude": np.ma.ones(profile_count, dtype=np.float32) * FILL_VALUE_FLOAT,
        "Parallel_Detection_Flags_532": np.zeros(
            (profile_count, altitude_count), dtype=np.uint8
        ),
        "Perpendicular_Detection_Flags_532": np.zeros(
            (profile_count, altitude_count), dtype=np.uint8
        ),
        "Detection_Flags_1064": np.zeros(
            (profile_count, altitude_count), dtype=np.uint8
        ),
        "Composite_Detection_Flags": np.zeros(
            (profile_count, altitude_count), dtype=np.uint8
        ),
    }


def _load_slice(request, profile_min, profile_max, previous, following, file_max):
    reader = open_caliop_reader(
        request,
        request.granule_date,
        request.current_directory,
        profile_min,
        profile_max,
    )
    data = load_processing_variables(reader)
    result = SliceData(input=data)

    if profile_min == 0 and previous is not None:
        time_gap = np.abs(data["Profile_Time"][0] - previous["Profile_Time"][-1])
        print(
            "\tTime between last profile of previous file and first profile "
            f"of current file = {time_gap:.2f} s"
        )
        if profiles_are_consecutive(previous["Profile_Time"][-1], data["Profile_Time"][0]):
            print("\tAppend previous granule")
            result.input = append_adjacent_profiles(data, previous, "start")
            result.previous_profiles_used = True
        else:
            print("\tPrevious granule does not seem consecutive. First profiles not processed.")
    elif profile_min == 0:
        print("\tNo previous file to load. First profiles not processed.")
    elif profile_max == file_max and following is not None:
        time_gap = np.abs(following["Profile_Time"][0] - data["Profile_Time"][-1])
        print(
            "\tTime between last profile of current file and first profile "
            f"of next file = {time_gap:.2f} s"
        )
        if profiles_are_consecutive(data["Profile_Time"][-1], following["Profile_Time"][0]):
            print("\tAppend next granule")
            result.input = append_adjacent_profiles(data, following, "end")
            result.next_profiles_used = True
        else:
            print("\tNext granule does not seem consecutive. Last profiles not processed.")
    elif profile_max == file_max:
        print("\tNo next file to load. Last profiles not processed.")

    return result


def _trim_neighbor_profiles(slice_data):
    if slice_data.previous_profiles_used:
        side = "start"
    elif slice_data.next_profiles_used:
        side = "end"
    else:
        return

    print(f"\n\n*****Remove profiles from {side} adjacent file...*****")
    for mapping in (slice_data.input, slice_data.masks, slice_data.development):
        for name, values in mapping.items():
            mapping[name] = trim_profiles(values, NB_PROF_OVERLAP, side)


def _store_slice(output, slice_data, profile_min, profile_max, file_min):
    store_min = profile_min - file_min
    store_max = profile_max - file_min

    for name in PROFILE_METADATA:
        output[name][store_min : store_max + 1] = np.copy(slice_data.input[name])

    if slice_data.previous_profiles_used:
        output_slice = slice(store_min, store_max - NB_PROF_EDGE + 1)
        input_slice = slice(None, -NB_PROF_EDGE)
    elif slice_data.next_profiles_used:
        output_slice = slice(store_min + NB_PROF_EDGE, store_max + 1)
        input_slice = slice(NB_PROF_EDGE, None)
    else:
        output_slice = slice(store_min + NB_PROF_EDGE, store_max - NB_PROF_EDGE + 1)
        input_slice = slice(NB_PROF_EDGE, -NB_PROF_EDGE)

    for name in DETECTION_MASKS:
        output[name][output_slice, :] = slice_data.masks[name][input_slice, :]


def _store_development(
    output,
    slice_development,
    profile_min,
    profile_max,
    file_min,
    profile_count,
):
    """Assemble development arrays using one shared profile dimension."""

    store_min = profile_min - file_min
    store_max = profile_max - file_min + 1

    for name, values in slice_development.items():
        values = np.asanyarray(values)
        if values.ndim == 2:
            profile_axis = 0
        elif values.ndim == 3:
            profile_axis = 1
        else:
            raise ValueError(
                f"Unsupported development array shape for {name!r}: "
                f"{values.shape}."
            )

        if name not in output:
            shape = list(values.shape)
            shape[profile_axis] = profile_count
            fill_value = FILL_VALUE_FLOAT if values.dtype.kind == "f" else 0
            if np.ma.isMaskedArray(values):
                output[name] = np.ma.masked_all(shape, dtype=values.dtype)
                output[name].set_fill_value(fill_value)
            else:
                output[name] = np.full(shape, fill_value, dtype=values.dtype)

        if profile_axis == 0:
            output[name][store_min:store_max, :] = values
        else:
            output[name][:, store_min:store_max, :] = values


def _process_slice(request, profile_min, profile_max, previous, following, reader):
    print(
        "\n\n############################################################\n"
        f"Processing slice with profile indexes from {profile_min} to {profile_max}..."
    )
    print("\n\n*****Load slice data...*****")
    with timer("Load slice data"):
        slice_data = _load_slice(
            request,
            profile_min,
            profile_max,
            previous,
            following,
            reader.prof_max,
        )

    print("\n\n*****Surface detection...*****")
    with timer("Surface detection"):
        surfaces = detect_surfaces(slice_data.input)

    print("\n\n*****Feature detection...*****")
    with timer("Feature detection"):
        slice_data.masks, slice_data.development = detect_channel_features(
            slice_data.input,
            surfaces,
        )

    _trim_neighbor_profiles(slice_data)

    print("\n\n*****Merged 3 channels feature detection...*****")
    with timer("Merged 3 channels feature detection"):
        slice_data.masks["Composite_Detection_Flags"] = merged_feature_masks(
            slice_data.masks["Parallel_Detection_Flags_532"],
            slice_data.masks["Perpendicular_Detection_Flags_532"],
            slice_data.masks["Detection_Flags_1064"],
        )

    return slice_data


def process_granule(request: ProcessingRequest):
    """Process one resolved CALIOP granule request."""

    print("\n*****CALIOP L1 data...*****")
    with timer("CALIOP L1 data"):
        reader, previous, following, starts, ends = _load_initial_data(request)
        profile_count = reader.prof_max - reader.prof_min + 1
        altitude = reader.get_data("Lidar_Data_Altitudes")
        output = _empty_output(profile_count, altitude.size)
        print(f"\tNumber of profiles to process: {profile_count}")

    development = {}

    print("\n*****Apply algorithm by slice...*****")
    with timer("Apply algorithm by slice"):
        for index, (profile_min, profile_max) in enumerate(zip(starts, ends), start=1):
            with timer(f"Process slice {index:d}/{starts.size:d}"):
                print(f"\n\n*****Slice {index:d}/{starts.size:d}...*****")
                slice_data = _process_slice(
                    request,
                    profile_min,
                    profile_max,
                    previous,
                    following,
                    reader,
                )
                if request.save_development_data:
                    _store_development(
                        development,
                        slice_data.development,
                        profile_min,
                        profile_max,
                        reader.prof_min,
                        profile_count,
                    )

                print("\n\n*****Copy slice data to the whole data arrays...*****")
                with timer("Copy slice data to the whole data arrays"):
                    _store_slice(
                        output,
                        slice_data,
                        profile_min,
                        profile_max,
                        reader.prof_min,
                    )

    result = ProcessingResult(
        data=output,
        development=development,
        altitude=altitude,
        longitude_min=reader.lon_min,
        longitude_max=reader.lon_max,
    )

    print("\n\n############################################################\n*****Save data in netCDF file...*****")
    with timer("Save data in netCDF file"):
        return write_product(request, result)
