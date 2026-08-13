"""Top-level processing pipeline."""

from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import re
import time

import numpy as np

from .caliop.constants import FILL_VALUE_FLOAT
from .caliop.discovery import find_granule_file, find_neighbor_granules
from .caliop.input import open_granule, read_slice
from .output.product import write_product
from .utils.timing import timer
from .workflow.models import ProcessingRequest, ProcessingResult, SliceData
from .workflow.neighbors import (
    append_adjacent_profiles,
    profiles_are_consecutive,
)
from .workflow.processing import process_slice
from .workflow.settings import NB_PROF_EDGE, NB_PROF_OVERLAP, NB_PROF_SLICE
from .workflow.slicing import plan_slices
from .version import get_full_version


_GRANULE_ID_PATTERN = re.compile(
    r"\.(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z[DN])\.hdf$"
)

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


def _granule_id(file_path):
    """Extract the full granule identifier, including day/night."""

    if file_path is None:
        return None

    file_path = Path(file_path)
    match = _GRANULE_ID_PATTERN.search(file_path.name)
    if match is None:
        raise ValueError(f"Invalid CALIOP filename format: {file_path.name}")

    return match.group(1)


def _normalized_version(version):
    """Return a version with the upper-case prefix used in product metadata."""

    if version[:1].lower() == "v":
        return f"V{version[1:]}"
    return f"V{version}"


def _altitude_index(max_altitude_km):
    """Map a supported maximum altitude to the regular-grid index."""

    if max_altitude_km == 30:
        return 600
    if max_altitude_km == 40:
        return None

    raise ValueError(
        "processing.max_altitude_km must be either 30 or 40."
    )


def _output_directory(output_cfg, granule_date, version):
    """Build the output directory from the configured root and path format."""

    relative_path = output_cfg["path_format"].format(
        version=version.removeprefix("V"),
        year=granule_date.year,
        month=granule_date.month,
        day=granule_date.day,
    )

    return Path(output_cfg["root_directory"]) / relative_path


def resolve_processing_request(cfg):
    """Resolve input paths and build a processing request from configuration."""

    granule_time = datetime.strptime(
        cfg["granule"][:19],
        "%Y-%m-%dT%H-%M-%S",
    )
    current_file = find_granule_file(cfg, granule_time)
    previous_file, next_file = find_neighbor_granules(cfg, current_file)

    processing_cfg = cfg.get("processing", {})
    output_cfg = cfg["output"]
    subset_cfg = cfg.get("subset", {})
    caliop_cfg = cfg["cal_lid_l1"]
    version = _normalized_version(get_full_version())
    granule_date = _granule_id(current_file)
    granule_time = datetime.strptime(
        granule_date[:19],
        "%Y-%m-%dT%H-%M-%S",
    )

    return ProcessingRequest(
        granule_date=granule_date,
        caliop_version=_normalized_version(str(caliop_cfg["version"])),
        current_directory=Path(current_file).parent,
        previous_granule=_granule_id(previous_file),
        previous_directory=(
            Path(previous_file).parent if previous_file is not None else None
        ),
        next_granule=_granule_id(next_file),
        next_directory=(
            Path(next_file).parent if next_file is not None else None
        ),
        subset_mode=subset_cfg.get("mode", "profindex"),
        subset_start=subset_cfg.get("start"),
        subset_end=subset_cfg.get("end"),
        save_development_data=processing_cfg.get(
            "save_development_data", False
        ),
        output_version=version,
        output_product_type=output_cfg.get("product_type", "Dev"),
        output_directory=_output_directory(output_cfg, granule_time, version),
        maximum_altitude_km=processing_cfg["max_altitude_km"],
        maximum_altitude_index=_altitude_index(
            processing_cfg["max_altitude_km"]
        ),
    )


def _print_request(request, granule, previous_granule, next_granule):
    """Print only the input and processing settings useful to the user."""

    print("\n*****2D-McDA configuration*****")
    print(f"2D-McDA version        : {request.output_version}")
    print(f"CALIOP L1 version      : {request.caliop_version}")
    print(f"Save development data : {request.save_development_data}")
    print(f"Maximum altitude       : {request.maximum_altitude_km} km")
    print(f"Subset mode            : {request.subset_mode}")
    print(f"Subset limits          : {request.subset_start} -> {request.subset_end}")
    print("Granules:")
    if previous_granule is not None:
        print(f"  Previous : {previous_granule.filepath}")
    print(f"  Current  : {granule.filepath}")
    if next_granule is not None:
        print(f"  Next     : {next_granule.filepath}")


def _open_inputs(request, stack):
    """Open the current granule and read only the optional neighbor overlaps."""

    granule = stack.enter_context(
        open_granule(
            request,
            request.granule_date,
            request.current_directory,
            request.subset_start,
            request.subset_end,
            request.subset_mode,
        )
    )
    starts, ends = plan_slices(
        granule.prof_min,
        granule.prof_max,
        NB_PROF_SLICE,
        NB_PROF_OVERLAP,
    )
    granule_last_profile = granule.data_reader.nb_profiles - 1

    previous_granule = None
    previous = None
    if starts[0] == 0 and request.previous_granule is not None:
        previous_granule = stack.enter_context(
            open_granule(
                request,
                request.previous_granule,
                request.previous_directory,
                -NB_PROF_OVERLAP,
                None,
            )
        )
        previous = read_slice(
            previous_granule,
            previous_granule.prof_min,
            previous_granule.prof_max,
        )

    next_granule = None
    following = None
    if ends[-1] == granule_last_profile and request.next_granule is not None:
        next_granule = stack.enter_context(
            open_granule(
                request,
                request.next_granule,
                request.next_directory,
                None,
                NB_PROF_OVERLAP - 1,
            )
        )
        following = read_slice(
            next_granule,
            next_granule.prof_min,
            next_granule.prof_max,
        )

    _print_request(request, granule, previous_granule, next_granule)
    return granule, previous, following, starts, ends


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


def _read_processing_slice(
    profile_min,
    profile_max,
    granule,
    previous,
    following,
):
    """Read one current-granule slice and add a neighbor at file edges."""

    data = read_slice(granule, profile_min, profile_max)
    slice_data = SliceData(input=data)
    granule_last_profile = granule.data_reader.nb_profiles - 1

    if profile_min == 0 and previous is not None:
        time_gap = np.abs(data["Profile_Time"][0] - previous["Profile_Time"][-1])
        print(
            "\tTime between last profile of previous file and first profile "
            f"of current file = {time_gap:.2f} s"
        )
        if profiles_are_consecutive(previous["Profile_Time"][-1], data["Profile_Time"][0]):
            print("\tAppend previous granule")
            slice_data.input = append_adjacent_profiles(data, previous, "start")
            slice_data.previous_profiles_used = True
        else:
            print("\tPrevious granule does not seem consecutive. First profiles not processed.")
    elif profile_min == 0:
        print("\tNo previous file to load. First profiles not processed.")
    elif profile_max == granule_last_profile and following is not None:
        time_gap = np.abs(following["Profile_Time"][0] - data["Profile_Time"][-1])
        print(
            "\tTime between last profile of current file and first profile "
            f"of next file = {time_gap:.2f} s"
        )
        if profiles_are_consecutive(data["Profile_Time"][-1], following["Profile_Time"][0]):
            print("\tAppend next granule")
            slice_data.input = append_adjacent_profiles(data, following, "end")
            slice_data.next_profiles_used = True
        else:
            print("\tNext granule does not seem consecutive. Last profiles not processed.")
    elif profile_max == granule_last_profile:
        print("\tNo next file to load. Last profiles not processed.")

    return slice_data


def _store_slice(output, slice_data, profile_min, profile_max, file_min):
    store_min = profile_min - file_min
    store_max = profile_max - file_min

    for name in PROFILE_METADATA:
        output[name][store_min:store_max + 1] = np.copy(slice_data.input[name])

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


def assemble_results(output, development, altitude, granule):
    """Build the complete product payload from assembled slice outputs."""

    return ProcessingResult(
        data=output,
        development=development,
        altitude=altitude,
        longitude_min=granule.lon_min,
        longitude_max=granule.lon_max,
    )


def _execute_pipeline(request):
    with ExitStack() as stack:
        print("\n*****CALIOP L1 data...*****")
        with timer("CALIOP L1 data"):
            granule, previous, following, starts, ends = _open_inputs(
                request,
                stack,
            )
            profile_count = granule.prof_max - granule.prof_min + 1
            altitude = granule.get_data("Lidar_Data_Altitudes")
            output = _empty_output(profile_count, altitude.size)
            print(f"\tNumber of profiles to process: {profile_count}")

        development = {}
        print("\n*****Apply algorithm by slice...*****")
        with timer("Apply algorithm by slice"):
            for index, (profile_min, profile_max) in enumerate(
                zip(starts, ends),
                start=1,
            ):
                with timer(f"Process slice {index:d}/{starts.size:d}"):
                    print(f"\n\n*****Slice {index:d}/{starts.size:d}...*****")
                    print("\n\n*****Load slice data...*****")
                    with timer("Load slice data"):
                        slice_data = _read_processing_slice(
                            profile_min,
                            profile_max,
                            granule,
                            previous,
                            following,
                        )
                    slice_data = process_slice(slice_data)
                    if request.save_development_data:
                        _store_development(
                            development,
                            slice_data.development,
                            profile_min,
                            profile_max,
                            granule.prof_min,
                            profile_count,
                        )

                    print("\n\n*****Copy slice data to the whole data arrays...*****")
                    with timer("Copy slice data to the whole data arrays"):
                        _store_slice(
                            output,
                            slice_data,
                            profile_min,
                            profile_max,
                            granule.prof_min,
                        )

        result = assemble_results(output, development, altitude, granule)

    print("\n\n############################################################\n*****Save data in netCDF file...*****")
    with timer("Save data in netCDF file"):
        return write_product(request, result)


def run_granule_pipeline(cfg):
    """Resolve, process, and write one configured CALIOP granule."""

    start_time = datetime.now().astimezone()
    start_tic = time.perf_counter()
    print(f"Start time: {start_time}\n")

    request = resolve_processing_request(cfg)
    output_path = _execute_pipeline(request)

    end_time = datetime.now().astimezone()
    total_time = time.perf_counter() - start_tic
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"End time: {end_time}")
    print(f"Total runtime: {int(hours)} h {int(minutes)} min {seconds:.1f} s")
    return output_path
