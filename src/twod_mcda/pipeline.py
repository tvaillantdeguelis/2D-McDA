"""Top-level processing pipeline."""

from datetime import datetime
from pathlib import Path
import re
import time

import numpy as np
import xarray as xr

from .algorithm.composite import merged_feature_masks
from .algorithm.features import detect_features_in_channel
from .algorithm.surface import detect_surface_in_channel
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
from .workflow.settings import NB_PROF_CONTEXT, NB_PROF_SLICE
from .workflow.slicing import plan_slices, trim_profiles
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
        cfg["granule"][:-2], # Remove the trailing 'ZD' or 'ZN' from the granule identifier
        "%Y-%m-%dT%H-%M-%S",
    )
    current_file = find_granule_file(cfg, granule_time)
    previous_file, next_file = find_neighbor_granules(cfg, current_file)

    processing_cfg = cfg.get("processing", {})
    output_cfg = cfg["output"]
    subset_cfg = cfg.get("subset")
    subset_active = (
        subset_cfg is not None
        and subset_cfg.get("activate", True)
    )
    caliop_cfg = cfg["cal_lid_l1"]
    version = _normalized_version(get_full_version())
    granule_date = _granule_id(current_file)
    granule_time = datetime.strptime(
        granule_date[:-2], # Remove the trailing 'ZD' or 'ZN' from the granule identifier
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
        subset_active=subset_active,
        subset_mode=(
            subset_cfg.get("mode", "profindex")
            if subset_active else "profindex"
        ),
        subset_start=(subset_cfg.get("start") if subset_active else None),
        subset_end=(subset_cfg.get("end") if subset_active else None),
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


def _print_request(
    request,
    current_granule,
    previous_granule_path,
    next_granule_path,
    profile_count,
    slice_count,
    previous_context_count,
    next_context_count,
):
    """Print only the input and processing settings useful to the user."""

    if request.subset_active and request.subset_mode == "profindex":
        subset_start = current_granule.prof_min
        subset_end = current_granule.prof_max
        subset_limits_label = "Profile limits"
    elif request.subset_active:
        subset_start = request.subset_start
        subset_end = request.subset_end
        subset_limits_label = "Longitude limits"

    print("\n################# Configuration #################")
    print(f"2D-McDA version        : {request.output_version}")
    print(f"CALIOP L1 version      : {request.caliop_version}")
    print(f"Save development data  : {request.save_development_data}")
    print(f"Maximum altitude       : {request.maximum_altitude_km} km")
    if request.subset_active:
        print(f"Subset mode            : {request.subset_mode}")
        print(f"{subset_limits_label:<23}: {subset_start} -> {subset_end}")
    else:
        print("Subset mode            : false")
    print("#################################################")

    print(f"\n=> Current L1 file to process :\n{current_granule.filepath}")

    if previous_context_count:
        if previous_granule_path is None:
            print(
                "\n=> Previous L1 file: Not found. The algorithm will run "
                "without start context and this may introduce artifacts in "
                f"the first {previous_context_count} profiles."
            )
        else:
            print(
                "\n=> Previous L1 file (used to provide context at the "
                f"start):\n{previous_granule_path}"
            )

    if next_context_count:
        if next_granule_path is None:
            print(
                "\n=> Next L1 file: Not found. The algorithm will run "
                "without end context and this may introduce artifacts in "
                f"the last {next_context_count} profiles."
            )
        else:
            print(
                "\n=> Next L1 file (used to provide context at the end):"
                f"\n{next_granule_path}"
            )

    print(
        f"\nNumber of profiles to process: {profile_count} in "
        f"{slice_count} slices\n"
    )


def _read_adjacent_profiles(
    request,
    granule_date,
    directory,
    profile_start,
    profile_end,
):
    """Load context profiles from one adjacent granule, then close its file."""

    with open_granule(
        request,
        granule_date,
        directory,
        profile_start,
        profile_end,
    ) as adjacent_granule:
        adjacent_profiles = read_slice(
            adjacent_granule,
            adjacent_granule.prof_min,
            adjacent_granule.prof_max,
        )
        adjacent_granule_path = adjacent_granule.filepath

    return adjacent_profiles, adjacent_granule_path


def _slice_description(
    index,
    slice_count,
    profile_min,
    profile_max,
    context_min,
    context_max,
):
    """Describe both the retained profiles and the full algorithm input."""

    return (
        f"Process slice {index:d}/{slice_count:d} "
        f"(profiles {profile_min:d} to {profile_max:d} using slice "
        f"{context_min:d} to {context_max:d})"
    )


def _empty_output(profile_count, altitude, profile_start=0):
    """Allocate an xarray product dataset with named coordinates."""

    if np.isscalar(altitude):
        altitude_values = np.arange(int(altitude))
    else:
        altitude_values = np.asarray(altitude)
    coords = {
        "profile": np.arange(profile_start, profile_start + profile_count),
        "altitude": altitude_values,
    }
    profile_dims = ("profile",)
    grid_dims = ("profile", "altitude")
    return xr.Dataset(
        {
            "Profile_ID": xr.DataArray(
                np.full(profile_count, int(FILL_VALUE_FLOAT), dtype=np.int32),
                dims=profile_dims,
            ),
            "Profile_Time": xr.DataArray(
                np.full(profile_count, FILL_VALUE_FLOAT, dtype=np.float64),
                dims=profile_dims,
            ),
            "Profile_UTC_Time": xr.DataArray(
                np.full(profile_count, FILL_VALUE_FLOAT, dtype=np.float64),
                dims=profile_dims,
            ),
            "Latitude": xr.DataArray(
                np.full(profile_count, FILL_VALUE_FLOAT, dtype=np.float32),
                dims=profile_dims,
            ),
            "Longitude": xr.DataArray(
                np.full(profile_count, FILL_VALUE_FLOAT, dtype=np.float32),
                dims=profile_dims,
            ),
            **{
                name: xr.DataArray(
                    np.zeros((profile_count, altitude_values.size), dtype=np.uint8),
                    dims=grid_dims,
                )
                for name in DETECTION_MASKS
            },
        },
        coords=coords,
    )


def _read_processing_slice(
    profile_min,
    profile_max,
    granule,
    previous,
    following,
):
    """Read one current-granule slice and add context at file edges."""

    data = read_slice(granule, profile_min, profile_max)
    slice_data = SliceData(input=data)
    granule_last_profile = granule.data_reader.nb_profiles - 1

    if profile_min == 0 and previous is not None:
        first_time = data["Profile_Time"].isel(profile=0).item()
        previous_time = previous["Profile_Time"].isel(profile=-1).item()
        time_gap = np.abs(first_time - previous_time)
        print(
            "\tTime between last profile of previous file and first profile "
            f"of current file = {time_gap:.2f} s"
        )
        if profiles_are_consecutive(previous_time, first_time):
            print("\tAppend previous granule")
            slice_data.input = append_adjacent_profiles(data, previous, "start")
            slice_data.previous_context_count = previous.sizes["profile"]
        else:
            print(
                "\tPrevious granule does not seem consecutive. "
                "No start context added."
            )

    if profile_max == granule_last_profile and following is not None:
        following_time = following["Profile_Time"].isel(profile=0).item()
        last_time = data["Profile_Time"].isel(profile=-1).item()
        time_gap = np.abs(following_time - last_time)
        print(
            "\tTime between last profile of current file and first profile "
            f"of next file = {time_gap:.2f} s"
        )
        if profiles_are_consecutive(last_time, following_time):
            print("\tAppend next granule")
            slice_data.input = append_adjacent_profiles(
                slice_data.input,
                following,
                "end",
            )
            slice_data.next_context_count = following.sizes["profile"]
        else:
            print("\tNext granule does not seem consecutive. No end context added.")

    return slice_data


def _store_slice(
    output,
    slice_data,
    profile_min,
    profile_max,
    input_profile_min,
    file_min,
):
    """Store only the result interval, excluding its processing context."""

    file_max = file_min + output.sizes["profile"] - 1
    expected_profiles = np.arange(file_min, file_max + 1)
    if not np.array_equal(output.coords["profile"], expected_profiles):
        output.coords["profile"] = expected_profiles
    copy_min = max(profile_min, file_min)
    copy_max = min(profile_max, file_max)
    if copy_min > copy_max:
        return

    selected_profiles = slice(copy_min, copy_max)

    for name in PROFILE_METADATA:
        output[name].loc[{"profile": selected_profiles}] = slice_data.input[
            name
        ].sel(profile=selected_profiles)

    for name in DETECTION_MASKS:
        output[name].loc[{"profile": selected_profiles}] = slice_data.masks[
            name
        ].sel(profile=selected_profiles)


def _store_development(
    output,
    slice_development,
    profile_min,
    profile_max,
    input_profile_min,
    file_min,
    profile_count,
):
    """Store development data without the processing context."""

    file_max = file_min + profile_count - 1
    copy_min = max(profile_min, file_min)
    copy_max = min(profile_max, file_max)
    if copy_min > copy_max:
        return

    if "profile" not in output.coords:
        output.coords["profile"] = np.arange(
            file_min,
            file_min + profile_count,
        )

    for name, values in slice_development.items():
        if "profile" not in values.dims:
            raise ValueError(
                f"Unsupported development array shape for {name!r}: "
                f"{values.shape}."
            )

        selected = values.sel(profile=slice(copy_min, copy_max))
        if name not in output:
            fill_value = FILL_VALUE_FLOAT if values.dtype.kind == "f" else 0
            output[name] = values.reindex(
                profile=output.coords["profile"],
                fill_value=fill_value,
            )

        output[name].loc[{"profile": selected.coords["profile"]}] = selected


def assemble_results(output, development, altitude, granule):
    """Build the complete product payload from assembled slice outputs."""

    return ProcessingResult(
        data=output,
        development=development,
        altitude=altitude,
        longitude_min=granule.lon_min,
        longitude_max=granule.lon_max,
    )


def run_granule_pipeline(cfg):
    """Run the complete scientific pipeline for one CALIOP granule."""

    start_time = datetime.now().astimezone()
    start_tic = time.perf_counter()
    print(f"\nStart time: {start_time}")

    with timer("Locate current and neighboring CALIOP files"):
        processing_request = resolve_processing_request(cfg)

    with timer("Open current CALIOP granule"):
        current_granule_reader = open_granule(
            processing_request,
            processing_request.granule_date,
            processing_request.current_directory,
            processing_request.subset_start,
            processing_request.subset_end,
            processing_request.subset_mode,
        )

    # This ``with`` guarantees that the HDF file closes, even after an error.
    with current_granule_reader as current_granule:
        with timer("Plan profile slices and their overlapping context"):
            (
                profile_starts,
                profile_ends,
                context_starts,
                context_ends,
            ) = plan_slices(
                current_granule.prof_min,
                current_granule.prof_max,
                NB_PROF_SLICE,
                NB_PROF_CONTEXT,
            )
            profile_count = (
                current_granule.prof_max - current_granule.prof_min + 1
            )
            slice_count = profile_starts.size
            last_profile_in_file = current_granule.data_reader.nb_profiles - 1
            previous_context_count = max(0, -int(context_starts[0]))
            next_context_count = max(
                0,
                int(context_ends[-1]) - last_profile_in_file,
            )

        with timer("Load neighboring granule context profiles"):
            previous_profiles = None
            previous_granule_path = None
            if (
                previous_context_count
                and processing_request.previous_granule is not None
            ):
                (
                    previous_profiles,
                    previous_granule_path,
                ) = _read_adjacent_profiles(
                    processing_request,
                    processing_request.previous_granule,
                    processing_request.previous_directory,
                    -previous_context_count,
                    None,
                )

            next_profiles = None
            next_granule_path = None
            if (
                next_context_count
                and processing_request.next_granule is not None
            ):
                next_profiles, next_granule_path = _read_adjacent_profiles(
                    processing_request,
                    processing_request.next_granule,
                    processing_request.next_directory,
                    None,
                    next_context_count - 1,
                )

        _print_request(
            processing_request,
            current_granule,
            previous_granule_path,
            next_granule_path,
            profile_count,
            slice_count,
            previous_context_count,
            next_context_count,
        )

        with timer("Initialize whole-granule output datasets"):
            altitude = current_granule.get_data("Lidar_Data_Altitudes")
            granule_detection_product = _empty_output(
                profile_count,
                altitude.values,
                current_granule.prof_min,
            )
            granule_development_data = xr.Dataset(
                coords=granule_detection_product.coords
            )

        planned_slices = zip(
            profile_starts,
            profile_ends,
            context_starts,
            context_ends,
        )
        for slice_index, (
            profile_min,
            profile_max,
            context_min,
            context_max,
        ) in enumerate(planned_slices, start=1):
            first_profile_to_load = max(int(context_min), 0)
            last_profile_to_load = min(
                int(context_max),
                last_profile_in_file,
            )
            description = _slice_description(
                slice_index,
                slice_count,
                profile_min,
                profile_max,
                context_min,
                context_max,
            )

            # ``timer`` only measures and prints the duration of this block.
            with timer(description):
                with timer("Load slice data"):
                    slice_data = _read_processing_slice(
                        first_profile_to_load,
                        last_profile_to_load,
                        current_granule,
                        previous_profiles,
                        next_profiles,
                    )
                lidar_data = slice_data.input

                with timer("Surface detection at 532_par"):
                    parallel_532_surface = detect_surface_in_channel(
                        lidar_data,
                        "532_par",
                    )
                with timer("Surface detection at 532_per"):
                    perpendicular_532_surface = detect_surface_in_channel(
                        lidar_data,
                        "532_per",
                    )
                with timer("Surface detection at 1064"):
                    infrared_1064_surface = detect_surface_in_channel(
                        lidar_data,
                        "1064",
                    )

                with timer("Feature detection at 532_par"):
                    (
                        parallel_532_detections,
                        parallel_532_development,
                    ) = detect_features_in_channel(
                        lidar_data,
                        parallel_532_surface,
                        "532_par",
                    )

                with timer("Feature detection at 532_per"):
                    (
                        perpendicular_532_detections,
                        perpendicular_532_development,
                    ) = detect_features_in_channel(
                        lidar_data,
                        perpendicular_532_surface,
                        "532_per",
                    )

                with timer("Feature detection at 1064"):
                    (
                        infrared_1064_detections,
                        infrared_1064_development,
                    ) = detect_features_in_channel(
                        lidar_data,
                        infrared_1064_surface,
                        "1064",
                    )

                with timer("Combine the three channel datasets"):
                    slice_data.masks = xr.merge(
                        [
                            parallel_532_detections,
                            perpendicular_532_detections,
                            infrared_1064_detections,
                        ]
                    )
                    slice_data.development = xr.merge(
                        [
                            parallel_532_development,
                            perpendicular_532_development,
                            infrared_1064_development,
                        ]
                    )

                with timer("Remove neighboring granule context profiles"):
                    if slice_data.previous_context_count:
                        profiles_to_remove = slice_data.previous_context_count
                        slice_data.input = trim_profiles(
                            slice_data.input,
                            profiles_to_remove,
                            "start",
                        )
                        slice_data.masks = trim_profiles(
                            slice_data.masks,
                            profiles_to_remove,
                            "start",
                        )
                        slice_data.development = trim_profiles(
                            slice_data.development,
                            profiles_to_remove,
                            "start",
                        )

                    if slice_data.next_context_count:
                        profiles_to_remove = slice_data.next_context_count
                        slice_data.input = trim_profiles(
                            slice_data.input,
                            profiles_to_remove,
                            "end",
                        )
                        slice_data.masks = trim_profiles(
                            slice_data.masks,
                            profiles_to_remove,
                            "end",
                        )
                        slice_data.development = trim_profiles(
                            slice_data.development,
                            profiles_to_remove,
                            "end",
                        )

                with timer("Merged 3 channels feature detection"):
                    slice_data.masks["Composite_Detection_Flags"] = (
                        merged_feature_masks(
                            slice_data.masks[
                                "Parallel_Detection_Flags_532"
                            ],
                            slice_data.masks[
                                "Perpendicular_Detection_Flags_532"
                            ],
                            slice_data.masks["Detection_Flags_1064"],
                        )
                    )

                with timer("Copy slice results to whole-granule datasets"):
                    if processing_request.save_development_data:
                        _store_development(
                            granule_development_data,
                            slice_data.development,
                            profile_min,
                            profile_max,
                            first_profile_to_load,
                            current_granule.prof_min,
                            profile_count,
                        )
                    _store_slice(
                        granule_detection_product,
                        slice_data,
                        profile_min,
                        profile_max,
                        first_profile_to_load,
                        current_granule.prof_min,
                    )

        with timer("Assemble arrays and metadata for the NetCDF product"):
            product_to_write = ProcessingResult(
                data=granule_detection_product,
                development=granule_development_data,
                altitude=altitude,
                longitude_min=current_granule.lon_min,
                longitude_max=current_granule.lon_max,
            )

    print(
        "\n\n############################################################"
        "\n*****Save data in netCDF file...*****"
    )
    with timer("Save data in netCDF file"):
        output_path = write_product(processing_request, product_to_write)

    end_time = datetime.now().astimezone()
    total_time = time.perf_counter() - start_tic
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"End time: {end_time}")
    print(f"Total runtime: {int(hours)} h {int(minutes)} min {seconds:.1f} s")
    return output_path
