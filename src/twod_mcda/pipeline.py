"""Top-level processing pipeline.

``run_granule_pipeline`` is the entry point. It locates and opens one CALIOP
granule, prepares it for processing (see ``workflow.preparation``), then
applies the 2D-McDA scientific algorithm slice by slice:

    1. Detect the surface in each of the three lidar channels.
    2. Detect atmospheric features in each channel.
    3. Remove the profiles borrowed from the neighboring granules.
    4. Merge the three channels into one composite detection mask.

The result is written to a netCDF product at the end.
"""

from datetime import datetime
import time

from .algorithm.composite import merged_feature_masks
from .algorithm.features import detect_features_in_3_channels
from .algorithm.surface import detect_surface_in_3_channels
from .caliop.input import open_granule
from .output.product import write_product
from .utils.timing import timer
from .workflow.output_assembly import assemble_results, store_development, store_slice
from .workflow.preparation import prepare_granule
from .workflow.request import resolve_processing_request
from .workflow.slice_loading import describe_slice, load_slice
from .workflow.slicing import trim_slice_context


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
        preparation = prepare_granule(processing_request, current_granule)

        planned_slices = zip(
            preparation.profile_starts,
            preparation.profile_ends,
            preparation.context_starts,
            preparation.context_ends,
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
                preparation.last_profile_in_file,
            )
            description = describe_slice(
                slice_index,
                preparation.slice_count,
                profile_min,
                profile_max,
                context_min,
                context_max,
            )

            # ``timer`` only measures and prints the duration of this block.
            with timer(description):
                with timer("Load slice data"):
                    slice_data = load_slice(
                        first_profile_to_load,
                        last_profile_to_load,
                        current_granule,
                        preparation.previous_profiles,
                        preparation.next_profiles,
                    )

                # ---------------------------------------------------------
                # 2D-McDA scientific algorithm, applied to this slice only.
                # ---------------------------------------------------------
                with timer("Detect the surface in the three lidar channels"):
                    surfaces = detect_surface_in_3_channels(slice_data.input)

                with timer("Detect features in the three lidar channels"):
                    slice_data.masks, slice_data.development = (
                        detect_features_in_3_channels(slice_data.input, surfaces)
                    )

                with timer("Remove neighboring granule context profiles"):
                    trim_slice_context(slice_data)

                with timer("Merge the three channels into a composite mask"):
                    slice_data.masks["Composite_Detection_Flags"] = (
                        merged_feature_masks(
                            slice_data.masks["Parallel_Detection_Flags_532"],
                            slice_data.masks["Perpendicular_Detection_Flags_532"],
                            slice_data.masks["Detection_Flags_1064"],
                        )
                    )
                # ---------------------------------------------------------

                with timer("Copy slice results to whole-granule datasets"):
                    if processing_request.save_development_data:
                        store_development(
                            preparation.granule_development_data,
                            slice_data.development,
                            profile_min,
                            profile_max,
                            first_profile_to_load,
                            current_granule.prof_min,
                            preparation.profile_count,
                        )
                    store_slice(
                        preparation.granule_detection_product,
                        slice_data,
                        profile_min,
                        profile_max,
                        first_profile_to_load,
                        current_granule.prof_min,
                    )

        with timer("Assemble arrays and metadata for the NetCDF product"):
            product_to_write = assemble_results(
                preparation.granule_detection_product,
                preparation.granule_development_data,
                preparation.altitude,
                current_granule,
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
