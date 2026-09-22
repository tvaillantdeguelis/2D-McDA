"""Top-level processing pipeline.

``run_granule_pipeline`` is the entry point. It locates and opens one CALIOP
granule, sets up the three things the algorithm needs (the profile slices, the
context profiles of the adjacent granules, and the empty output datasets), then
applies the 2D-McDA scientific algorithm slice by slice.

The result is written to a netCDF product at the end.
"""

from datetime import datetime
import time

from .algorithm.composite import merged_feature_masks
from .algorithm.features import detect_features_in_3_channels
from .algorithm.surface import detect_surface_in_3_channels
from .config import resolve_processing_request
from .output.assembly import (
    assemble_results,
    empty_outputs,
    store_development,
    store_slice,
)
from .output.product import write_product
from .parameters import NB_PROF_CONTEXT, NB_PROF_SLICE
from .reading.access import open_granule
from .slicing import (
    load_adjacent_context,
    load_slice,
    plan_slices,
    trim_slice_context,
)
from .utils.reporting import print_processing_summary
from .utils.timing import timer


def run_granule_pipeline(cfg):
    """Run the complete scientific pipeline for one CALIOP granule."""

    start_time = datetime.now().astimezone()
    start_tic = time.perf_counter()
    print(f"\nStart time: {start_time}")

    with timer("Resolve processing configuration and locate CALIOP files"):
        processing_request = resolve_processing_request(cfg)

    # Opens the granule's HDF file and resolves its subset bounds, but does not
    # load any scientific array yet (that happens per slice, in load_slice()).
    with timer("Open current CALIOP granule"):
        current_granule_reader = open_granule(processing_request)

    # This ``with`` guarantees that the HDF file closes, even after an error.
    with current_granule_reader as current_granule_reader:
        # -----------------------------------------------------------------
        # One-time setup for this granule, before the slice loop.
        # -----------------------------------------------------------------
        with timer("Plan profile slices and their overlapping context"):
            slices = plan_slices(
                current_granule_reader,
                NB_PROF_SLICE,
                NB_PROF_CONTEXT,
            )

        with timer("Load neighboring granule context profiles"):
            adjacent_context = load_adjacent_context(processing_request, slices)

        print_processing_summary(
            processing_request,
            current_granule_reader,
            slices,
            adjacent_context,
        )

        with timer("Initialize the output datasets"):
            outputs = empty_outputs(current_granule_reader)
        # -----------------------------------------------------------------

        for index, bounds in enumerate(slices, start=1):
            # ``timer`` only measures and prints the duration of this block.
            with timer(
                f"Process slice {index:d}/{len(slices):d} "
                f"(profiles {bounds.profile_min:d} to {bounds.profile_max:d} "
                f"using slice {bounds.context_min:d} to {bounds.context_max:d})"
            ):
                # Reads this slice's profiles from the current granule, plus
                # (at file edges only) context profiles from the neighboring
                # granule, so the algorithm below never sees an artificial edge.
                with timer("Load slice data"):
                    slice_data = load_slice(
                        bounds,
                        current_granule_reader,
                        adjacent_context,
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

                with timer("Copy slice results to the output datasets"):
                    if processing_request.save_development_data:
                        store_development(
                            outputs.development,
                            slice_data.development,
                            bounds,
                            current_granule_reader,
                        )
                    store_slice(
                        outputs.detection,
                        slice_data,
                        bounds,
                        current_granule_reader,
                    )

        with timer("Assemble arrays and metadata for the NetCDF product"):
            product_to_write = assemble_results(outputs, current_granule_reader)

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
