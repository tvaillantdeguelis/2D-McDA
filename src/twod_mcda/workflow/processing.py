"""Apply the 2D-McDA algorithm to one in-memory profile slice."""

from twod_mcda.algorithm.composite import merged_feature_masks
from twod_mcda.algorithm.features import detect_features_in_3_channels
from twod_mcda.algorithm.surface import detect_surface_in_3_channels
from twod_mcda.utils.timing import timer
from twod_mcda.workflow.slicing import trim_profiles


def _trim_neighbor_profiles(slice_data):
    context_by_side = (
        ("start", slice_data.previous_context_count),
        ("end", slice_data.next_context_count),
    )

    for side, profile_count in context_by_side:
        if profile_count == 0:
            continue
        print(f"\n\n*****Remove context from {side} adjacent file...*****")
        slice_data.input = trim_profiles(slice_data.input, profile_count, side)
        slice_data.masks = trim_profiles(slice_data.masks, profile_count, side)
        slice_data.development = trim_profiles(
            slice_data.development,
            profile_count,
            side,
        )


def process_slice(slice_data):
    """Apply surface, channel, and composite detection to loaded data."""

    surfaces = detect_surface_in_3_channels(slice_data.input)

    slice_data.masks, slice_data.development = detect_features_in_3_channels(
        slice_data.input,
        surfaces,
    )

    _trim_neighbor_profiles(slice_data)

    with timer("Merged 3 channels feature detection"):
        slice_data.masks["Composite_Detection_Flags"] = merged_feature_masks(
            slice_data.masks["Parallel_Detection_Flags_532"],
            slice_data.masks["Perpendicular_Detection_Flags_532"],
            slice_data.masks["Detection_Flags_1064"],
        )

    return slice_data
