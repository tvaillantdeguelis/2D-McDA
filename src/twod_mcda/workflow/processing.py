"""Apply the 2D-McDA algorithm to one in-memory profile slice."""

from twod_mcda.algorithm.composite import merged_feature_masks
from twod_mcda.utils.timing import timer
from twod_mcda.workflow.channels import (
    detect_channel_features,
    detect_surfaces,
)
from twod_mcda.workflow.settings import NB_PROF_OVERLAP
from twod_mcda.workflow.slicing import trim_profiles


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


def process_slice(slice_data):
    """Apply surface, channel, and composite detection to loaded data."""

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
