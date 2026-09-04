"""Read one profile slice, with context from the neighboring granules."""

import numpy as np

from twod_mcda.workflow.models import SliceData
from twod_mcda.workflow.neighbors import (
    append_adjacent_profiles,
    profiles_are_consecutive,
)
from twod_mcda.caliop.input import read_slice


def describe_slice(
    index, slice_count, profile_min, profile_max, context_min, context_max
):
    """Describe both the retained profiles and the full algorithm input."""

    return (
        f"Process slice {index:d}/{slice_count:d} "
        f"(profiles {profile_min:d} to {profile_max:d} using slice "
        f"{context_min:d} to {context_max:d})"
    )


def load_slice(profile_min, profile_max, granule, previous, following):
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
