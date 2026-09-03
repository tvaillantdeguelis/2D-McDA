"""Prepare a CALIOP granule for processing.

This gathers everything that must happen before the algorithm can run on
the first slice: planning the slices, loading context profiles from the
neighboring granules, printing the run summary, and allocating the empty
whole-granule output datasets.
"""

import xarray as xr

from twod_mcda.utils.timing import timer
from twod_mcda.workflow.models import GranulePreparation
from twod_mcda.workflow.neighbors import read_adjacent_profiles
from twod_mcda.workflow.output_assembly import empty_output
from twod_mcda.workflow.reporting import print_processing_summary
from twod_mcda.workflow.settings import NB_PROF_CONTEXT, NB_PROF_SLICE
from twod_mcda.workflow.slicing import plan_slices


def _load_context_profiles(request, previous_context_count, next_context_count):
    """Load context profiles from the previous and next granules, if any."""

    previous_profiles = None
    previous_granule_path = None
    if previous_context_count and request.previous_granule is not None:
        previous_profiles, previous_granule_path = read_adjacent_profiles(
            request,
            request.previous_granule,
            request.previous_directory,
            -previous_context_count,
            None,
        )

    next_profiles = None
    next_granule_path = None
    if next_context_count and request.next_granule is not None:
        next_profiles, next_granule_path = read_adjacent_profiles(
            request,
            request.next_granule,
            request.next_directory,
            None,
            next_context_count - 1,
        )

    return previous_profiles, previous_granule_path, next_profiles, next_granule_path


def prepare_granule(request, granule):
    """Plan the slices, load context profiles, and allocate the output datasets."""

    with timer("Plan profile slices and their overlapping context"):
        profile_starts, profile_ends, context_starts, context_ends = plan_slices(
            granule.prof_min,
            granule.prof_max,
            NB_PROF_SLICE,
            NB_PROF_CONTEXT,
        )
        profile_count = granule.prof_max - granule.prof_min + 1
        slice_count = profile_starts.size
        last_profile_in_file = granule.data_reader.nb_profiles - 1
        previous_context_count = max(0, -int(context_starts[0]))
        next_context_count = max(0, int(context_ends[-1]) - last_profile_in_file)

    with timer("Load neighboring granule context profiles"):
        (
            previous_profiles,
            previous_granule_path,
            next_profiles,
            next_granule_path,
        ) = _load_context_profiles(
            request,
            previous_context_count,
            next_context_count,
        )

    print_processing_summary(
        request,
        granule,
        previous_granule_path,
        next_granule_path,
        profile_count,
        slice_count,
        previous_context_count,
        next_context_count,
    )

    with timer("Initialize whole-granule output datasets"):
        altitude = granule.get_data("Lidar_Data_Altitudes")
        granule_detection_product = empty_output(
            profile_count,
            altitude.values,
            granule.prof_min,
        )
        granule_development_data = xr.Dataset(
            coords=granule_detection_product.coords
        )

    return GranulePreparation(
        profile_starts=profile_starts,
        profile_ends=profile_ends,
        context_starts=context_starts,
        context_ends=context_ends,
        slice_count=slice_count,
        profile_count=profile_count,
        last_profile_in_file=last_profile_in_file,
        previous_profiles=previous_profiles,
        next_profiles=next_profiles,
        altitude=altitude,
        granule_detection_product=granule_detection_product,
        granule_development_data=granule_development_data,
    )
