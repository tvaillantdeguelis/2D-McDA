"""Console summaries printed while running the granule pipeline."""


def print_processing_summary(
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
