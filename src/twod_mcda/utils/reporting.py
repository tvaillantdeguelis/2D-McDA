"""Console summaries printed while running the granule pipeline."""


def print_processing_summary(
    request,
    current_granule_reader,
    slices,
    adjacent_context,
):
    """Print only the input and processing settings useful to the user."""

    previous_granule_path = adjacent_context.previous_granule_path
    next_granule_path = adjacent_context.next_granule_path
    nb_profiles_previous_context = adjacent_context.nb_profiles_previous_context
    nb_profiles_next_context = adjacent_context.nb_profiles_next_context

    if request.subset_active and request.subset_mode == "profindex":
        subset_start = current_granule_reader.prof_min
        subset_end = current_granule_reader.prof_max
        subset_limits_label = "Profile limits"
    elif request.subset_active and request.subset_mode == "longitude":
        subset_start = request.subset_start
        subset_end = request.subset_end
        subset_limits_label = "Longitude limits"
    elif request.subset_active:
        raise ValueError(
            f"Error: subset_mode = '{request.subset_mode}' is not defined. "
            "Please use 'profindex' or 'longitude'\n"
        )

    print("\n################# Configuration #################")
    print(f"2D-McDA version        : v{request.output_version}")
    print(f"CALIOP L1 version      : v{request.caliop_version}")
    print(f"Save development data  : {request.save_development_data}")
    print(f"Maximum altitude       : {request.maximum_altitude_km} km")
    if request.subset_active:
        print(f"Subset mode            : {request.subset_mode}")
        print(f"{subset_limits_label:<23}: {subset_start} -> {subset_end}")
    else:
        print("Subset mode            : false")
    print("#################################################")

    print(f"\n=> Current L1 file to process :\n{current_granule_reader.filepath}")

    if nb_profiles_previous_context:
        if previous_granule_path is None:
            print(
                "\n=> Previous L1 file: Not found. The algorithm will run "
                "without start context and this may introduce artifacts in "
                f"the first {nb_profiles_previous_context} profiles."
            )
        else:
            print(
                "\n=> Previous L1 file (used to provide context at the "
                f"start):\n{previous_granule_path}"
            )

    if nb_profiles_next_context:
        if next_granule_path is None:
            print(
                "\n=> Next L1 file: Not found. The algorithm will run "
                "without end context and this may introduce artifacts in "
                f"the last {nb_profiles_next_context} profiles."
            )
        else:
            print(
                "\n=> Next L1 file (used to provide context at the end):"
                f"\n{next_granule_path}"
            )

    print(
        f"\nNumber of profiles to process: {current_granule_reader.nb_profiles} "
        f"in {len(slices)} slices\n"
    )
