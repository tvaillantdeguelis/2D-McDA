def process_granule(config, granule):

    data = read_l1_data(config, granule)

    data = detect_surface(data)

    data = detect_features(data)

    data = merge_feature_masks(data)

    if config.processing.make_classification:
        data = classify_features(data)

    write_outputs(data, config)