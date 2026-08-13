# Changelog

## Unreleased

### Changed

- Replaced HDF4 product output with compressed netCDF-4 output and CF 1.13 metadata.
- Simplified the pipeline to invoke the current processing implementation directly.

### Removed

- Removed the output format setting and the obsolete HDF4 writers.
- Removed the legacy and archived processing implementations.

## [1.1.10] - 2026-08-10

### Changed

- Modified README.

## [1.1.9] - 2026-08-10

### Changed

- Filled the CHANGELOG.

## [1.1.8] - 2026-08-10

### Changed

- Reformatted the README.

## [1.1.7] - 2026-08-10

### Changed

- Cleaned up console output.

## [1.1.6] - 2026-07-31

### Added

- Added `config/common.example.yaml`, containing the CALIOP input, processing, and output settings shared by all run configurations.

## [1.1.5] - 2026-07-31

### Changed

- Stopped publishing the `develop` branch as part of the release publication script.

## [1.1.4] - 2026-07-31

### Added

- Added configurable input and output path formats and automatic versioned output directories.
- Expanded the README with input/output descriptions, configuration reference, and execution instructions.

### Changed

- Split configuration into a shared `common.yaml` file and run-specific `period.yaml` and `single_granule.yaml` files.
- Simplified the Conda environment to direct project dependencies.

### Removed

- Removed generated run configuration files from version control.
- Removed the obsolete `processing.make_classification` option.

## [1.1.3] - 2026-07-31

### Changed

- Documented Conda installation, environment creation, activation, editable package installation, and environment updates.
- Refreshed and reduced the exported Conda environment specification.

## [1.1.2] - 2026-07-30

### Changed

- Simplified the release preparation script by removing redundant branch synchronization steps.

## [1.1.1] - 2026-07-30

### Added

- Added scripts to prepare a versioned release and publish it to the main branch with a Git tag.

## [1.1.0] - 2026-07-30

### Added

- Introduced the installable `twod_mcda` Python package using a `src` layout and `pyproject.toml`.
- Added a configuration-driven pipeline for local, single-granule Slurm, and period Slurm processing.
- Added CALIOP granule discovery, neighboring-granule handling, HDF/NetCDF output utilities, and pipeline regression tests.
- Added example YAML configuration files and ignored machine-specific local configurations.

### Changed

- Integrated the required utilities directly into the repository instead of using the `my_modules` Git submodule.
- Refactored the legacy monolithic implementation into modules for configuration, I/O, detection, merging, preprocessing, and pipeline orchestration.
- Moved the previous implementation under `src/legacy` for compatibility and regression testing.

### Removed

- Removed the old shell launchers, environment creation script, and Git submodule.

## [1.0.4] - 2026-07-24

### Added

- Added configurable input-folder and maximum-altitude handling.
- Added a JIT-compiled two-dimensional Gaussian averaging implementation.

### Changed

- Enabled two-way-transmittance correction down to a transmittance limit of `0.1`.
- Updated feature-detection parameters and the Conda environment.
- Improved handling of development-only output data.

## [1.0.3] - 2024-09-10

### Added

- Added optional transmission correction to feature detection.
- Corrected signal averaging below 8.2 km.

### Changed

- Renamed the Conda environment used by the launch scripts.

## [1.0.2] - 2024-08-21

### Added

- Initial tagged release of the two-dimensional, multi-channel CALIOP feature-detection workflow.
- Added batch and single-granule launch scripts and a reproducible Conda environment.

[1.1.10]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.1.9...v1.1.10
[1.1.9]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.1.8...v1.1.9
[1.1.8]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.1.7...v1.1.8
[1.1.7]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.1.6...v1.1.7
[1.1.6]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.1.5...v1.1.6
[1.1.5]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.1.4...v1.1.5
[1.1.4]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.1.3...v1.1.4
[1.1.3]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.0.4...v1.1.0
[1.0.4]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/tvaillantdeguelis/2D-McDA/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/tvaillantdeguelis/2D-McDA/releases/tag/v1.0.2
