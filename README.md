# 2D-McDA

Two-dimensional and multi-channel detection algorithm for the CALIPSO/CALIOP lidar measurements.

2D-McDA processes CALIOP Level 1 attenuated-backscatter profiles and produces two-dimensional feature-detection masks from the 532 nm parallel, 532 nm perpendicular, and 1064 nm channels.

## Source-code organization

The package is divided by responsibility:

```text
twod_mcda/
├── algorithm/   # scientific detection, filtering and mask combination
├── caliop/      # HDF4 access, CALIOP conventions and grid transformations
├── workflow/    # granule, neighbor and slice orchestration
├── output/      # netCDF schema and serialization
├── utils/       # cross-cutting utilities
└── pipeline.py  # public processing entry point
```

The scientific algorithm operates on NumPy arrays and masked arrays. CALIOP
file access is confined to `caliop`, while `output` is responsible only for
turning processing results into the final netCDF product.

## 1. Input and output data

### 1.1 Input

The pipeline expects CALIOP Level 1 HDF files named according to the official CALIOP convention, for example:

```text
CAL_LID_L1-Standard-V5-00.2010-01-18T00-19-57ZN.hdf
```

Files must be stored below the directory configured by `cal_lid_l1.root_directory`, using the structure defined by `cal_lid_l1.path_format` (see [Common settings](#31-common-settings-configcommonyaml)). With the example configuration, the file above is expected at:

```text
<CALIOP_ROOT>/CAL_LID_L1.v5.00/2010/2010_01_18/
```

For each requested granule, the pipeline also looks for the preceding and following granules. They provide the neighboring profiles required to process the edges of the current granule continuously. The search includes the previous and next calendar days to handle granules close to midnight.

### 1.2 Output

The current pipeline writes one netCDF-4 file (using HDF5 storage) containing:

- profile identifiers, observation times, latitude, longitude, and altitude;
- detection flags for the 532 nm parallel channel;
- detection flags for the 532 nm perpendicular channel;
- detection flags for the 1064 nm channel;
- a composite detection mask combining the three channels.

When `processing.save_development_data` is enabled, the file also contains intermediate detection masks, attenuated scattering ratios, and cumulative two-way transmittances.

Outputs are written below `output.root_directory` using the structure defined by `output.path_format`. With the example configuration, this gives:

```text
<OUTPUT_ROOT>/2D_McDA.<version>/<year>/<year>_<month>_<day>/
```

An output filename looks like:

```text
CAL_LID_L2_2D_McDA-Dev-V1-1-0.2010-01-18T00-19-57ZN_lon_75.45_73.80.nc
```

Output metadata follow CF 1.13. The CALIOP orbit is represented as a
`trajectoryProfile` feature, with explicit time, latitude, longitude, and
altitude coordinates. Arrays are compressed losslessly in the netCDF-4 file.


## 2. Installation

### 2.1 Install Conda

This project uses Conda to provide Python and the compiled scientific dependencies required by the processing pipeline. If Conda is not already available on your system, install [Miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/) for your operating system, then open a new terminal.

Verify the installation with:

```bash
conda --version
```

### 2.2 Create the Python environment

Clone the repository and enter its directory:

```bash
git clone https://github.com/tvaillantdeguelis/2D-McDA.git
cd 2D-McDA
```

Create the `twod-mcda` environment from `environment.yml`:

```bash
conda env create --file environment.yml
```

Activate it and install 2D-McDA in editable mode:

```bash
conda activate twod-mcda
python -m pip install -e .
```

## 3. Configuration

The repository stores example configuration files under `config/*.example.yaml`.
Copy them locally before running the pipeline:

```bash
cp config/common.example.yaml config/common.yaml
cp config/period.example.yaml config/period.yaml
cp config/single_granule.example.yaml config/single_granule.yaml
```

### 3.1 Common settings: `config/common.yaml`

This file defines the CALIOP data location, processing options, and output directory shared by single-granule and period runs:

```yaml
cal_lid_l1:
  root_directory: "/path/to/CALIOP"
  version: "5.00"
  path_format: "CAL_LID_L1.v{version}/{year}/{year}_{month:02d}_{day:02d}"

processing:
  save_development_data: false
  max_altitude_km: 30

output:
  root_directory: "/path/to/2D-McDA-output"
  product_type: "Dev"
  path_format: "2D_McDA.v{version}/{year}/{year}_{month:02d}_{day:02d}"
```

The fields have the following meanings:

- `cal_lid_l1.root_directory`: root directory containing the CALIOP Level 1 archive.
- `cal_lid_l1.version`: input product version without the leading `V`.
- `cal_lid_l1.path_format`: path below the root directory. The placeholders  `version`, `year`, `month`, and `day` are filled for each granule.

The CALIOP Level 1 product type is fixed to `Standard` and is not configurable.

- `processing.save_development_data`: include intermediate algorithm arrays in the output netCDF file. This increases its size.
- `processing.max_altitude_km`: maximum processed altitude. The supported values are 30 and 40 km.
- `output.root_directory`: root directory in which result directories are created.
- `output.product_type`: label included in the output filename; it defaults to `Dev`.
- `output.path_format`: path below the output root directory. The placeholders `version`, `year`, `month`, and `day` are filled for each granule.

### 3.2 Single-granule settings: `config/single_granule.yaml`

```yaml
include: common.yaml

granule: "2010-01-18T00-19-57ZN"

subset:
  mode: "profindex"
  start: 4000
  end: 5000
```

- `include` loads the common settings from a file in the same directory.
- `granule` is the timestamp identifying the CALIOP file.
- `subset.mode` can be `profindex` or `longitude`.
- For `profindex`, `start` and `end` are zero-based profile indexes and both bounds are included. Use `start: 0` and `end: null` for the complete granule.
- For `longitude`, `start` and `end` are longitude bounds in degrees. Their order must follow the direction of the satellite track.

### 3.3 Period settings: `config/period.yaml`

```yaml
include: common.yaml

period:
  start_date: "2010-01-01"
  end_date: "2010-01-02"

slurm:
  max_parallel_jobs: 75
```

- `period.start_date` and `period.end_date` delimit the days to search, using the `YYYY-MM-DD` format.
- `slurm.max_parallel_jobs` limits the number of this user's pending and running Slurm jobs. Submission waits when the limit is reached.

The period launcher creates one complete run configuration per discovered granule under `runs/<year>/<year-month>/`, then submits one Slurm job for each granule.

## 4. Running the pipeline

### 4.1 Single granule processing

Local execution:

```bash
python scripts/process_granule.py config/single_granule.yaml
```

Slurm execution:

```bash
python scripts/submit_granule.py config/single_granule.yaml
```

### 4.2 Period processing

Period processing is currently submitted through Slurm:

```bash
python scripts/submit_period.py config/period.yaml
```

## 5. Author

**Thibault Vaillant de Guélis**

Research Scientist

Email: thibault.vaillantdeguelis@outlook.com
