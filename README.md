# 2D-McDA

Two-dimensional and multi-channel detection algorithm for the CALIPSO/CALIOP lidar measurements.

2D-McDA processes CALIOP Level 1 attenuated-backscatter profiles and produces two-dimensional feature-detection masks from the 532 nm parallel, 532 nm perpendicular, and 1064 nm channels.

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
cp config/case_studies.example.yaml config/case_studies.yaml
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
  activate: true
  mode: "profindex"
  start: 4000
  end: 5000
```

- `include` loads the common settings from a file in the same directory.
- `granule` is the timestamp identifying the CALIOP file.
- `subset.activate`: set to `false` to process the complete granule. The complete granule is also processed when the `subset` block is absent. When the block is present and `activate` is omitted, it defaults to `true` for backward compatibility.
- `subset.mode` can be `profindex` or `longitude`.
- For `profindex`, `start` and `end` are zero-based profile indexes and both bounds are included. Use `end: null` to extend an active subset to the last profile.
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

Complete-granule jobs are named `2D-McDA_<granule>`. Jobs for active subsets are named `2D-McDA_<granule>_<lon/prof>_<start>_<end>`.

### 4.2 Period processing

Period processing is currently submitted through Slurm:

```bash
python scripts/submit_period.py config/period.yaml
```

### 4.3 Case-study processing

Set `enabled: true` for each case study to process, then submit one Slurm job per enabled entry:

```bash
python scripts/submit_case_studies.py config/case_studies.yaml
```

The launcher creates complete run configurations under `runs/case_studies/`. Each case study defines its granule and the `mode`, `start`, and `end` bounds of its active subset.

### 4.4 Interactive visualization

The interactive notebook under `visualization/` displays the 2D-McDA masks together with the corresponding CALIOP Level 1 attenuated-backscatter signals, and the CALIOP VFM. All panels share their profile and altitude ranges, so zooming or panning one panel updates the complete layout.

Create a local viewer configuration before opening the notebook:

```bash
cp visualization/config.example.yaml visualization/config.yaml
```

Edit `visualization/config.yaml` as needed:

```yaml
2d_mcda:
  root_directory: data/output
  version: "2.3.1"
  product_type: "Dev"
  path_format: "2D_McDA.v{version}/{year}/{year}_{month:02d}_{day:02d}"
  granule: "2018-08-31T21-33-53ZN_lon_67.00_60.00"

cal_lid_l1:
  root_directory: "<CALIOP_ROOT>"
  version: "5.00"
  product_type: "Standard"
  path_format: "CAL_LID_L1.v{version}/{year}/{year}_{month:02d}_{day:02d}"

cal_lid_l2_vfm:
  root_directory: "<CALIOP_ROOT>"
  version: "5.00"
  product_type: "Standard"
  path_format: "VFM.v{version}/{year}/{year}_{month:02d}_{day:02d}"

plot:
  width: 500
  height: 280
  longitude_range: [67.00, 60.00]
  altitude_range: [0, 30]
```

- `2d_mcda.root_directory`, `2d_mcda.version`, and `2d_mcda.path_format`: location and directory structure of the 2D-McDA archive. The viewer searches the matching daily directory recursively, falling back to the root directory when that daily directory is absent.
- `2d_mcda.granule`: required suffix identifying the 2D-McDA NetCDF file. It may identify the complete granule, for example `2018-08-31T21-33-53ZN`, or a longitude section, for example `2018-08-31T21-33-53ZN_lon_67.00_60.00`. The configured value must match exactly one NetCDF file.
- `cal_lid_l1` and `cal_lid_l2_vfm`: locations and directory structures of the CALIOP Level 1 and VFM archives. The viewer derives the complete CALIOP identifier, such as `2018-08-31T21-33-53ZN`, from `2d_mcda.granule` by removing an optional `_lon_<start>_<end>` suffix, then selects the corresponding L1 and VFM HDF files automatically.
- `plot.width` and `plot.height`: dimensions in pixels of each interactive panel.
- `plot.longitude_range`: required pair of longitude bounds. The nearest profiles define the initial horizontal limits of all twelve panels and the highlighted section on the map.
- `plot.altitude_range`: initial altitude limits in kilometres, or `null` to display the complete vertical range.

Relative paths are resolved from the repository root. `visualization/config.yaml` is ignored by Git, while `visualization/config.example.yaml` is tracked and provides the portable defaults.

Start JupyterLab and open the viewer:

```bash
conda run -n twod-mcda jupyter lab visualization/2d_mcda_interactive_viewer.ipynb
```

## 5. Author

**Thibault Vaillant de Guélis**

Research Scientist

Email: thibault.vaillantdeguelis@outlook.com
