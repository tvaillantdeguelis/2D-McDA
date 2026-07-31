## Installation

### Install Conda

This project uses Conda to provide Python and the compiled scientific
dependencies required by the processing pipeline. If Conda is not already
available on your system, install
[Miniconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/)
for your operating system, then open a new terminal.

Verify the installation with:

```bash
conda --version
```

### Create the Python environment

Clone the repository and enter its directory:

```bash
git clone https://github.com/tvaillantdeguelis/2D-McDA.git
cd 2D-McDA
```

Create the `twod-mcda` environment from `environment.yml`:

```bash
conda env create --name twod-mcda --file environment.yml
```

Activate it and install 2D-McDA in editable mode:

```bash
conda activate twod-mcda
python -m pip install -e .
```

The editable installation makes the `twod_mcda` package available while using
the source files from the cloned repository. Verify the installation with:

```bash
python -c "import twod_mcda; print(twod_mcda.__version__)"
```

The environment must be activated again in each new terminal before running
the program:

```bash
conda activate twod-mcda
```

### Update an existing environment

After pulling a change to `environment.yml`, update the local environment and
remove dependencies that are no longer declared with:

```bash
conda env update --name twod-mcda --file environment.yml --prune
```

## Configuration

The repository stores example configuration files under `config/*.yaml.example`.
Copy the examples locally before running the pipeline:

```bash
cp config/default.yaml.example config/default.yaml
cp config/period.yaml.example config/period.yaml
cp config/single_granule.yaml.example config/single_granule.yaml
```

## Launch

### Single granule processing

#### Local execution
`python scripts/process_granule.py config/single_granule.yaml`

#### Slurm execution
`python scripts/submit_granule.py config/single_granule.yaml`


### Period processing

#### Slurm execution
`python scripts/submit_period.py config/period.yaml`


## Author

**Thibault Vaillant de Guélis**

Research Scientist

Email: thibault.vaillantdeguelis@outlook.com
