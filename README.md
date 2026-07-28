## Installation

Create the Conda environment:

conda env create -n twod-mcda -f environment.yml

Activate it:

conda activate twod-mcda




module load python

python -m venv ~/venvs/twod-mcda

source ~/venvs/twod-mcda/bin/activate

python -m pip install -e .






## Launch

### Single granule processing

#### Local execution
`python scripts/process_granule.py --config config/single_granule.yaml`

#### Slurm execution
`python scripts/submit_granule.py config/single_granule.yaml`


### Period processing

#### Slurm execution
`python scripts/submit_period.py \
    --config config/period.yaml`


## Author

**Thibault Vaillant de Guélis**

Research Scientist

Email: thibault.vaillantdeguelis@outlook.com