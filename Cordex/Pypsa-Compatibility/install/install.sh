#####################
## DEPRECATION WARNING
## This file is deprecated and will be removed in a future release.
#####################

#!/bin/bash

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

## install environment
conda install mamba=1.5.9 -n base -c conda-forge
mamba env create -f environment.yaml -n cd2es 
## otherwise dask can make problems
conda activate cd2es
pip install dask

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ## on Linux, cdo can be installed via conda
    conda config --add channels conda-forge
    conda install cdo
else
    ## Windows: wget must be downloaded and copied to C:/Windows/System32
    python retrieveWget.py

    ## install cdo in the virtual linux
    wsl sudo apt update
    wsl sudo apt-get install cdo
fi