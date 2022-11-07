#!/bin/bash
source $HOME/anaconda3/etc/profile.d/conda.sh
conda activate cv_detection 

python -u $HOME/systemd/startup.py
