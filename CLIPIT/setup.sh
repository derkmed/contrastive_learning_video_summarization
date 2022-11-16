#!/usr/bin/env bash

# Environment setup
conda env create -f ./conda_env.yml
conda activate clipit

# Download dataset
# VSUM
wget -c https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz -O - | tar -xz
cd ydata-tvsum50-v1_1
unzip ydata-tvsum50-video.zip
unzip ydata-tvsum50-data.zip
