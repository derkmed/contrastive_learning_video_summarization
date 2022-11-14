#!/usr/bin/env bash
cd BMT
conda env create -f ./video_features/conda_env_i3d.yml
conda env create -f ./video_features/conda_env_vggish.yml