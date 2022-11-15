#!/usr/bin/env bash
cd BMT
conda env create -f ./video_features/conda_env_i3d.yml
conda env create -f ./video_features/conda_env_vggish.yml
conda env create -f ./BMT/conda_env.yml
conda env create -f ../conda_env.yml
conda deactivate
conda activate bmt
conda install ffmpeg
python -m spacy download en

# download pretrained model
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/best_prop_model.pt
wget https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/bmt/best_cap_model.pt
wget https://storage.googleapis.com/audioset/vggish_model.ckpt -P ./BMT/submodules/video_features/models/vggish/checkpoints