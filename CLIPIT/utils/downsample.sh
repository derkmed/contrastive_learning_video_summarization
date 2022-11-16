#!/usr/bin/env bash
VIDEO_DIR=/home/hojaeyoon/summe/videos
OUTPUT_DIR=/home/hojaeyoon/summe/downsampled
mkdir $OUTPUT_DIR
arr=(${VIDEO_DIR}/*.mp4)
for i in ${!arr[*]}; do
    FILE_NAME=$(basename ${arr[i]})
    ffmpeg -i ${arr[i]} -filter:v fps=8 ${OUTPUT_DIR}/downsampled_8fps_${FILE_NAME}
done
