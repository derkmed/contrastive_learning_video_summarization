#!/usr/bin/env bash
FEATURE_EXTRACTOR_BASE=./BMT/submodules/video_features
VIDEO_BASE=/home/hojaeyoon/summe/downsampled
OUTPUT=$(pwd)/output_summe
mkdir -p $OUTPUT
arr=(${VIDEO_BASE}/*.mp4)
VIDEO_LIST=""
for i in ${!arr[*]}; do
    VIDEO_LIST+=" ${arr[i]}"
done
echo $VIDEO_LIST

cd $FEATURE_EXTRACTOR_BASE

# extract i3d features
conda deactivate
conda activate i3d
python main.py \
    --feature_type i3d \
    --on_extraction save_numpy \
    --device_ids 0 \
    --extraction_fps 25 \
    --video_paths ${VIDEO_LIST} \
    --output_path ${OUTPUT}

# extract vgg features
echo $VIDEO_LIST
conda deactivate
conda activate vggish
python main.py \
    --feature_type vggish \
    --on_extraction save_numpy \
    --device_ids 0 \
    --video_paths ${VIDEO_LIST} \
    --output_path ${OUTPUT}

conda deactivate
conda activate bmt
cd ../../
for i in ${!arr[*]};
do
   ID=$(basename ${arr[i]} .mp4)
   echo $ID
   python ./sample/single_video_prediction.py \
    --output_file ${OUTPUT}/${ID}_captions.pth \
    --prop_generator_model_path ./sample/best_prop_model.pt \
    --pretrained_cap_model_path ./sample/best_cap_model.pt \
    --vggish_features_path ${OUTPUT}/${ID}_vggish.npy \
    --rgb_features_path ${OUTPUT}/${ID}_rgb.npy \
    --flow_features_path ${OUTPUT}/${ID}_flow.npy \
    --duration_in_secs $(
        ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 ${arr[i]}
    ) \
    --device_id 0 \
    --max_prop_per_vid 100 \
    --nms_tiou_thresh 0.4
done
cd ../

# extract clip embedding
conda deactivate
conda activate ml
python text_clip_embedding.py \
    --caption_dir=${OUTPUT} \
    --output_dir=${OUTPUT} \
    --model RN101