import torch
import torchvision.transforms as T
import torchvision.io as io
import pandas as pd
import numpy as np
import clip
import os

NUM_FRAMES = 10

video_base = './ydata-tvsum50-v1_1'
processed_base = './ydata-tvsum50-v1_1/processed'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load larger models like ViT-L/14 later. This is just for testing.
model, preprocess = clip.load('RN50', device=device)
to_pil = T.ToPILImage()

if not os.path.exists(processed_base):
    os.mkdir(processed_base)

def get_evenly_spaced_ids(size, num_elements):
    return np.round(np.linspace(0, size - 1, num_elements)).astype(int)

video_info = pd.read_csv(os.path.join(
    video_base, 'data', 'ydata-tvsum50-info.tsv'), sep='\t')
video_ids = video_info['video_id'].tolist()
for video_id in video_ids:
    video_path = os.path.join(video_base, 'video', video_ids[0] + '.mp4')
    file_path = os.path.join(processed_base, video_id + f'.{NUM_FRAMES}' + '.pth')
    if os.path.exists(file_path):
        print('File already exists')
    video_path = os.path.join(video_base, 'video', video_id + '.mp4')
    video, _, meta = io.read_video(video_path, pts_unit='sec')
    ids = get_evenly_spaced_ids(video.shape[0], NUM_FRAMES)
    
    target_frames = video[ids]
    clip_feature = []
    to_pil = T.ToPILImage()
    with torch.no_grad():
        for frame in target_frames:
           image = preprocess(
                to_pil(
                    frame.numpy()
                )
            ).unsqueeze(0).to(device)
           clip_feature.append(
               model.encode_image(image)
           )

    data = {
        'meta': meta,
        'video': video[ids],
        'clip_feature': clip_feature
    }
    torch.save(data, file_path)



