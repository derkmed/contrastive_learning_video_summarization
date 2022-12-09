from typing import List
import torchvision.io as io
import numpy as np
import os

def generate_video(video_path: str, segment_ids: List[int], 
                   downsized_fps: int=16, segment_fps: int = 16, output_dir='./output') -> None:
    segment_ids.sort()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    video, _, md = io.read_video(video_path)
    final_video = None
    segment_size = downsized_fps * segment_fps
    for i, sid in enumerate(segment_ids):
        segment = video[sid * segment_size: (sid + 1) * segment_size]
        if i:
            final_video = np.append(final_video, segment, axis=0)
        else:
            final_video = segment
    output_path = f'{output_dir}/{os.path.basename(video_path)}'    
    io.write_video(output_path, final_video, md['video_fps'])

if __name__ == '__main__':
    generate_video('/home/hojaeyoon/summe/videos/Paintball.mp4', [1,2,3,4,5,6], segment_fps=6, downsized_fps=6)