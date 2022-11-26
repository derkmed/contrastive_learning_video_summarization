from cv2 import split
import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms as transforms
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import json
import torch
import imageio
import mat73
import math

from torch.utils.data import DataLoader, Dataset
import scipy.io
from typing import List, Tuple


class Summarizer_Dataset(Dataset):

    def __init__(self, 
        dataset_name: str,
        data_list_file: str = "../data/splits/summe_all.txt",
        gt_root: str = "../data/gt", # this should be set to a file if tvsum!
        # min_frame_length: int = 832, 
        num_frames_per_segment: int = 32, 
        size: int = 112,
        is_shuffle: bool = False,
        is_video_only: bool = False,
        debug_mode: bool = False
    ):

        # This is a file with each line being a space-delimited
        # [absolute_path, class_index]
        self.data_list_file = data_list_file

        # This is the root ground truth folder for SumMe OR the ground truth
        # FILE for TVSum.
        self.gt_root = gt_root
        
        # This should be the dimension that size each frame is resized to.
        # The resized image will be a square of (self.size x self.size).
        self.size = size

        # This is the minimum number of frames a video MUST have.
        # self.req_frame_count = min_frame_length
        
        # This is the length of each segment, in frames. Each segment will be
        # passed through TCLR to obtain a contrastive embedding.
        self.num_frames_per_segment = num_frames_per_segment
        
        # self.num_sec = self.req_frame_count / float(self.fps)
        
        self.dataset_name = dataset_name
        self.data = open(self.data_list_file,'r').read().splitlines()
        assert_msg = lambda f : f"{f} is misconfigured. Expecting space delimited [FILE_NAME, IDX]"
        assert len(self.data[0].split(' ')) == 2, assert_msg(data_list_file)

        self.is_shuffle = is_shuffle
        if self.is_shuffle:
            random.shuffle(self.data)

        
        self.videos = {int(row.split(' ')[1]): row.split(' ')[0] for row in self.data}
        self.is_video_only = is_video_only
        
        self.debug_mode = debug_mode

    
    def __len__(self):
        return len(self.videos)

    def _get_resized_video(self, video_path: str, resized_dim: int) -> torch.Tensor:
        '''
        Get all the video frames and resize them into resized_dim x resized_dim boxes
        to pass into TCLR.

        :param video_path: the input video path.
        :param resized_dim: new square dimension
        :return: Tensor of video frames [B, C, H, W]
        '''
        # Get all the frames.
        frames, _, _ = io.read_video(video_path, pts_unit="sec")
        if self.debug_mode:
          print(f"[GetResizedVideo] " +\
                f"{video_path} has {len(frames)} frames.")

        # [B, H, W, C] -> [B, C, H, W]
        # Resize frames to resized_dimension x resized_dimension image. Stack them in
        # the `video` list.
        frames = frames.permute(0, 3, 1, 2)
        video = []
        for frame in frames:
            video.append(transforms.Resize((resized_dim, resized_dim))(frame))
        video = torch.stack(video)
        video = video.view(-1, 3, resized_dim, resized_dim)

        # [B, C, H, W] 
        return video

    def _get_video_segments(self, 
        video_path: str, resized_dim: int, frames_per_segment: int) -> List[torch.Tensor]:
        '''
        Breaks up a video into segments of self.num_frames_per_segment length.

        :param video_path: the input video path
        :param resized_dim: new square dimension
        :param frames_per_segment: number of frames to include in each segment
        :return: List[Tensor] list of segments [B, H, W, C]
        '''
        video = self._get_resized_video(video_path, resized_dim)
        nframes = video.shape[0]
        # Pad with zeros to ensure that each segment is whole.
        is_divisible = (nframes % frames_per_segment == 0)
        next_segment_start_idx = math.ceil(nframes / frames_per_segment) * frames_per_segment
        if not is_divisible and nframes < next_segment_start_idx:
          if self.debug_mode:
            print(f"[GetVideoSegments] " +\
                  f"Input video of {nframes} is not evenly divisible" +\
                  f" into {frames_per_segment} frames...")
            
          zeros = torch.zeros(
              (next_segment_start_idx - nframes, 3, resized_dim, resized_dim),
              dtype=torch.uint8)
          video = torch.cat((video, zeros), axis=0)
        
        # [B, C, H, W] --> [B, H, W, C]
        video = video.permute(0, 2, 3, 1)
        splitted = torch.split(video, frames_per_segment)
        return splitted


    def _get_vidname(self, filepath: str) -> str:
        # remove the rightmost file-extension and return the rightmost filename
        # without the absolute path.
        return (filepath.rsplit('.')[0]).rsplit('/', 1)[-1]

    def _get_summe_labels(self, vid_name: str) -> torch.FloatTensor:
        '''
        All summe ground truths are in individual .mat files with filenames corresponding
        to their video counterpats.
        '''
        gt_file = f"{vid_name}.mat"
        gt_data = scipy.io.loadmat(f"{self.gt_root}/{gt_file}")
        gt_score = gt_data.get('gt_score')
        return np.squeeze(gt_score)

    def _get_tvsum_labels(self, vid_name: str):
        '''
        All tvsum ground truths are in a single .mat file.
        '''
        MAX_SCORE = 5.0 # this is needed to normalize the scores.
        gt = mat73.loadmat(self.gt_root)
        videos = gt['tvsum50']['video']
        gt_idx = videos.index(vid_name)
        gt_scores = gt['tvsum50']['gt_score'][gt_idx]
        return gt_scores / MAX_SCORE 


    def _get_labels_for_downsampled_segments(self, dwns_nframes: int, 
        num_frames_per_segment: int,
        orig_labels: torch.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        The original FPS *may be higher than* the downsampled segments. This means we
        must aggregate the original PER-FRAME labels from the original video such
        that they correspond to the downsampled video.

        This is achieved by first deriving the number of segments in each video based
        on the downsampled video. Then, we refer to the original video PER-FRAME labels
        and aggregate them based on which segment the fall under. The label is set to
        be the max of the binary per-frame labels. The score is calculated as the average
        however, resulting in a score in [0, 1].

        :param downs_nframes: the number of frames in the downampled video
        :param orig_labels: the per-frame label assignment in the original video
        :return: aggregated segment label_scores
        '''
        
        # Find out how many frames would be in each segment if we
        # were to divide the original video into the same number of
        # segments.
        
        orig_nframes = len(orig_labels)
        if orig_nframes == dwns_nframes:
          if self.debug_mode:
            print(f"[GetLabelsForDownsampledSegments] No aggregation needed." +\
                  f" videos both have {orig_nframes} frames.")
          return torch.squeeze(orig_labels)

        nsegments = int(dwns_nframes / num_frames_per_segment)
        orig_frames_per_segment = int(orig_nframes / nsegments)
        label_scores = torch.zeros(nsegments)
        if self.debug_mode:
          print(f"[GetLabelsForDownsampledSegments] " +\
                f"Aggregating labels for {orig_nframes}-frames video" +\
                f" into {nsegments} segments for downsampled video of" +\
                f" {dwns_nframes} frames.")
        for s_i in range(nsegments):
            label_scores[s_i] = orig_labels[
                s_i * orig_frames_per_segment : 
                (s_i + 1) * orig_frames_per_segment
            ].mean()
        return torch.squeeze(label_scores)

    def _getfps(self, video_filepath: str) -> int:
        reader = io.VideoReader(video_filepath, "video")
        fps = reader.get_metadata()["video"]["fps"][0]
        return fps

    def __getitem__(self, idx):
        try:
          # Obtain the next video.        
          video_filepath = self.videos[idx]
          vid_name = self._get_vidname(video_filepath)
          segments = self._get_video_segments(video_filepath, self.size,
                                              self.num_frames_per_segment)
          dwns_nframes = len(segments) * self.num_frames_per_segment
          

          # Obtain the ground truth data. Given that the FPS of the input
          # may not match the original source in the original Summe & TVSum
          # datasets, we will need to aggregate the frame-level importance scores
          # accordingly.
          orig_labels = None
          if self.dataset_name == "summe":
              # print("Video: ", video_filepath)
              orig_labels = self._get_summe_labels(vid_name)
              # num_frames_in_video = len(orig_labels)
          elif self.dataset_name == "tvsum":
              orig_labels = self._get_tvsum_labels(vid_name)
          else:
              raise ValueError('Invalid dataset provided.')

          
          # downsampled_fps = self._getfps(video_filepath)
          label_scores = self._get_labels_for_downsampled_segments(
              dwns_nframes, self.num_frames_per_segment, orig_labels)
          
          assert label_scores.shape[0] == len(segments), ""

          if self.is_video_only:
              return {"segments": segments}
          else:
              return {"segments": segments, "scores": label_scores}
        
        except Exception as e:
          print(e)
          print("Cannot procure item. Error encountered")
          return None

if __name__ == '__main__':
    # Use this for SumMe
    # train_dataset = Summarizer_Dataset(
    #     dataset_name='summe', 
    #     data_list_file = "/content/cs6998_05/drive/gdata/splits/summe_all.txt",
    #     # data_list_file = "/content/cs6998_05/drive/gdata/splits/2fps_summe_all.txt",
    #     gt_root = "/content/cs6998_05/drive/gdata/SumMe/GT", 
    #     debug_mode = True)

    # Use this for TVSum
    train_dataset = Summarizer_Dataset(
        dataset_name='tvsum', 
        data_list_file = "/content/cs6998_05/drive/gdata/splits/tvsum_all.txt",
        gt_root = "/content/cs6998_05/drive/gdata/TVSum/ydata-tvsum50.mat", 
        debug_mode = True)
    train_dataloader = DataLoader(train_dataset, batch_size=1)

    
    for i, data in enumerate(train_dataloader):
        print(f"{len(data['segments'])} segments of " +\
              f"{data['segments'][0].shape} shape and " +\
              f"{torch.squeeze(data['scores']).shape} importance scores obtained.")
        