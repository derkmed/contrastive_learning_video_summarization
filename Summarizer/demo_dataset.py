import mat73
import math
import numpy as np
import os
import pandas as pd
import random
import time
import re
import json
import scipy.io
import sys
import torch


import torch
from torch.utils.data import Dataset
import torchvision.io as io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from typing import List, Tuple


class Demo_Dataset(Dataset):

    def __init__(self, 
        data_list_file: str = "../data/splits/summe_all.txt",
        gt_root: str = "../data/gt", 
        req_segment_count: int = 40, 
        num_frames_per_segment: int = 16, 
        size: int = 112,
        is_randomized_start: bool = False,
        is_video_only: bool = False,
        debug_mode: bool = False
    ):

        # This is a file with each line being a space-delimited
        # [absolute_path, class_index]
        self.data_list_file = data_list_file
        
        # This should be the dimension that size each frame is resized to.
        # The resized image will be a square of (self.size x self.size).
        self.size = size
        
        # This is the length of each segment in frames. 
        self.num_frames_per_segment = num_frames_per_segment

        # Provides optional segment padding to ensure dataset item is extended to
        # a specified number of segments.
        self.req_segment_count = req_segment_count if req_segment_count != -1 else -1
        
        data = open(self.data_list_file,'r').read().splitlines()
        assert_msg = lambda f : f"{f} is misconfigured. Expecting space delimited [FILE_NAME, IDX]"
        assert len(data[0].split(' ')) == 2, assert_msg(data_list_file)        
        self.videos = {int(row.split(' ')[1]): row.split(' ')[0] for row in data}
        self.is_video_only = is_video_only
        self.is_randomized_start = is_randomized_start
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
        
        # video = video.view(-1, 3, resized_dim, resized_dim) TODO(derekahmed) I don't think this is necessary
        return video # [B, C, H, W] 


    def _get_video_segments(self, video: torch.Tensor, frames_per_segment: int) -> torch.Tensor:
        '''
        Breaks up a video into segments of self.num_frames_per_segment 

        :param video: the input video as a tensor [B, C, H, W]
        :param frames_per_segment: number of frames to include in each segment
        :return: tensor of video segments (segments is 0th dimension)
        '''
        nframes =  video.shape[0]
        is_divisible = (nframes % frames_per_segment == 0)
        next_segment_start_idx = math.ceil(nframes / frames_per_segment) * frames_per_segment
        if not is_divisible and nframes < next_segment_start_idx:
            # Pad with zeros to ensure that each segment is whole.
            if self.debug_mode:
                print(f"[GetVideoSegments] " +\
                    f"Input video of {nframes} is not evenly divisible" +\
                    f" by {frames_per_segment} frames. Padding to {next_segment_start_idx}.")
            zeros = torch.zeros(
                (next_segment_start_idx - nframes, video.shape[1], video.shape[2], video.shape[3]),
                dtype=torch.uint8)
            video = torch.cat((video, zeros), axis=0)
        
        # [B, C, H, W] --> [B, H, W, C]
        video = video.permute(0, 2, 3, 1)
        splitted = torch.split(video, frames_per_segment)
        # [S, F, H, W, C] where S x F = B
        segments = torch.stack(splitted) 
        s, f, h, w, c = segments.shape
        return segments

    def _get_vidname(self, filepath: str) -> str:
        # remove the rightmost file-extension and return the rightmost filename
        # without the absolute path.
        return (filepath.rsplit('.')[0]).rsplit('/', 1)[-1]


    def _pad_segments(self, segments: torch.Tensor,  padded_length: int) -> torch.Tensor:
        '''
        Expects a segments tensor of 5 dimensions (imagine as a n_segments-length list of frames_per_segments frames
        where each frame is h x w x c). Pads the tensor such that the new length is padded_length.
        '''
        n_segments, frames_per_segment, h, w, c = segments.shape
        if padded_length < n_segments:
            raise ValueError(f"padded_length={padded_length} must be larger than the current number of segments={n_segments}")
        n_added_segments = padded_length - n_segments
        padding = torch.zeros((n_added_segments, frames_per_segment, h, w, c), dtype=torch.uint8)
        segments = torch.cat([segments, padding])
        return segments


    def _getfps(self, video_filepath: str) -> int:
        reader = io.VideoReader(video_filepath, "video")
        fps = reader.get_metadata()["video"]["fps"][0]
        return fps


    def __getitem__(self, idx):
        try:
            # Obtain the next video.        
            video_filepath = self.videos[idx]
            fps = self._getfps(video_filepath)
            vid_name = self._get_vidname(video_filepath)
            video = self._get_resized_video(video_filepath, self.size)
            nframes = video.shape[0]
            assert nframes != 0, "No video data retrieved: 0 frames read."
            
            # Obtain the video segments.
            segments = self._get_video_segments(video, self.num_frames_per_segment)
            nsegments, nsgement_frames, _, _, _ = segments.shape
            assert nsegments * nsgement_frames >= nframes, f"Segments should contain more or equivalent # frames as {nframes}."
            

            # Slice or pad to ensure self.req_segment_count number of segments.
            if self.req_segment_count != -1:
                if self.req_segment_count < nsegments:
                    if self.debug_mode:
                        print(f"Slicing segments {nsegments}->{self.req_segment_count}.")
                    segments = segments[:self.req_segment_count]
                elif self.req_segment_count > nsegments:
                    if self.debug_mode:
                        print(f"Padding segments {nsegments}->{self.req_segment_count}.")
                    segments = self._pad_segments(segments, self.req_segment_count)

            # [nsegments, nframes, H, W, nchannels] -> [nsegments, nchannels, nframes, H, W]
            segments = segments.permute(0, 4, 1, 2, 3)
            
            return {
                # "vid_name": vid_name,
                "segments": segments,
                # "fps": fps,
                # "n_frames": nframes,
                # "n_frames_per_segment": self.num_frames_per_segment,
                }
        
        except Exception as e:
            print(e)
            print("Cannot procure item. Error encountered")
            return None
