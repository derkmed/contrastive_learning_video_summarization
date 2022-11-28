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
import mat73
import math

from torch.utils.data import DataLoader, Dataset
import scipy.io
from typing import List, Tuple


class Summarizer_Dataset(Dataset):

    def __init__(self, 
        dataset_name: str = 'summe',
        data_list_file: str = "../data/splits/summe_all.txt",
        gt_root: str = "../data/gt", # this should be set to a file if tvsum!
        req_segment_count: int = -1, 
        num_frames_per_segment: int = 16, 
        size: int = 112,
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
        
        # This is the length of each segment in frames. 
        self.num_frames_per_segment = num_frames_per_segment

        # Provides optional segment padding to ensure dataset item is extended to
        # a specified number of segments.
        self.req_segment_count = req_segment_count if req_segment_count != -1 else -1

        # 'tvsum' or 'summe'. This is only relevant if is_video_only = False
        self.dataset_name = dataset_name
        
        data = open(self.data_list_file,'r').read().splitlines()
        assert_msg = lambda f : f"{f} is misconfigured. Expecting space delimited [FILE_NAME, IDX]"
        assert len(data[0].split(' ')) == 2, assert_msg(data_list_file)        
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
        
        # video = video.view(-1, 3, resized_dim, resized_dim) TODO(derekahmed) I don't think this is necessary
        return video # [B, C, H, W] 


    def _get_video_segments(self, 
        video_path: str, resized_dim: int, frames_per_segment: int) -> Tuple[List[torch.Tensor], int]:
        '''
        Breaks up a video into segments of self.num_frames_per_segment length.

        :param video_path: the input video path
        :param resized_dim: new square dimension
        :param frames_per_segment: number of frames to include in each segment
        :return: List[Tensor], int list of segments [B, H, W, C]
        '''
        video = self._get_resized_video(video_path, resized_dim)
        nframes = video.shape[0]
        assert nframes != 0, "No video data retrieved: 0 frames read."

        
        is_divisible = (nframes % frames_per_segment == 0)
        next_segment_start_idx = math.ceil(nframes / frames_per_segment) * frames_per_segment
        if not is_divisible and nframes < next_segment_start_idx:
            # Pad with zeros to ensure that each segment is whole.
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
        # [S, F, H, W, C] where S x F = B
        segments = torch.stack(splitted) 
        return segments, nframes


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
        nsegments: int,
        orig_labels: torch.FloatTensor) -> torch.Tensor:
        '''
        The original FPS *may be higher than* the downsampled segments. This means we
        must aggregate the original PER-FRAME labels from the original video such
        that they correspond to the downsampled video.

        This is achieved by first deriving the number of segments in each video based
        on the downsampled video. Then, we refer to the original video PER-FRAME labels
        and aggregate them based on which segment the fall under. The segment-level label
        score is calculated as the average of frame-level label scores, resulting in a score in [0, 1].

        :param downs_nframes: the number of frames in the downampled video
        :param num_frames_per_segment: the number of frames in each segment
        :param nsegments: the number of segments in each video
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


    def _pad_segments(self, segments: torch.Tensor,  padded_length: int) -> torch.Tensor:
        n_segments, frames_per_segment, h, w, c = segments[0].shape
        n_added_segments = padded_length - n_segments
        padding = torch.zeros((n_added_segments, frames_per_segment, h, w, c), dtype=torch.uint8)
        segments = torch.cat([segments, padding])
        return segments


    def _pad_labels(self, label_scores: torch.Tensor, padded_length: int):
        return torch.cat(
            label_scores, torch.zeros(padded_length - len(label_scores), dtype=torch.uint8))


    def _getfps(self, video_filepath: str) -> int:
        reader = io.VideoReader(video_filepath, "video")
        fps = reader.get_metadata()["video"]["fps"][0]
        return fps


    def __getitem__(self, idx):
        try:
            # Obtain the next video.        
            video_filepath = self.videos[idx]
            vid_name = self._get_vidname(video_filepath)
            segments, nframes = self._get_video_segments(video_filepath, self.size,
                                                self.num_frames_per_segment)
            nsegments = len(nsegments)
            dwns_nframes = nsegments * self.num_frames_per_segment
            

            # Obtain the ground truth data. Given that the FPS of the input
            # may not match the original source in the original Summe & TVSum
            # datasets, we will need to aggregate the frame-level importance scores
            # accordingly.
            orig_labels = None
            if self.dataset_name == "summe":
                orig_labels = self._get_summe_labels(vid_name)
            elif self.dataset_name == "tvsum":
                orig_labels = self._get_tvsum_labels(vid_name)
            else:
                raise ValueError('Invalid dataset provided.')

            
            # downsampled_fps = self._getfps(video_filepath)
            label_scores = self._get_labels_for_downsampled_segments(
                dwns_nframes, self.num_frames_per_segment, nsegments, orig_labels)
            
            if self.req_segment_count != -1:
                # We need to pad segments to the length specified by the Dataset.
                segments = self._pad_segments(segments, self.req_segment_count)
                label_scores = self._pad_labels(label_scores, self.req_segment_count)
            elif self.req_segment_count < nsegments:
                raise ValueError(f"Segment constraint of {self.req_segment_count} is too small for video with {nsegments} segments.")
            
            assert label_scores.shape[0] == nsegments, "Something is wrong with the code. Segment lengths do not match."
            if self.is_video_only:
                return {
                    "segments": segments,
                    "n_frames": nframes,
                    "n_frames_per_segment": self.num_frames_per_segment
                    }
            else:
                return {
                    "segments": segments,
                    "scores": label_scores,
                    "n_frames": nframes,
                    "n_frames_per_segment": self.num_frames_per_segment
                    }
        
        except Exception as e:
            print("Encountered Exception :( : " + e)
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

    ###### SumMe Testing
    # labels = train_dataset._get_summe_labels('playing_ball')
    # segments = train_dataset._get_video_segments('/content/cs6998_05/drive/gdata/SumMe/videos/playing_ball.mp4', 112, 32)
    # label_scores = train_dataset._get_labels_for_downsampled_segments(
    #           1844, 32, len(segments), labels)
    # e = train_dataset.__getitem__(0)


    # # Use this for TVSum
    # train_dataset = Summarizer_Dataset(
    #     dataset_name='tvsum', 
    #     data_list_file = "../data/splits/summe_all.txt",
    #     gt_root = "../data/gt/summe", 
    #     debug_mode = True)
    # train_dataloader = DataLoader(train_dataset, batch_size=1)

    # for i, data in enumerate(train_dataloader):
    #     print(f"{len(data['segments'])} segments of " +\
    #           f"{data['segments'][0].shape} shape and " +\
    #           f"{torch.squeeze(data['scores']).shape} importance scores obtained.")
        