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

TVSUM_GT_FILE = 'ydata-tvsum50.mat'
SUMME_VIDEO_NAMES = [
    'Air_Force_One',
    'Base_jumping',
    'Bearpark_climbing',
    'Bike_Polo',
    'Bus_in_Rock_Tunnel',
    'Car_railcrossing',
    'Cockpit_Landing',
    'Cooking',
    'Eiffel_Tower',
    'Excavators_river_crossing',
    'Fire_Domino',
    'Jumps',
    'Kids_playing_in_leaves',
    'Notre_Dame',
    'Paintball',
    'Playing_on_water_slide',
    'Saving_dolphines',
    'Scuba',
    'St_Maarten_Landing',
    'Statue_of_Liberty',
    'Uncut_Evening_Flight',
    'Valparaiso_Downhill',
    'car_over_camera',
    'paluma_jump',
    'playing_ball'
    ]

class Summarizer_Dataset(Dataset):

    def __init__(self, 
        data_list_file: str = "../data/splits/summe_all.txt",
        gt_root: str = "../data/gt", 
        req_segment_count: int = 40, 
        num_frames_per_segment: int = 16, 
        size: int = 112,
        is_randomized_start: bool = True,
        is_video_only: bool = False,
        debug_mode: bool = False
    ):

        # This is a file with each line being a space-delimited
        # [absolute_path, class_index]
        self.data_list_file = data_list_file

        # This is the root ground truth folder.
        self.gt_root = gt_root
        
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


    def _get_orig_labels(self, vid_name: str) -> torch.FloatTensor:
        orig_labels = None
        # TODO(derekahmed) This code is really bad and assumes that if the dataset is not SumMe, it's TVSum.
        # It most definitely should be changed.
        if vid_name in SUMME_VIDEO_NAMES:
            orig_labels = self._get_summe_labels(vid_name)
        else:
            orig_labels = self._get_tvsum_labels(vid_name)
        return orig_labels

    def _get_segment_labels(self, vid_name: str, orig_labels: torch.Tensor, s_dimsize: int, npad_frames: int = -1) :
        '''
        Obtain the labels and pad and aggregate as necessary such that each segment can have
        a corresponding label_score.

        :param vid_name: the video name
        :param orig_labels: the original labels
        :param s_dimsize: the number of segments
        :param npad_frames: the number of frames to pad prior to aggregation. Ignore padding if -1.
        :return: tensor of label scores
        '''
        orig_labels = torch.FloatTensor(orig_labels)
        labels = orig_labels
        if npad_frames != -1:
            padded_length = len(orig_labels) + npad_frames
            if self.debug_mode:
                print(f"[GetSegmentLabels] " +\
                    f"Padding {len(labels)} labels to match the {padded_length} segment frames.")
            # We still need to pad to ensure a 1:1 mapping between segments frames & scores.
            labels = self._pad_labels(orig_labels, padded_length)
        
        if vid_name in SUMME_VIDEO_NAMES:
            return self._summe_aggregate_labels(labels, s_dimsize)
        else:
            return self._tvsum_aggregate_labels(labels, s_dimsize)

    def _get_segment_label_scores(self, vid_name: str, orig_labels: torch.Tensor, s_dimsize: int, npad_frames: int = -1) -> torch.FloatTensor:
        '''
        Obtain the label scores and pad and aggregate as necessary such that each segment can have
        a corresponding label_score.

        :param vid_name: the video to read ground truth data for
        :param s_dimsize: the number of segments
        :param npad_frames: the number of frames to pad prior to aggregation. Ignore padding if -1.
        :return: tensor of label scores
        '''            
        orig_labels = torch.FloatTensor(orig_labels)
        label_scores = orig_labels
        if npad_frames != -1:
            padded_length = len(orig_labels) + npad_frames
            if self.debug_mode:
                print(f"[GetSegmentLabelScores] " +\
                    f"Padding {len(label_scores)} label scores to match the {padded_length} segment frames.")
            # We still need to pad to ensure a 1:1 mapping between segments frames & scores.
            label_scores = self._pad_labels(orig_labels, padded_length)
        label_scores = self._aggregate_label_scores(orig_labels, s_dimsize)
        return label_scores

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
        gt = mat73.loadmat(f"{self.gt_root}/{TVSUM_GT_FILE}")
        videos = gt['tvsum50']['video']
        gt_idx = videos.index(vid_name)
        gt_scores = gt['tvsum50']['gt_score'][gt_idx]
        return gt_scores / MAX_SCORE 


    def _summe_aggregate_labels(self, orig_labels: torch.IntTensor, nsegments: int) -> torch.Tensor:
        '''
        Summe uses binary labels. We aggregate base on max count of labels from either binary value.
        '''
        if self.debug_mode:
            print(f"[AggregateLabels] aggregating {len(orig_labels)} into {nsegments} segments")

        # if orig_labels.is_floating_point():
            # raise ValueError("Summe aggregation assumes that labels are of Integer Type. But inputs are of floating point.")
        orig_nframes = len(orig_labels)
        orig_frames_per_segment = int(orig_nframes / nsegments)
        labels = torch.zeros(nsegments)
        for s_i in range(nsegments):
            count_1 = np.count_nonzero(
                    orig_labels[
                        s_i * orig_frames_per_segment : 
                        (s_i + 1) * orig_frames_per_segment
                    ]
                )
            count_0 = orig_frames_per_segment - count_1
            labels[s_i] = 1 if count_1 > count_0 else 0
        return torch.squeeze(labels)        

    def _tvsum_aggregate_labels(self, orig_labels: torch.FloatTensor, nsegments: int) -> torch.Tensor:
        '''
        Tvsum uses 1 to 5 scale, but we normalized it. We aggregate base on max count of labels >= 0.5.
        '''
        if self.debug_mode:
            print(f"[AggregateLabels] aggregating {len(orig_labels)} into {nsegments} segments")

        orig_nframes = len(orig_labels)
        orig_frames_per_segment = int(orig_nframes / nsegments)
        labels = torch.zeros(nsegments)
        for s_i in range(nsegments):
            pos_count = (orig_labels[s_i * orig_frames_per_segment :  (s_i + 1) * orig_frames_per_segment] >= 0.5)\
                .sum()
            neg_count = orig_frames_per_segment - pos_count
            labels[s_i] = 1 if pos_count > neg_count else 0
        return torch.squeeze(labels)        


    def _aggregate_label_scores(self, orig_labels: torch.FloatTensor, nsegments: int) -> torch.Tensor:
        '''
        The original FPS *may be higher than* the downsampled segments. This means we
        must aggregate the original PER-FRAME labels from the original video such
        that they correspond to the downsampled video.

        This is done via segmenting the original labels into contiguous groups. There may be some numerical
        inconsistencies, but in general this should work.

        :param downs_nframes: the number of frames in the downampled video
        :param orig_labels: the per-frame label scores in the original video
        :param nsegments: the number of segments to aggregate into
        :return: aggregated segment label_scores
        '''
        
        # Find out how many frames would be in each segment if we
        # were to divide the original video into the same number of
        # segments.
        if self.debug_mode:
            print(f"[AggregateLabelScores] aggregating {len(orig_labels)} into {nsegments} segments")

        orig_nframes = len(orig_labels)
        orig_frames_per_segment = int(orig_nframes / nsegments)
        label_scores = torch.zeros(nsegments)
        for s_i in range(nsegments):
            label_scores[s_i] = orig_labels[
                s_i * orig_frames_per_segment : 
                (s_i + 1) * orig_frames_per_segment
            ].mean()
        return torch.squeeze(label_scores)


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


    def _pad_labels(self, label_scores: torch.Tensor, padded_length: int):
        return torch.cat((
                label_scores, 
                torch.zeros(padded_length - len(label_scores), dtype=torch.uint8)))


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
            # TODO (derekahmed): ADD RANDOM OFFSET HERE
            randomized_start_segment = -1
            nrandomized_segments = -1
            if self.is_randomized_start:
                start_window_end = nsegments - self.req_segment_count
                start_window_end = start_window_end if start_window_end >= 0 else nsegments # if there are not enough frames, just default to the end
                randomized_start_segment = random.randint(0, start_window_end)
                if self.debug_mode:
                    print(f"Starting video segment is {randomized_start_segment}.")
                segments = segments[randomized_start_segment:,]
                nrandomized_segments, _, _, _, _ = segments.shape
            

            # Obtain the ground truth data.
            npad_frames = nsegments * nsgement_frames - nframes
            orig_labels = self._get_orig_labels(vid_name)            
            labels = self._get_segment_labels(vid_name, orig_labels, nsegments, npad_frames)
            label_scores = self._get_segment_label_scores(vid_name, orig_labels, nsegments, npad_frames)
            # TODO (derekahmed): ADD RANDOM OFFSET HERE
            if self.is_randomized_start:
                if self.debug_mode:
                    print(f"Starting segment label & score is {randomized_start_segment}.")
                labels = labels[randomized_start_segment:]
                label_scores = label_scores[randomized_start_segment:]
                nsegments = nrandomized_segments
            assert len(labels) == len(label_scores),\
                f"Incongruency detected between {len(labels)} labels and {len(label_scores)} scores."
            assert len(label_scores) == nsegments,\
                f"Incongruency detected between {len(label_scores)} scores and {nsegments} segments."
            

            # Slice or pad to ensure self.req_segment_count number of segments.
            if self.req_segment_count != -1:
                if self.req_segment_count < nsegments:
                    if self.debug_mode:
                        print(f"Slicing segments {nsegments}->{self.req_segment_count}.")
                        print(f"Slicing scores {len(label_scores)}->{self.req_segment_count}.")
                    segments = segments[:self.req_segment_count]
                    labels = labels[:self.req_segment_count]
                    label_scores = label_scores[:self.req_segment_count]

                elif self.req_segment_count > nsegments:
                    if self.debug_mode:
                        print(f"Padding segments {nsegments}->{self.req_segment_count}.")
                        print(f"Padding scores {len(label_scores)}->{self.req_segment_count}.")
                    # We need to pad segments to the length specified by the Dataset.
                    segments = self._pad_segments(segments, self.req_segment_count)
                    labels = self._pad_labels(labels, self.req_segment_count)
                    label_scores = self._pad_labels(label_scores, self.req_segment_count)
            
            # [nsegments, nframes, H, W, nchannels] -> [nsegments, nchannels, nframes, H, W]
            segments = segments.permute(0, 4, 1, 2, 3)
            if self.is_video_only:
                return {
                    # "vid_name": vid_name,
                    "segments": segments,
                    # "fps": fps,
                    "n_frames": nframes,
                    # "n_frames_per_segment": self.num_frames_per_segment,
                    }
            else:
                return {
                    # "vid_name": vid_name,
                    # "fps": fps,
                    "segments": segments,
                    "labels": labels,
                    "label_scores": label_scores,
                    "n_frames": nframes,
                    # "n_frames_per_segment": self.num_frames_per_segment,
                    }
        
        except Exception as e:
            print(e)
            print("Cannot procure item. Error encountered")
            return None


if __name__ == '__main__':

    rs_summe_dataset = Summarizer_Dataset(
        data_list_file = "../data/splits/all_summe.txt",
        req_segment_count = 256,
        debug_mode = True)
    summe_dataset = Summarizer_Dataset(
        data_list_file = "../data/splits/all_summe.txt",
        debug_mode = True)
    rs_tvsum_dataset = Summarizer_Dataset(
        data_list_file = "../data/splits/all_tvsum.txt",
        req_segment_count = 256,
        debug_mode = True)
    tvsum_dataset = Summarizer_Dataset(
        data_list_file = "../data/splits/all_tvsum.txt",
        debug_mode = True)

    
    # print("################################# DATASET AGNOSTIC TESTING #################################")
    # summarizer_dataset = rs_summe_dataset 

    # print("\n################################# Segment Padding Testing #################################")
    # rand_segments = torch.rand(50, 16, 112, 112, 3)
    # padded_length = 200
    # padded_segments = summarizer_dataset._pad_segments(rand_segments, padded_length)
    # assert padded_segments.shape[0] == padded_length, "_pad_segments(...) failed"
    # print("Check!")

    # print("\n################################# Label Score Aggregation Testing #################################")
    # rand_labels = torch.rand(4494)
    # down_nframes = 1440
    # nsegments =int(down_nframes / 16)
    # agg_scores = summarizer_dataset._aggregate_label_scores(rand_labels, nsegments)
    # assert agg_scores.shape[0] == nsegments, "_aggregate_label_scores(...) failed"
    # print("Check!")

    # print("\n################################# Getting Video Segments testing #################################")
    # video_filepath = summarizer_dataset.videos[0]
    # print(f"[TEST] Getting {video_filepath}")
    # video = summarizer_dataset._get_resized_video(video_filepath, summarizer_dataset.size)
    # nframes = video.shape[0]
    # segments = summarizer_dataset._get_video_segments(video, summarizer_dataset.num_frames_per_segment)
    # s, f, _, _, _ = segments.shape
    # assert s * f == math.ceil(nframes / summarizer_dataset.num_frames_per_segment) * summarizer_dataset.num_frames_per_segment,\
    #     f"_get_video_segments(...) failed {s * f} != {math.ceil(nframes / summarizer_dataset.num_frames_per_segment) * summarizer_dataset.num_frames_per_segment}"
    # print("Check!")

    # print("\n################################# DATASET SPECIFIC TESTING #################################")
    # print("\n################################# Label Aggregation Testing #################################")
    # rand_labels = torch.rand(4494)
    # down_nframes = 1440
    # nsegments =int(down_nframes / 16)
    # agg_scores = rs_summe_dataset._summe_aggregate_labels(rand_labels, nsegments)
    # assert agg_scores.shape[0] == nsegments, "_summeaggregate_labels(...) failed"
    # print("Check!")
    # rand_labels = torch.rand(4494)
    # down_nframes = 1440
    # nsegments =int(down_nframes / 16)
    # agg_scores = rs_tvsum_dataset._tvsum_aggregate_labels(rand_labels, nsegments)
    # assert agg_scores.shape[0] == nsegments, "_tvsumggregate_labels(...) failed"
    # print("Check!")

    # print("\n################################# Test summe importance score loading. #################################")
    # summe_vid = 'Air_Force_One'
    # summe_vid_is = rs_summe_dataset._get_summe_labels(summe_vid)
    # assert len(summe_vid_is) == 4494, "Air_Force_One should have 4494 frames"
    # print("Check!")
    
    # print("\n################################# Test tvsum importance score loading. #################################")
    # tvsum_vid = 'eQu1rNs0an0'
    # tvsum_vidis = rs_tvsum_dataset._get_tvsum_labels(tvsum_vid)
    # assert len(tvsum_vidis) == 4931, "eQu1rNs0an0 should have 4931 frames"
    # print("Check!")
    
    # print("\n################################# Getting Summe Item testing #################################")
    # # In this test, we are *PADDING* segments.
    # rs_summe_data = rs_summe_dataset.__getitem__(0) # Should be playing_ball.mp4 padded up to 256 segments.
    # padded_segments = rs_summe_data['segments']
    # padded_scores = rs_summe_data['label_scores']
    # padded_nframes = rs_summe_data['n_frames']
    # padded_nframes_per_seg = rs_summe_data['n_frames_per_segment']
    # assert padded_scores.shape[0] == rs_summe_dataset.req_segment_count,\
    #     f"{padded_segments.shape[0]} segments found but {rs_summe_dataset.req_segment_count} segments wanted"
    # assert len(padded_scores) == rs_summe_dataset.req_segment_count,\
    #     f"{len(padded_scores)} scores found but {rs_summe_dataset.req_segment_count} scores wanted"    
    # # Check that the unpadded portions match.
    # segment_padding_start_idx = int(padded_nframes / padded_nframes_per_seg) + 1
    # derived_unpadded_segments = padded_segments[:segment_padding_start_idx]
    # derived_unpadded_scores = padded_scores[:segment_padding_start_idx]
    # summe_data = summe_dataset.__getitem__(0)
    # assert torch.all(derived_unpadded_segments == summe_data['segments']).item(), "Summe segment padding is incorrect."
    # assert torch.all(derived_unpadded_scores == summe_data['label_scores']).item(), "Summe score padding is incorrect."
    # print("Check!")

    # print("\n################################# Getting Tvsum Item testing #################################")
    # # In this test, we are *REMOVING* segments because there are too many.
    # rs_tvsum_data = rs_tvsum_dataset.__getitem__(0) # Should be eQu1rNs0an0.mp4 sliced down to 256 segments.
    # sliced_segments = rs_tvsum_data['segments']
    # sliced_scores = rs_tvsum_data['label_scores']
    # sliced_nframes = rs_tvsum_data['n_frames']
    # sliced_nframes_per_seg = rs_tvsum_data['n_frames_per_segment']
    # assert sliced_scores.shape[0] == rs_tvsum_dataset.req_segment_count,\
    #     f"{sliced_segments.shape[0]} segments found but {rs_tvsum_dataset.req_segment_count} segments wanted"
    # assert len(sliced_scores) == rs_tvsum_dataset.req_segment_count,\
    #     f"{len(sliced_scores)} scores found but {rs_tvsum_dataset.req_segment_count} scores wanted"    
    # # Check that slicing occurred.
    # tvsum_data = tvsum_dataset.__getitem__(0)
    # assert sliced_segments.shape[0] < tvsum_data['segments'].shape[0], "tvsum segment slicing is incorrect."
    # assert len(sliced_scores) < len(tvsum_data['label_scores']), "tvsum score slicing is incorrect."
    # print("Check!")



    # # ###################################################################################################
    # print("\n################################# Iteration Test #################################")
    # dataset = Summarizer_Dataset(data_list_file = "../data/splits/8fps_augmented_80p_tvsum.txt",
    #     req_segment_count=100)
    # train_dataloader = DataLoader(dataset, batch_size=6)
    # last_og_labels, last_og_segment_time = None, None
    # for i, data in enumerate(train_dataloader):
    #     for j in range(data['segments'].shape[0]):
    #         print(f"Batch={i}, item={j}")
    #         segments = data['segments'][j]
    #         name = data['vid_name'][j]
    #         fps = data['fps'][j].item()
    #         nframes = data['n_frames'][j]
    #         labels = data['labels'][j]
    #         scores = data['label_scores'][j]
    #         nframes_per_segment = data['n_frames_per_segment'][j].item()
            
    #         nsegments = segments.shape[0]
    #         og_labels = dataset._get_orig_labels(name)
    #         og_nframes = og_labels.shape[0]
    #         segment_time = nframes_per_segment / fps
    #         last_og_segment_time = segment_time
    #         print(f"One segment spans {segment_time} seconds with a total video time of {nsegments * segment_time}.")
    #         print(f"OG nframes_per_segment = {og_nframes / nsegments}")
    #         print(f"{name} at {fps} FPS with {og_nframes} frames --> {nframes} frames --> {nsegments} segments @ {nframes_per_segment} frames per segment.")
    #         print(f"{torch.squeeze(scores).shape[0]} importance scores obtained.")        
    #         print(f"Sum disparity between labels ({labels.sum()}) and scores ({scores.sum()}) observed.")
    #         if len((labels == 1).nonzero()) != 0:
    #             start_marker = (labels == 1).nonzero()[0].item() * nframes_per_segment / fps
    #             print(f"First important segment is at {start_marker} s.")
    #             print(labels)
    #             print(scores)
    #             last_og_labels = og_labels
                
    #             print(og_labels.shape)
    #         break
    #     break
            
    # This is just a manual check of the first n segment scores / labels.
    # last_og_fps = 30
    # last_og_segment_size = int(last_og_segment_time * last_og_fps)
    # for i in range(50):
    #     print(f"{last_og_segment_time * i}: {last_og_labels[i * last_og_segment_size : (i + 1) * last_og_segment_size].mean()}")
    # for i in range(50):
    #     print(f"{last_og_segment_time * i}: {last_og_labels[i * last_og_segment_size : (i + 1) * last_og_segment_size].sum()}")


    # # ###################################################################################################
    # print("\n################################# Aggregation Test #################################")
    # dataset = tvsum_dataset
    # assert torch.all(dataset._tvsum_aggregate_labels(torch.Tensor([0.2, 0, 0.3, 0, 0, 1, 1, 1, 1]), 3) == torch.Tensor([0, 0, 1])).item(), "_tvsum_aggregate_labels(...) is wrong"

    # dataset = tvsum_dataset
    # assert torch.all(dataset._summe_aggregate_labels(torch.Tensor([0.2, 0, 0.3, 0, 0, 1, 1, 1, 1]), 3) == torch.Tensor([1, 0, 1])).item(), "_summe_aggregate_labels(...) is wrong"

    # # ###################################################################################################
    # print("\n################################# Randomized Start Test #################################")
    random_dataset = Summarizer_Dataset(
        data_list_file = "../data/splits/all_summe.txt",
        req_segment_count = 40,
        is_randomized_start=True,
        debug_mode = True)
    train_dataloader = DataLoader(random_dataset, batch_size=6)
    last_og_labels, last_og_segment_time = None, None
    for i, data in enumerate(train_dataloader):
        for j in range(data['segments'].shape[0]):
            break
        break



