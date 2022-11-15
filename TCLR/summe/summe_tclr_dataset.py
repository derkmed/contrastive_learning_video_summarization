import json
import math
import os
import pickle
import random
# from tqdm import tqdm
import time
from typing import Tuple

import cv2

import numpy as np

import summe.summe_config as scfg
import summe.summe_parameters as sparams
import torch
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from torchvision.transforms.functional import InterpolationMode

# from decord import VideoReader


class SummeTCLRDataset(Dataset):
    
    # This is the minimum number of frames a video input must have.
    MIN_FRAME_THRESHOLD = 56

    def __init__(self, repeats: int = sparams.n_reads_per_video, 
                 dataset_list_file: str = scfg.dataset_list_path, 
                 shuffle = True, data_percentage = 1.0):
        '''
        
        :param repeats: number of times to resample from the same video data files. Unlike UCF-101, SumMe has fewer but longer video files. 
        :param dataset_list_file: the file path location for the selected dataset elements (we may only want a subset of videos). This will replace "split."
        :param shuffle: Whether to randomly shuffle the data.
        :param data_percentage: the percentage of data specified in dataset_list to actually use.
        :param mode: 
        
        '''
        self.repeats = repeats
        
        # Reads a .txt file mapping file path to class ID (this should align with summe_classes.json). 
        # This will be fed into self.all_paths, a tuple of [file name, class ID].
        def _split_on_rightmost_delim(s: str, delim: str = ' ') -> Tuple[str, str]:
            return s.rsplit(' ', 1)[0]        
        all_paths_with_class_id = open(os.path.join(scfg.dataset_list_dir, dataset_list_file),'r')\
            .read().splitlines()
        self.all_paths = [_split_on_rightmost_delim(p) for p in all_paths_with_class_id]


        # Reads a JSON file mapping classes to an ID.
        self.classes= json.load(open(scfg.class_mapping))['classes']
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.all_paths)
        
        # Acquire the data. self.data should be List[Tuple[str, str]]
        self.data_percentage = data_percentage
        self.data_limit = int(len(self.all_paths) * self.data_percentage)
        
        unrepeated_data = self.all_paths[0 : self.data_limit] 
        self.data = [vid for vid in unrepeated_data for _ in range(self.repeats)]
        self.erase_size = 19
                       
    
    def __len__(self):
        return len(self.data)        

    def __getitem__(self, index):
        sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, \
                a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, list_sparse, list_dense, vid_path = self.process_data(index)
        return sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, \
                a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, list_sparse, list_dense, vid_path

    def process_data(self, idx):
        vid_path = f"{scfg.data_dir}/videos/{self.data[idx]}"
        sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, \
            a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, list_sparse, list_dense = self.build_clip(vid_path)
        return sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, \
                a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, list_sparse, list_dense, vid_path

    def build_clip(self, vid_path):
        '''
        Builds a (1) sparse clip that includes every `params.sr_ratio`-th frame for `params.num_frames` frames, (2) dense clips that span the 
        same length (in frames) of the sparse clip, but with `params.num_frames` contiguous frames broken up into 4 contiguous subsections and (3) the
        augmentations of the sparse clip and 4 dense clips. 
        
        Also returns a list of the frames in the sparse clip and all the frames in each of the 4 dense clips.
        '''

        try:
            # The following is just OpenCV2 stuff kept for consistency (from TCLR repo). 
            # Flag Reference can be found in https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html#ggaeb8dd9c89c10a5c63c139bf7c4f5704dadadc646b31cfd2194794a3a80b8fa6c2 .
            cap = cv2.VideoCapture(vid_path) 
            cap.set(1, 0)                 #  cv.CAP_ANY
            frame_count = cap.get(7)      # cv.CAP_PROP_FRAME_COUNT
            
            if frame_count <= SummeTCLRDataset.MIN_FRAME_THRESHOLD:
                raise ValueError(f"Video {vid_path} has insufficient frames")
            ############################# frame_list maker start here#################################
            #################################### SEE APPENDIX C.3 ####################################

            min_sparse_temporal_span = sparams.num_frames * sparams.sr_ratio     # 64
            if frame_count > min_sparse_temporal_span:
                # If there are "enough" frames, choose a random starting frame such that 
                # start_frame + min_sparse_temporal_span < total_video_frames.
                start_frame = np.random.randint(0, frame_count - min_sparse_temporal_span) 
                
                #Dynamic skip rate experiment
                # skip_max = int((frame_count - start_frame)/params.num_frames)
                # # here 4 is the skip rate ratio = 4 chunks
                # if skip_max >= 16:
                #     sr_sparse = np.random.choice([4,8,12,16])
                # elif (skip_max<16) and (skip_max>=12):
                #     sr_sparse = np.random.choice([4,8,12])
                # elif (skip_max<12) and (skip_max>=8):
                #     sr_sparse = np.random.choice([4,8])
                # else:
                sr_sparse = 4
            else:
                start_frame = 0
                sr_sparse = 4
            sr_dense = int(sr_sparse/4)
            
            # the global clip is composed of sparams.num_frames=16 frames sampled with a skip rate of sr_sparse=4.
            frames_sparse = [start_frame] + [start_frame + (i * sr_sparse) for i in range(1, sparams.num_frames)]

            # the 4 local clips also have sparams.num_frames=16 frames but are sampled with a skip rate of 1
            frames_dense = [[frames_sparse[j*4]] + [frames_sparse[j*4] + (i * sr_dense) for i in range(1, sparams.num_frames)] for j in range(4)]            

            ################################ frame list maker finishes here ###########################
            ################################ actual clip builder starts here ##########################

            sparse_clip = []
            dense_clip0 = []
            dense_clip1 = []
            dense_clip2 = []
            dense_clip3 = []

            a_sparse_clip = []
            a_dense_clip0 = []
            a_dense_clip1 = []
            a_dense_clip2 = []
            a_dense_clip3 = []

            list_sparse = []
            list_dense = [[] for i in range(4)]
            count = -1

            # We randomly sample 10 rows for each of the 10 clips = (2 augmentations * 1 sparse frame) + (2 augmentations * 4 dense frames).
            # We randomly sample 8 columns for each of the augmentation probabilities.
            random_array = np.random.rand(10, 8)

            ##### random augmentation hyperparameters starts here #####
            x_erase = np.random.randint(0, sparams.reso_h, size=(10,))
            y_erase = np.random.randint(0, sparams.reso_w, size=(10,))

            cropping_factor1 = np.random.uniform(0.6, 1, size = (10,)) # on an average cropping factor is 80% i.e. covers 64% area
            x0 = [np.random.randint(0, sparams.ori_reso_w - sparams.ori_reso_w * cropping_factor1[ii] + 1) for ii in range(10)]          
            y0 = [np.random.randint(0, sparams.ori_reso_h - sparams.ori_reso_h * cropping_factor1[ii] + 1) for ii in range(10)]

            contrast_factor1 = np.random.uniform(0.75, 1.25, size=(10,))
            hue_factor1 = np.random.uniform(-0.1, 0.1, size=(10,))
            saturation_factor1 = np.random.uniform(0.75, 1.25, size=(10,))
            brightness_factor1 = np.random.uniform(0.75, 1.25, size=(10,))
            erase_size1 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(10,))
            erase_size2 = np.random.randint(int(self.erase_size/2), self.erase_size, size=(10,))
            random_color_dropped = np.random.randint(0, 3, (10))
            gamma1 = np.random.uniform(0.75, 1.25, size=(10,))
            ##### random augmentation hyperparameters finishes here #####

            while(cap.isOpened()): 
                count += 1
                ret, frame = cap.read()
                if ((count not in frames_sparse) and (count not in frames_dense[0]) \
                    and (count not in frames_dense[1]) and (count not in frames_dense[2]) \
                    and (count not in frames_dense[3])) and (ret == True): 
                    continue
                if ret == True:
                    if (count in frames_sparse):
                        sparse_clip.append(self.augmentation(frame, random_array[0], x_erase[0], y_erase[0], cropping_factor1[0],\
                                x0[0], y0[0], contrast_factor1[0], hue_factor1[0], saturation_factor1[0], brightness_factor1[0],\
                                gamma1[0],erase_size1[0],erase_size2[0], random_color_dropped[0]))
                        a_sparse_clip.append(self.augmentation(frame, random_array[1], x_erase[1], y_erase[1], cropping_factor1[1],\
                                x0[1], y0[1], contrast_factor1[1], hue_factor1[1], saturation_factor1[1], brightness_factor1[1],\
                                gamma1[1],erase_size1[1],erase_size2[1], random_color_dropped[1]))
                        list_sparse.append(count)
                    if (count in frames_dense[0]):
                        dense_clip0.append(self.augmentation(frame, random_array[2], x_erase[2], y_erase[2], cropping_factor1[2],\
                                x0[2], y0[2], contrast_factor1[2], hue_factor1[2], saturation_factor1[2], brightness_factor1[2],\
                                gamma1[2],erase_size1[2],erase_size2[2], random_color_dropped[2]))
                        a_dense_clip0.append(self.augmentation(frame, random_array[3], x_erase[3], y_erase[3], cropping_factor1[3],\
                                x0[3], y0[3], contrast_factor1[3], hue_factor1[3], saturation_factor1[3], brightness_factor1[3],\
                                gamma1[3],erase_size1[3],erase_size2[3], random_color_dropped[3]))
                        list_dense[0].append(count)
                    if (count in frames_dense[1]):
                        dense_clip1.append(self.augmentation(frame, random_array[4], x_erase[4], y_erase[4], cropping_factor1[4],\
                                x0[4], y0[4], contrast_factor1[4], hue_factor1[4], saturation_factor1[4], brightness_factor1[4],\
                                gamma1[4],erase_size1[4],erase_size2[4], random_color_dropped[4]))
                        a_dense_clip1.append(self.augmentation(frame, random_array[5], x_erase[5], y_erase[5], cropping_factor1[5],\
                                x0[5], y0[5], contrast_factor1[5], hue_factor1[5], saturation_factor1[5], brightness_factor1[5],\
                                gamma1[5],erase_size1[5],erase_size2[5], random_color_dropped[5]))
                        list_dense[1].append(count)
                    if (count in frames_dense[2]):
                        dense_clip2.append(self.augmentation(frame, random_array[6], x_erase[6], y_erase[6], cropping_factor1[6],\
                                x0[6], y0[6], contrast_factor1[6], hue_factor1[6], saturation_factor1[6], brightness_factor1[6],\
                                gamma1[6],erase_size1[6],erase_size2[6], random_color_dropped[6]))
                        a_dense_clip2.append(self.augmentation(frame, random_array[7], x_erase[7], y_erase[7], cropping_factor1[7],\
                                x0[7], y0[7], contrast_factor1[7], hue_factor1[7], saturation_factor1[7], brightness_factor1[7],\
                                gamma1[7],erase_size1[7],erase_size2[7], random_color_dropped[7]))
                        list_dense[2].append(count)
                    if (count in frames_dense[3]):
                        dense_clip3.append(self.augmentation(frame, random_array[8], x_erase[8], y_erase[8], cropping_factor1[8],\
                                x0[8], y0[8], contrast_factor1[8], hue_factor1[8], saturation_factor1[8], brightness_factor1[8],\
                                gamma1[8],erase_size1[8],erase_size2[8], random_color_dropped[8]))
                        a_dense_clip3.append(self.augmentation(frame, random_array[9], x_erase[9], y_erase[9], cropping_factor1[9],\
                                x0[9], y0[9], contrast_factor1[9], hue_factor1[9], saturation_factor1[9], brightness_factor1[9],\
                                gamma1[9],erase_size1[9],erase_size2[9], random_color_dropped[9]))
                        list_dense[3].append(count)
                    
                else:
                    break
            if len(sparse_clip) < sparams.num_frames and len(sparse_clip) > 13:
                # if sparams.num_frames - len(sparse_clip) >= 1:
                #     print(f'sparse_clip {vid_path} is missing {params.num_frames - len(sparse_clip)} frames')
                remaining_num_frames = sparams.num_frames - len(sparse_clip)
                sparse_clip = sparse_clip + sparse_clip[::-1][1:remaining_num_frames+1]
                a_sparse_clip = a_sparse_clip + a_sparse_clip[::-1][1:remaining_num_frames+1]

            if len(dense_clip3) < sparams.num_frames and len(dense_clip3)>7:
                
                # if sparams.num_frames - len(dense_clip3) >= 1:
                #     print(f'dense_clip3 {vid_path} is missing {params.num_frames - len(dense_clip3)} frames')
                remaining_num_frames = sparams.num_frames - len(dense_clip3)
                dense_clip3 = dense_clip3 + dense_clip3[::-1][1:remaining_num_frames+1]    
                a_dense_clip3 = a_dense_clip3 + a_dense_clip3[::-1][1:remaining_num_frames+1]    
           
            try:
                assert len(sparse_clip)==sparams.num_frames, f"Sparse Clip has {len(sparse_clip)}, but expected {sparams.num_frames}."
                assert len(dense_clip0)==sparams.num_frames, f"Dense Clip 0 has {len(dense_clip0)}, but expected {sparams.num_frames}."
                assert len(dense_clip1)==sparams.num_frames, f"Dense Clip 1 has {len(dense_clip1)}, but expected {sparams.num_frames}."
                assert len(dense_clip2)==sparams.num_frames, f"Dense Clip 2 has {len(dense_clip2)}, but expected {sparams.num_frames}."
                assert len(dense_clip3)==sparams.num_frames, f"Dense Clip 3 has {len(dense_clip3)}, but expected {sparams.num_frames}."

                return sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, \
                        a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, \
                        list_sparse, list_dense
            except Exception as e:
                print(e)
                print(f'Clip {vid_path} has some frames reading issue, failed. This was most likely due to clip frame lengths')
                return None, None, None, None, None, None, None, None, None, None, None, None
        except Exception as e:
            print(e)
            print(f'Clip {vid_path} has some unknown issue, failed. This was most likely due to OpenCv or File issues.')
            return None, None, None, None, None, None, None, None, None, None, None, None

    def augmentation(self, image, random_array, x_erase, y_erase, cropping_factor1,\
        x0, y0, contrast_factor1, hue_factor1, saturation_factor1, brightness_factor1,\
        gamma1,erase_size1,erase_size2, random_color_dropped):
        
        image = transforms.functional.resized_crop(
            transforms.ToPILImage()(image),
            y0, x0,
            int(sparams.ori_reso_h*cropping_factor1),
            int(sparams.ori_reso_h*cropping_factor1),
            (sparams.reso_h, sparams.reso_w),
            interpolation=InterpolationMode.BILINEAR)

        if random_array[0] < 0.125:
            image = transforms.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[1] < 0.3 :
            image = transforms.functional.adjust_hue(image, hue_factor = hue_factor1) # hue factor will be between [-0.1, 0.1]
        if random_array[2] < 0.3 :
            image = transforms.functional.adjust_saturation(image, saturation_factor = saturation_factor1) # brightness factor will be between [0.75, 1,25]
        if random_array[3] < 0.3 :
            image = transforms.functional.adjust_brightness(image, brightness_factor = brightness_factor1) # brightness factor will be between [0.75, 1,25]
        if random_array[0] > 0.125 and random_array[0] < 0.25:
            image = transforms.functional.adjust_contrast(image, contrast_factor = contrast_factor1) #0.75 to 1.25
        if random_array[4] > 0.70:
            if random_array[4] < 0.875:
                image = transforms.functional.to_grayscale(image, num_output_channels = 3)
                if random_array[5] > 0.25:
                    image = transforms.functional.adjust_gamma(image, gamma = gamma1, gain=1) #gamma range [0.8, 1.2]
            else:
                image = transforms.functional.to_tensor(image)
                image[random_color_dropped,:,:] = 0
                image = transforms.ToPILImage()(image)

        if random_array[6] > 0.5:
            image = transforms.functional.hflip(image)

        image = transforms.functional.to_tensor(image)

        if random_array[7] < 0.5 :
            image = transforms.functional.erase(image, x_erase, y_erase, erase_size1, erase_size2, v=0) 

        return image

def collate_fn2(batch):
    '''
    The function passed as the collate_fn argument into a DataLoader is used to collate lists of samples into batches.
    '''

    sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, a_sparse_clip, \
    a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, \
    list_sparse, list_dense, vid_path = [], [], [], [], [], [], [], [], [], [], [], [], []
    for item in batch:
        if not (None in item):
            sparse_clip.append(torch.stack(item[0],dim=0)) 
            dense_clip0.append(torch.stack(item[1],dim=0))
            dense_clip1.append(torch.stack(item[2],dim=0))
            dense_clip2.append(torch.stack(item[3],dim=0))
            dense_clip3.append(torch.stack(item[4],dim=0))

            a_sparse_clip.append(torch.stack(item[5],dim=0))
            a_dense_clip0.append(torch.stack(item[6],dim=0))
            a_dense_clip1.append(torch.stack(item[7],dim=0))
            a_dense_clip2.append(torch.stack(item[8],dim=0))
            a_dense_clip3.append(torch.stack(item[9],dim=0))


            list_sparse.append(np.asarray(item[10]))
            list_dense.append(np.asarray(item[11]))
            vid_path.append(item[12])
        
    
    sparse_clip = torch.stack(sparse_clip, dim=0)
    dense_clip0 = torch.stack(dense_clip0, dim=0)
    dense_clip1 = torch.stack(dense_clip1, dim=0)
    dense_clip2 = torch.stack(dense_clip2, dim=0)
    dense_clip3 = torch.stack(dense_clip3, dim=0)

    a_sparse_clip = torch.stack(a_sparse_clip, dim=0)
    a_dense_clip0 = torch.stack(a_dense_clip0, dim=0)
    a_dense_clip1 = torch.stack(a_dense_clip1, dim=0)
    a_dense_clip2 = torch.stack(a_dense_clip2, dim=0)
    a_dense_clip3 = torch.stack(a_dense_clip3, dim=0)

    return sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, \
        a_sparse_clip, a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, \
        list_sparse, list_dense, vid_path
            
if __name__ == '__main__':
    train_dataset = SummeTCLRDataset(shuffle = True, data_percentage = 1.0)
    train_dataloader = DataLoader(train_dataset, batch_size=40, \
        shuffle=False, num_workers=4, collate_fn=collate_fn2)
    print(f'Step involved: {len(train_dataset)/24}')
    t=time.time()

    for i, (sparse_clip, dense_clip0, dense_clip1, dense_clip2, dense_clip3, a_sparse_clip, \
            a_dense_clip0, a_dense_clip1, a_dense_clip2, a_dense_clip3, \
            list_sparse, list_dense, vid_path) in enumerate(train_dataloader):
        if (i+1)%25 == 0:
            print(sparse_clip.shape)
            print(dense_clip3.shape)
            print()

    print(f'Time taken to load data is {time.time()-t}')