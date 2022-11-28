import numpy as np
import os
import scipy.io
import argparse

DATALIST = '/Users/derekahmed/Documents/cs6998_05/data/splits/summe_allTrainTestList.txt'
GT_ROOT = '/Users/derekahmed/Documents/cs6998_05/data/gt'

def getSummeMatFileName(vid_abs_path: str) -> str:
    '''
    Helper function to obtain the .mat file that contains the summe ground truth
    filename corresponding to a provided absolute path.
    '''
    vid_extless = vid_abs_path.rsplit('.', 1)[0]
    vid_name = vid_extless.rsplit('/', 1)[-1]
    return f"{vid_name}.mat"

def getSummeGroundTruth(gt_root: str, matFile: str):
    '''
    Helper function to access the .mat files that store ground truth
    information for summe.
    '''
    gt_file = f"{gt_root}/{matFile}"
    gt_data = scipy.io.loadmat(gt_file)
    # Use the following keys to access the ground truth:
    gt_score = gt_data.get('gt_score')
    n_frames = gt_data.get('nFrames')
    print(gt_data.keys())
    print(gt_data['user_score'].shape)
    return gt_score, n_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summe GT reader')
    parser.add_argument('--datalist', type=str, required=True)
    parser.add_argument('--gt_root', type=str, required=True)
    args = parser.parse_args()
    lines = open(args.datalist, 'r').read().splitlines()
    final_vid = None
    for line in lines:
        vid_file, idx = line.split(" ")    
        latest_mat_file = getSummeMatFileName(vid_file)

    gt_score, n_frames = getSummeGroundTruth(args.gt_root, latest_mat_file) 
    print(gt_score)
    print(n_frames)