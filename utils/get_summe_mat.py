import numpy as np
import os
import scipy.io


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

def getSummeGroundTruth(matFile: str):
    '''
    Helper function to access the .mat files that store ground truth
    information for summe.
    '''
    gt_file = f"{GT_ROOT}/{matFile}"
    gt_data = scipy.io.loadmat(gt_file)
    # Use the following keys to access the ground truth:
    #   gt_score = gt_data.get('gt_score')
    #   nFrames = gt_data.get('nFrames')
    return gt_data


if __name__ == "__main__":
    lines = open(DATALIST, 'r').read().splitlines()
    final_vid = None
    for line in lines:
        vid_file, idx = line.split(" ")    
        latest_mat_file = getSummeMatFileName(vid_file)
        print(latest_mat_file)

    gt_data = getSummeGroundTruth(latest_mat_file)
    print(gt_data['gt_score'])    