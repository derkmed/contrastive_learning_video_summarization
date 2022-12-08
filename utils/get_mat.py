import mat73
import numpy as np
import os
import scipy.io
from typing import List


GT_ROOT = '/home/derekhmd/cs6998_05/data/gt'
TVSUM_GT_FILE = 'ydata-tvsum50.mat'

def getSummeMatFileName(vid_abs_path: str) -> str:
    '''
    Helper function to obtain the .mat file that contains the summe ground truth
    filename corresponding to a provided absolute path.
    '''
    vid_extless = vid_abs_path.rsplit('.', 1)[0]
    vid_name = vid_extless.rsplit('/', 1)[-1]
    return vid_name

def getSummeGroundTruth(vid_name: str):
    gt_file = f"{GT_ROOT}/{vid_name}.mat"
    gt_data = scipy.io.loadmat(gt_file)
    # Use the following keys to access the ground truth:
    #   gt_score = gt_data.get('gt_score')
    #   nFrames = gt_data.get('nFrames')
    return gt_data.get('gt_score')

def getTvsumGroundTruth(vid_name: str):
    MAX_SCORE = 5.0 # this is needed to normalize the scores.
    gt = mat73.loadmat(f"{GT_ROOT}/{TVSUM_GT_FILE}")
    videos = gt['tvsum50']['video']
    gt_idx = videos.index(vid_name)
    gt_scores = gt['tvsum50']['gt_score'][gt_idx]
    return gt_scores / MAX_SCORE, gt['tvsum50']['nframes'][0]

def getAvgTvsumGroundTruthCoverage(selection: List[str] = None):
    '''
    Answers "What % of frames are included in the summary, on average for a selection of videos?"
    '''
    MAX_SCORE = 5.0 # this is needed to normalize the scores.
    gt = mat73.loadmat(f"{GT_ROOT}/{TVSUM_GT_FILE}")
    videos = gt['tvsum50']['video']
    selection = videos if not selection else selection
    coverages = []
    for i, v in enumerate(selection):
        gt_idx = videos.index(v)
        gt_scores = gt['tvsum50']['gt_score'][gt_idx] / MAX_SCORE
        score_sum = sum(gt_scores)
        nFrames = gt['tvsum50']['nframes'][gt_idx]
        coverage = float(score_sum) / float(nFrames) * 100
        coverages.append(coverage)

    return sum(coverages) / len(coverages)


if __name__ == "__main__":

    
    summe_video_names = []
    for v in summe_video_names:
        gt_data = getSummeGroundTruth(v)
        print(gt_data.sum())

    tvsum_video_names = ['uGu_10sucQo',
        'GsAD1KT1xo8',
        'b626MiF1ew4',
        'RBCABdttQmI',
        'EE-bNr36nyA',
        'JKpqYvAdIsw',
        'XzYM3PfTM4w',
        'xwqBXPGE9pQ',
        'cjibtmSLxQ4',
        'iVt07TCkFM0'
    ]
    print(f"Average over selection of videos: {getAvgTvsumGroundTruthCoverage(tvsum_video_names)}")
    print(f"Average over all videos: {getAvgTvsumGroundTruthCoverage()}")