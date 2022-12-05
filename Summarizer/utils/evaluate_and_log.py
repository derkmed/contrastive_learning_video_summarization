import os
from utils import AverageMeter, video_from_summary
from collections import OrderedDict
from prettytable import PrettyTable
import json
import numpy as np
from tqdm import tqdm
import sys

sys.path.append("../")
from visualization import generate_html


def evaluate_summary(machine_summary, gt_summary):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and gt_summary should be binary vectors of ndarray type.
    """
    machine_summary = np.asarray(machine_summary, dtype=np.float32)
    gt_summary = np.asarray(gt_summary, dtype=np.float32)
    # n_frames = gt_summary.shape[0]
    # if len(machine_summary) > n_frames:
    #     machine_summary = machine_summary[:n_frames]
    # elif len(machine_summary) < n_frames:
    #     zero_padding = np.zeros((n_frames - len(machine_summary)))
    #     machine_summary = np.concatenate([machine_summary, zero_padding])
    
    if gt_summary.sum() == 0:
        return 1.0, 1.0, 1.0, (-1, -1, -1)

    overlap_duration = (machine_summary * gt_summary).sum()
    precision = overlap_duration / (machine_summary.sum() + 1e-8)
    recall = overlap_duration / (gt_summary.sum() + 1e-8)
    if precision == 0 and recall == 0:
        f_score = 0.0
    else:
        f_score = (2 * precision * recall) / (precision + recall)

    return f_score, precision, recall, (overlap_duration, machine_summary.sum(), gt_summary.sum())

