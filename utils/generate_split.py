import glob
import os
import random

def genSplitList(vid_directory: str, fileExt: str, outFile: str):
    '''
    Use this to generate the data split for TCLR given a video directory.
    Video data with file extensions of type fileExt will be included.
    '''
    video_files = glob.glob(f"{vid_directory}/*.{fileExt}")
    random.shuffle(video_files)
    
    with open(outFile, "w") as dst:
        for i, f in enumerate(video_files):
            dst.write(f"{f} {i}\n")

if __name__ == "__main__":
    VIDEO_DIR = '/home/derekhmd/data/videos/ovp_videos'
    genSplitList(VIDEO_DIR, ".mpg", "/home/derekhmd/cs6998_05/data/splits/ovp_all.txt")