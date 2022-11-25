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
    VIDEO_DIR = '/Users/derekahmed/Documents/RepL/scp/data/videos/tvsumm_video'
    genSplitList(VIDEO_DIR, "mp4", f"/Users/derekahmed/Documents/RepL/scp/data/splits/all_tvsum_traintestlist.txt")