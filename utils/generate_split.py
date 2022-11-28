import glob
import os
import random
import argparse


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
    parser = argparse.ArgumentParser(
        description='Script to generate train/test splits')
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--video_extension', type=str,
                        required=False, default='mp4')
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    genSplitList(args.video_dir, args.video_extension, args.output_path)