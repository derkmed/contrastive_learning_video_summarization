import ffmpeg
import glob
import os

DATA_DIR = "./data/summe_videos"
DOWNSAMPLE_FPS = 2
DOWNSAMPLED_DIR = f"./data/downs_summe_videos_{DOWNSAMPLE_FPS}"

if not os.path.exists(DOWNSAMPLED_DIR):
  os.makedirs(DOWNSAMPLED_DIR)


mp4_video_files = glob.glob(f"{DATA_DIR}/*.mp4")
for i, f in enumerate(mp4_video_files):
  fname = os.path.basename(f)        # just the filename without the path
  print(f"Downsampling {fname}")
  stream = ffmpeg.input(f)
  stream = ffmpeg.filter(stream, 'fps', fps=DOWNSAMPLE_FPS, round='up')
  stream = ffmpeg.output(stream, f"{DOWNSAMPLED_DIR}/dwn_{fname}")
  ffmpeg.run(stream)