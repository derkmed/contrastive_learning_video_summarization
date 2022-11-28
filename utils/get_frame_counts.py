import glob
import sys
import mat73
import scipy


def getTvSumFrameCounts(mat_file: str):
    gt_data = mat73.loadmat(mat_file)['tvsum50']
    vid_frames = list(zip(gt_data['video'], list(map(lambda arr: arr.ravel()[0], gt_data['nframes']))))
    vid_length = {k: v for k, v in zip(gt_data['video'], list(map(lambda arr: arr.ravel()[0], gt_data['length'])))}
    max_frame_count = -1
    max_frame_count_duration = -1
    for vid, nFrames in vid_frames:
        print(f"TVSUM video {vid} has {nFrames} and is {vid_length[vid]} seconds")
        if max_frame_count < nFrames:
            max_frame_count = nFrames
            max_frame_count_duration = vid_length[vid]

    return max_frame_count, max_frame_count_duration

def getSummeGroundTruth(mat_dir: str):
    max_frame_count = -1
    max_frame_count_duration = -1
    for f in glob.glob(f"{mat_dir}/*.mat"):
        gt_data = scipy.io.loadmat(f)
        # print(gt_data['segments'].shape)
        nFrames = gt_data['nFrames'].ravel()[0]
        duration = gt_data['video_duration']
        ngt_scores = len(gt_data['gt_score'])
        print(f"SUMME Inspecting file {f} has {nFrames} frames with {ngt_scores} scores over {duration} seconds")

        if max_frame_count < nFrames:
                max_frame_count = nFrames
                max_frame_count_duration = nFrames

    return max_frame_count, max_frame_count_duration


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python get_frame_counts.py TVSUM_MAT_FILE SUMME_MAT_DIRECTORY")
        print("e.g. python get_frame_counts.py /Users/derekahmed/Documents/cs6998_05/data/gt/tvsum/ydata-tvsum50.mat /Users/derekahmed/Documents/cs6998_05/data/gt/summe")

    tvsum_mat = sys.argv[1]
    summe_mat = sys.argv[2]
  
    tvsum_max_frames, tvsum_max_duration = getTvSumFrameCounts(tvsum_mat)
    summe_max_frames, summe_max_duration = getSummeGroundTruth(summe_mat)
    if tvsum_max_frames > summe_max_frames:
        print(f"Longest video has {tvsum_max_frames} frames over {tvsum_max_duration} seconds")
    else:
        print(f"Longest video has {summe_max_frames} frames over {summe_max_duration} seconds")