import cv2
import model as sm
import demo_dataset

import sys
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.io as io

'''
This example script shows how to run a forward pass of the model.
'''


if __name__ == "__main__":
    # TODO (derekahmed) SMH this code is awful. We're hard-coding the 
    # original framerate, downsampled, etc. Rather than having it programmatically consistent
    # with the underlying data.
    OG_FPS = 30 # 30fps
    DS_FPS = 2 # 2fps
    BEST_MODEL = "/home/derekhmd/cs6998_05/data/checkpoints/exp_model_bs_24_lrbase_3e-05_lrimportance_3e-05_nsegments_56_nfps_16_nheads_8_nenc_6_dropout_0.1/epoch0220.pth.tar"
    K = 0.1
    RAW_VIDEO_PATH = "/home/derekhmd/data/videos/360cv_lecture.mp4"
    SUMMARY_PATH = "/home/derekhmd/data/videos/cv_summary.mp4"


    model_path = None
    if len(sys.argv) < 2:
        print("Usage: python forward_pass.py BEST_MODEL")
        print(f"Defaulting to {BEST_MODEL}")
        model_path = BEST_MODEL
    elif len(sys.argv) == 2:
        model_path = sys.argv[1]

    # Procure 1 2-FPS element dataset.
    dataset =  demo_dataset.Demo_Dataset(
        data_list_file = "../data/splits/cv_lecture.txt",
        req_segment_count = 200,
        debug_mode = True)
    train_dataloader = DataLoader(dataset, batch_size=1)
    model = sm.load_summarizer(BEST_MODEL, d_model=512, heads=8, enc_layers=6)
    v = model.cuda().float()
    v = None
    for i, data in enumerate(train_dataloader):
        v = data['segments']
        break
    
    # Call inference.
    v = v.cuda().float()
    segment_embeddings, logits = model(v)        
    segment_embeddings, logits = segment_embeddings[0], logits[0] # batch size of 1

    # These are the top K % segments.
    summary_ids = (logits.detach().cpu().view(-1).topk(int(K * logits.shape[0]))[1])
    summary_id_list = sorted(summary_ids.tolist())
    
    # Map segments to downsampled frames (start, end)
    ds_sec_ranges = [(16 * e / DS_FPS, 16 * (e + 1) / DS_FPS) for e in summary_id_list]
    og_frame_ranges = [(s * OG_FPS, e * OG_FPS) for s, e in ds_sec_ranges]
    
    # Now use the mapped frame ranges to generate a new summary at the original framerate.
    writer = cv2.VideoWriter(SUMMARY_PATH,
                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                            30, (640, 360))
    cap = cv2.VideoCapture(RAW_VIDEO_PATH) 
    cap.set(1, 0)                 #  cv.CAP_ANY
    frame_count = cap.get(7)      # cv.CAP_PROP_FRAME_COUNT

    summary_frames = []
    frame_i = 0
    frame_range_idx = 0
    while(cap.isOpened()):         
        curr_start, curr_end = og_frame_ranges[frame_range_idx]

        ret, frame = cap.read()
        if not ret or frame_range_idx >= len(og_frame_ranges):
            break
        elif frame_i >= curr_end:
            # Progress idx to the next range of frames to add.
            frame_range_idx += 1
            if frame_range_idx >= len(og_frame_ranges):
                break
            curr_start, curr_end = og_frame_ranges[frame_range_idx]
        
        if frame_i >= curr_start and frame_i < curr_end:
            writer.write(cv2.resize(frame,(640, 360)))
            summary_frames.append(frame)

        frame_i += 1
    print(f"Summary can be found in {SUMMARY_PATH}!")
    writer.release()