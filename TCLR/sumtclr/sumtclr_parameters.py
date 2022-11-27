num_frames = 16
batch_size = 8
learning_rate = 1e-3 #base learning rate
num_epochs = 1000 # training code will end after this num_of_epochs
data_percentage = 1.0 #just for debugging purpose, default = 1
temperature = 0.1 
weight_decay = 1e-9
sr_ratio = 4

# Original resolution parameters below.
ori_reso_h = 360
ori_reso_w = 480

# TCLR resolutions (should be 112 x 112).
reso_h = 112
reso_w = 112

warmup = 5 
warmup_array = [1/100,1/20,1/10,1/2,1]
scheduler_patience = 9


# Configure this value to determine how many times to randomly
# sample from each long video, i.e. setting this parameter to
# 5 will lead to 5 x [2 x (1 sparse clips) + 2 x (4 dense clips)]
# from each video. 
# This turned out to be pointless due to computational constraints.
n_reads_per_video = 1