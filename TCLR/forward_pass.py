from  torch.cuda.amp import autocast

from tclr_pretraining import mlp
from tclr_pretraining import r3d

import torch
import torch.nn as nn

import nn_retrieval.model as enc


SELECTED_MODEL_FILE = 'cs6998_05-tclr_summ/weights/model_best_e247_loss_9.7173.pth'


if __name__ == "__main__":


    # Randomly create 5 videos of 16 frame length, each frame with 3 112 x 112 channels. 
    input = torch.rand(5, 16, 3, 112, 112).cuda() 
    input = input.permute(0, 2, 1, 3, 4)
    
    model = enc.build_r3d_encoder_ret(num_classes = 102, saved_model_file=SELECTED_MODEL_FILE)
    
    if torch.cuda.is_available():
        # Make sure the model and input are both on cuda!
        # https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
        print('cuda found')
        model.cuda()
    else:
        raise RuntimeError("No GPU found. Aborting program")

    output = model(input)
    print(output.shape)
