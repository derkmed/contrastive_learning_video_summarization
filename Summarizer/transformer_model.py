
import math
import os
import re
from turtle import forward

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.cuda.amp import autocast

from Summarizer import r3d_classifier as r3d

class mlp(nn.Module):

    def __init__(self, final_embedding_size=128, use_normalization=True):
        
        super(mlp, self).__init__()

        self.final_embedding_size = final_embedding_size
        self.use_normalization = use_normalization
        self.fc1 = nn.Linear(512,512, bias = True)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(final_embedding_size)

        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, self.final_embedding_size, bias = False)
        self.temp_avg = nn.AdaptiveAvgPool3d((1,None,None))

    def forward(self, x):
        if torch.cuda.is_available():
            with autocast(): # only works with gpu
                x = self.temp_avg(x)                    # [B x 512 x 4 x 1 x 1] -> [B x 512 x 1 x 1 x 1]
                x = x.flatten(1)                        # [B x 512 x 1 x 1 x 1] -> [B x 512]
                x = self.relu(self.bn1(self.fc1(x)))
                x = nn.functional.normalize(self.bn2(self.fc2(x)), p=2, dim=1)
                return x
        else:
            x = self.temp_avg(x)
            x = x.flatten(1)
            x = self.relu(self.bn1(self.fc1(x)))
            x = nn.functional.normalize(self.bn2(self.fc2(x)), p=2, dim=1)
            return x

def load_tclr_backbone(saved_model_file: str = None, d_output: int = 128):
    model = r3d.r3d_18_classifier(pretrained=False, progress=False)
    # TCLR modifications from R3d.
    model.layer4[0].conv1[0] = nn.Conv3d(256, 512, kernel_size=(3, 3, 3),\
                                stride=(1, 2, 2), padding=(2, 1, 1),dilation = (2,1,1), bias=False)
    model.layer4[0].downsample[0] = nn.Conv3d(256, 512,\
                          kernel_size = (1, 1, 1), stride = (1, 2, 2), bias=False)

    # Weight copying.
    pretrained = None
    if torch.cuda.is_available():
        pretrained = torch.load(saved_model_file)
    else:
        pretrained = torch.load(saved_model_file, map_location=torch.device('cpu'))
    pretrained_kvpair = pretrained['state_dict']
    model_kvpair = model.state_dict()
    for layer_name, weights in pretrained_kvpair.items():
        if 'module.1.' in layer_name:
            continue              
        elif '1.' == layer_name[:2]:
            continue
        if 'module.0.' in layer_name:
            layer_name = layer_name.replace('module.0.','')
        if 'module.' in layer_name:
            layer_name = layer_name.replace('module.','')
        elif '0.' == layer_name[:2]:
            layer_name = layer_name[2:]
        model_kvpair[layer_name] = weights   
    model.load_state_dict(model_kvpair, strict=True)
    print(f'model {saved_model_file} loaded successsfully!')
    
    # Added for Summarization.
    model.fc = nn.Linear(512, d_output, bias=False)
    return model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TCLRSummarizer(nn.Module):

    def __init__(self, saved_model_file: str, d_model: int = 128, freeze_base: bool = True, 
        heads: int = 8, enc_layers: int = 6, dropout: float = 0.1) -> None:
        """
        d_model determines the output embedding dimensionality.
        """
        super(TCLRSummarizer, self).__init__()

        # NOTE: model expects expects [B x C x S x H x W] = [nsegments, nchannels, nframes_per_segments, size, size]
        self.base_model = load_tclr_backbone(saved_model_file, d_output=d_model)
        if freeze_base:
            # Freezes all layers except the last.
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.fc.requires_grad = True

        self.d_model = d_model
        self.mlp = mlp(final_embedding_size=d_model)
        self.pos_enc = PositionalEncoding(self.d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=heads, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=enc_layers
        )
        self.fc = nn.Linear(self.d_model, 1)

    def forward(self, video):
        # [n_segs, C, S, H, W] -> [n_segs x d_output x 4 x 1 x 1] --> [n_segs x d_output]
        video_emb = nn.Sequential(self.base_model, self.mlp)(video)
        
        # [n_segs, d_model] -> [1, n_segs, d_model]
        video_emb = video_emb.unsqueeze(0)
        video_emb = self.pos_enc(video_emb)  # Add pos enc as nn.Trasnformer doesnt have it
        
        video_emb = self.transformer_encoder(video_emb)
        video_emb = video_emb.contiguous().view(-1, self.d_model)
        logits = self.fc(video_emb)
        return video_emb, logits
    
def print_param_size(model_state_dict):
    for param_tensor in model_state_dict:
        print(param_tensor, "\t",model_state_dict[param_tensor].size())       


if __name__ == '__main__':
    # Test with `python -i -m Summarizer.model`.

    SAVED_MODEL_FILE = "/home/derekhmd/atv_summe_20221127_model_temp.pth"
    # print("Loading Manually")
    # model = torch.load(SAVED_MODEL_FILE, map_location=torch.device('cpu'))
    # model_state_dict = model['state_dict']
    # print_param_size(model_state_dict)


    print("Loading TCLR Backbone")
    tclr = load_tclr_backbone(SAVED_MODEL_FILE)
    summtclr1 = TCLRSummarizer(SAVED_MODEL_FILE, d_model=128)
    summtclr2 = TCLRSummarizer(SAVED_MODEL_FILE, d_model=256)
    summtclr5 = TCLRSummarizer(SAVED_MODEL_FILE, d_model=512)

    x = torch.rand(10, 3, 16, 112, 112) # 10 segments of 16 frames 3 x 112 x 112
    e = tclr(x)
    emb1, logits1 = summtclr1(x)
    emb2, logits2 = summtclr2(x)
    emb5, logits5 = summtclr5(x)
