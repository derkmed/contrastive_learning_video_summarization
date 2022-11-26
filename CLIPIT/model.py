import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LanguageGuidedAttentionBlock(nn.Module):
    def __init__(self, num_heads=32, embedding_dim=512):
        super(LanguageGuidedAttentionBlock, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads)
        self.caption_mlp = nn.LazyLinear(512)

    def forward(self, visual_features, textual_features):
        # concatenate M textual features
        textual_features_concat = textual_features.flatten()
        text_embedding = self.caption_mlp(textual_features_concat)
        text_embedding = torch.unsqueeze(text_embedding, dim=0)
        attn_output, attn_output_weights = self.multi_head_attention(
            visual_features, text_embedding, text_embedding)
        return attn_output

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


class FrameScoringTransformerBlock(nn.Module):
    def __init__(self, num_frames, d_model=512, num_head=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation= F.relu, layer_norm_eps=1e-5, batch_first=False, norm_first=False, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(FrameScoringTransformerBlock, self).__init__()
        self.d_model = d_model
        self.pos_enc = PositionalEncoding(self.d_model, 0.1)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model, num_head, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, **factory_kwargs)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )
        self.decoder = nn.Linear(d_model * num_frames, 1)
        self.activation = nn.Softmax()
        self._reset_parameters()

        self.d_model = d_model
        self.num_head = num_head
        self.num_frames = num_frames

        self.batch_first = batch_first

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, lg_attention):
        position = self.pos_enc(lg_attention)
        x = self.transformer_encoder(position)
        x = x.contiguous().view(-1, self.num_frames * self.d_model)
        x = self.decoder(x)
        x = x.flatten()
        x = self.activation(x)
        return x

class TLDWTransformerBlock(nn.Module):
    pass

if __name__ == '__main__':
    video_embedding = torch.rand(30, 512)
    text_embedding = torch.rand(7, 512)

    lga = LanguageGuidedAttentionBlock()
    fst = FrameScoringTransformerBlock(30)

    x = lga(video_embedding, text_embedding)
    x = fst(x)