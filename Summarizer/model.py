import torch
import torch.nn as nn
import torch.nn.functional as F
from TCLR.linear_eval.model import load_r3d_classifier

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
        attn_output, _ = self.multi_head_attention(
            visual_features, text_embedding, text_embedding)
        return attn_output

class SummarizerModel(nn.Module):
    def __init__(self, num_classes=102, saved_model_file=None, num_heads=8, freeze_tclr=True):
        super(SummarizerModel, self).__init__()
        self.tclr = load_r3d_classifier(num_classes, saved_model_file)
        if freeze_tclr:
            for param in self.tclr.parameters():
                param.requires_grad = False

        self.embedding_dims = 512
        self.multi_head_attention = nn.MultiheadAttention(self.embedding_dims, num_heads=num_heads)
        self.mlp = nn.LazyLinear(1)
    
    def forward(self, video):
        tclr_embeddings = []
        for segment in video:
            output = self.tclr(segment)
            tclr_embeddings.append(output)
        tclr_embeddings = torch.stack(tclr_embeddings)
        output = self.mlp(tclr_embeddings)
        output = output.flatten()
        return output