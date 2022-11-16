import torch.nn as nn


class LanguageGuidedAttentionBlock(nn.Module):
    def __init__(self, num_heads=4, embedding_dims=3):
        super(LanguageGuidedAttentionBlock, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            num_heads=num_heads, embedding_dims=embedding_dims)
        self.caption_mlp = nn.LazyLinear(512)

    def forward(self, visual_features, textual_features):
        # concatenate M textual features
        textual_features_concat = textual_features.flatten()
        text_embedding = self.caption_mlp(textual_features_concat)
        x = self.multi_head_attention(
            visual_features, text_embedding, text_embedding)
        return x


class FrameScoringTransformerBlock(nn.Module):
    def __init__(self):
        super(FrameScoringTransformerBlock).__init__()
        self.transformer = nn.Transformer(
            d_model=256, nhead=8, num_encoder=6, num_decoder=6)

    def forward(self, lg_attention):
        x = self.transformer(lg_attention, lg_attention)
        return x
