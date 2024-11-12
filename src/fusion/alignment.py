import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class CrossAttentionFusion(nn.Module):
    def __init__(self, audio_dim, text_dim, fused_dim):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, fused_dim)
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4)

    def forward(self, audio_features, text_features):
        aligned_audio, aligned_text = align_features(
            audio_features.cpu().detach().numpy(), 
            text_features.cpu().detach().numpy()
        )
        aligned_audio = aligned_audio.to(audio_features.device)
        aligned_text = aligned_text.to(text_features.device)

        audio_proj = self.audio_proj(audio_features)
        text_proj = self.text_proj(text_features)

        # Transpose for MultiheadAttention compatibility
        audio_proj = audio_proj.permute(1, 0, 2)
        text_proj = text_proj.permute(1, 0, 2)

        # Apply cross-attention
        fused_features, _ = self.cross_attention(audio_proj, text_proj, text_proj)

        # Transpose back
        fused_features = fused_features.permute(1, 0, 2)
        return fused_features



def align_features(audio_features, text_features):
    """
    Perform DTW alignment between audio and text features and return fused features.
    """
    # Compute pairwise cosine distance between audio and text features
    distance_matrix = cdist(audio_features, text_features, metric="cosine")

    # Perform DTW alignment
    row_indices, col_indices = linear_sum_assignment(distance_matrix)

    # Align audio and text features based on indices
    aligned_audio = torch.tensor(audio_features[row_indices], dtype=torch.float32)
    aligned_text = torch.tensor(text_features[col_indices], dtype=torch.float32)

    return aligned_audio, aligned_text
