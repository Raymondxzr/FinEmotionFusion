import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class CrossAttentionFusion(nn.Module):
    def __init__(self, audio_dim, text_dim, fused_dim):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, fused_dim) # ()
        self.text_proj = nn.Linear(text_dim, fused_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=4)

    def forward(self, audio_features, text_features):
        # aligned_audio, aligned_text = align_features(
        #     audio_features.cpu().detach().numpy(), 
        #     text_features.cpu().detach().numpy()
        # )
        # aligned_audio = aligned_audio.to(audio_features.device)
        # aligned_text = aligned_text.to(text_features.device)

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



# def align_features(audio_features, text_features):
#     """
#     Perform DTW alignment between audio and text features and return fused features.

#     Args:
#         audio_features (torch.Tensor): Audio features of shape (batch_size, audio_seq_len, audio_feature_dim).
#         text_features (torch.Tensor): Text features of shape (batch_size, text_seq_len, text_feature_dim).

#     Returns:
#         torch.Tensor, torch.Tensor: Aligned audio and text features for each batch.
#     """
#     batch_size = audio_features.size(0)
#     aligned_audio_list = []
#     aligned_text_list = []

#     for batch_idx in range(batch_size):
#         # Extract features for the current batch
#         audio_batch = audio_features[batch_idx]  # Shape: (audio_seq_len, audio_feature_dim)
#         text_batch = text_features[batch_idx]   # Shape: (text_seq_len, text_feature_dim)

#         # Compute pairwise cosine distance
#         distance_matrix = cdist(audio_batch.numpy(), text_batch.numpy(), metric="cosine")

#         # Perform DTW alignment using linear_sum_assignment
#         row_indices, col_indices = linear_sum_assignment(distance_matrix)

#         # Align features
#         aligned_audio = torch.tensor(audio_batch[row_indices], dtype=torch.float32)
#         aligned_text = torch.tensor(text_batch[col_indices], dtype=torch.float32)

#         aligned_audio_list.append(aligned_audio)
#         aligned_text_list.append(aligned_text)

#     # Stack aligned features into batch tensors
#     aligned_audio = torch.stack(aligned_audio_list, dim=0)
#     aligned_text = torch.stack(aligned_text_list, dim=0)

#     return aligned_audio, aligned_text

