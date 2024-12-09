from torch.utils.data import Dataset, DataLoader
import torch
    
from torch.nn.utils.rnn import pad_sequence

# Wrap the preprocessed batches into a custom dataset
class EmoDataset_Baseline(Dataset):
    def __init__(self, audio_preprocessed, labels):
        self.audio_preprocessed = audio_preprocessed
        self.labels = labels

    def __len__(self):
        return len(self.audio_preprocessed)

    def __getitem__(self, idx):
        return self.audio_preprocessed[idx], self.labels[idx] # Return pre-batched tensors directly


def collate_fn(batch):
    """
    Custom collate function to handle variable-length inputs by padding.
    
    Args:
        batch (list of tuples): Each tuple contains (audio_features, text_features, label).
        
    Returns:
        torch.Tensor: Padded audio features.
        torch.Tensor: Corresponding labels.
    """
    audio_features, labels = zip(*batch)
    # Pad audio features to the same length
    padded_audio_features = pad_sequence(
        [af.clone().detach() if isinstance(af, torch.Tensor) else torch.tensor(af, dtype=torch.float32) for af in audio_features],
        batch_first=True,
        padding_value=0
    )

    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_audio_features, labels


