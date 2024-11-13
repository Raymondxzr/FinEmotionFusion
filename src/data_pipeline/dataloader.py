from torch.utils.data import Dataset, DataLoader
import torch
    
from torch.nn.utils.rnn import pad_sequence

# Wrap the preprocessed batches into a custom dataset
class BatchedDataset(Dataset):
    def __init__(self, audio_preprocessed_batches, labels):
        self.audio_batches = audio_preprocessed_batches
        self.labels = labels

    def __len__(self):
        return len(self.audio_batches)

    def __getitem__(self, idx):
        return self.audio_batches[idx], self.labels[idx] # Return pre-batched tensors directly


def collate_fn(batch):
    """
    Custom collate function to handle variable-length inputs by padding.
    
    Args:
        batch (list of tuples): Each tuple contains (feature, label).
        
    Returns:
        torch.Tensor: Padded features.
        torch.Tensor: Corresponding labels.
    """
    features, labels = zip(*batch)

    # Pad features to the same length
    padded_features = pad_sequence(features, batch_first=True, padding_value=0)

    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_features, labels
