from torch.utils.data import Dataset, DataLoader

# Wrap the preprocessed batches into a custom dataset
class BatchedDataset(Dataset):
    def __init__(self, audio_preprocessed_batches):
        self.audio_batches = audio_preprocessed_batches  # List of pre-batched tensors

    def __len__(self):
        return len(self.audio_batches)

    def __getitem__(self, idx):
        return self.audio_batches[idx]  # Return pre-batched tensors directly