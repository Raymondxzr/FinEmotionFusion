from torch.utils.data import Dataset, DataLoader
import torch
    
from torch.nn.utils.rnn import pad_sequence

# Wrap the preprocessed batches into a custom dataset
class EmoDataset(Dataset):
    def __init__(self, audio_preprocessed, preprocessor_text, labels):
        self.audio_preprocessed = audio_preprocessed
        self.preprocessor_text = preprocessor_text
        self.labels = labels

    def __len__(self):
        return len(self.audio_preprocessed)

    def __getitem__(self, idx):
        return self.audio_preprocessed[idx], self.preprocessor_text[idx], self.labels[idx] # Return pre-batched tensors directly


def collate_fn(batch):
    """
    Custom collate function to handle variable-length inputs by padding.
    
    Args:
        batch (list of tuples): Each tuple contains (audio_features, text_features, label).
        
    Returns:
        torch.Tensor: Padded audio features.
        dict: Padded text features with keys 'input_ids' and 'attention_mask'.
        torch.Tensor: Corresponding labels.
    """
    audio_features, text_features, labels = zip(*batch)

    # Pad audio features to the same length
    padded_audio_features = pad_sequence(
        [af.clone().detach() if isinstance(af, torch.Tensor) else torch.tensor(af, dtype=torch.float32) for af in audio_features],
        batch_first=True,
        padding_value=0
    )


    # Extract and pad text features (input_ids and attention_mask)
    input_ids = pad_sequence(
        [torch.tensor(tf["input_ids"], dtype=torch.long) for tf in text_features],
        batch_first=True,
        padding_value=0
    )
    attention_mask = pad_sequence(
        [torch.tensor(tf["attention_mask"], dtype=torch.long) for tf in text_features],
        batch_first=True,
        padding_value=0
    )

    # Combine text features into a dictionary
    padded_text_features = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # Convert labels to a tensor
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_audio_features, padded_text_features, labels

