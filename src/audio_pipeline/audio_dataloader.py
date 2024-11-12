import os
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, audio_dir, feature_extractor, target_sample_rate=16000):
        self.audio_dir = audio_dir
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
        self.feature_extractor = feature_extractor
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load the audio file
        file_path = os.path.join(self.audio_dir, self.audio_files[idx])

        # Load .wav files with torchaudio, .mp3 files with librosa
        if file_path.endswith('.wav'):
            waveform, sample_rate = torchaudio.load(file_path)
        else:
            waveform, sample_rate = librosa.load(file_path, sr=self.target_sample_rate)
            waveform = torch.tensor(waveform).unsqueeze(0)  # Convert to tensor and add channel dimension

        
        # Ensure waveform is in float format and resample if needed
        waveform = waveform.float()
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        
        # Use feature extractor to preprocess
        inputs = self.feature_extractor(waveform, sampling_rate=self.target_sample_rate, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0)  # Return the input values for batch processing

def collate_fn(batch):
    # Find the max length in the batch
    max_length = max(item.size(1) for item in batch)
    
    # Pad each item in the batch to the max length
    padded_batch = [torch.nn.functional.pad(item, (0, max_length - item.size(1))) for item in batch]
    return torch.stack(padded_batch)