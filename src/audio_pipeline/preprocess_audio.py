import os
import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm

class AudioPreprocessor:
    def __init__(self, audio_dir, labels_dict, target_sample_rate=16000, feature_extractor="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """
        Initialize the AudioPreprocessor with directory and label information.
        """
        self.audio_dir = audio_dir
        self.labels_dict = labels_dict
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
        self.target_sample_rate = target_sample_rate
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor)

    def load_audio(self, file_path):
        """
        Load and preprocess a single audio file.
        """
        if file_path.endswith('.wav'):
            waveform, sample_rate = torchaudio.load(file_path)
        else:
            waveform, sample_rate = librosa.load(file_path, sr=self.target_sample_rate)
            waveform = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0).numpy()

    def preprocess(self):
        """
        Preprocess all audio files sequentially, returning features and labels as lists.
        """
        all_features = []
        all_labels = []

        for file_path in tqdm(self.audio_files, desc="Preprocessing audio files"):
            # Load audio
            waveform = self.load_audio(file_path)

            # Extract features (padding is deferred)
            features = self.feature_extractor(
                waveform, sampling_rate=self.target_sample_rate, return_tensors="pt", padding=False
            ).input_values

            # Get label
            file_name = os.path.basename(file_path)
            label = self.labels_dict[file_name]

            # Append to lists
            all_features.append(features.squeeze(0))  # Append individual features (remove batch dim)
            all_labels.append(label)  # Append individual label

        return all_features, all_labels
