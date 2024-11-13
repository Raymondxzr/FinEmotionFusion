import os
import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm

class AudioPreprocessor:
    def __init__(self, audio_dir, target_sample_rate=16000, batch_size=4, feature_extractor="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        self.audio_dir = audio_dir
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
        self.target_sample_rate = target_sample_rate
        self.batch_size = batch_size
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(feature_extractor)

    def load_audio(self, file_paths):
        """
        Load multiple audio files without manual padding.
        """
        waveforms = []
        for file_path in file_paths:
            if file_path.endswith('.wav'):
                waveform, sample_rate = torchaudio.load(file_path)
            else:
                waveform, sample_rate = librosa.load(file_path, sr=self.target_sample_rate)
                waveform = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension| change here

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)

            waveform = waveform.squeeze(0).numpy()

            waveforms.append(waveform)  # Remove channel dimension for compatibility with feature_extractor
            
        return waveforms

    def preprocess_batch(self, batch_files):
        """
        Preprocess a batch of audio files to extract features.
        """
        waveforms = self.load_audio(batch_files)
        inputs = self.feature_extractor(
            waveforms, sampling_rate=self.target_sample_rate, return_tensors="pt", padding=True
        )
        return inputs.input_values  # Return the extracted features

    def preprocess(self):
        """
        Preprocess all audio files in batches.
        """
        all_features = []
        for i in tqdm(range(0, len(self.audio_files), self.batch_size), desc="Preprocessing batches"):
            batch_files = self.audio_files[i:i + self.batch_size]
            batch_features = self.preprocess_batch(batch_files)
            all_features.append(batch_features)
            print(f"batch_features's shape: {batch_features.shape}")

        print(f"all_features's length: {len(all_features)}")
        return all_features 
