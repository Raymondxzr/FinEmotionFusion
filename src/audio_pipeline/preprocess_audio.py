import os
import torch
import torchaudio
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from tqdm import tqdm

class AudioPreprocessor:
    def __init__(self, audio_dir, text_preprocessor, labels_dict, target_sample_rate=16000, feature_extractor="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        """
        Initialize the AudioPreprocessor with directory and label information.
        """
        self.audio_dir = audio_dir
        self.text_preprocessor = text_preprocessor
        self.transcripts = text_preprocessor.load_transcripts()

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
        Preprocess all audio files, aligning audio features, tokenized text, and labels.
        """
        all_audio = []
        all_text = []
        all_label = []
        
        for file_path in tqdm(self.audio_files, desc="Preprocessing files"):
            # Load audio
            waveform = self.load_audio(file_path)

            # Extract audio features
            audio_features = self.feature_extractor(
                waveform, sampling_rate=self.target_sample_rate, return_tensors="pt", padding=False
            ).input_values.squeeze(0)  # Remove batch dimension

            # Get file name
            file_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            # Get text transcription and tokenize
            if file_name_without_ext in self.transcripts:
                text = self.transcripts[file_name_without_ext]
                tokenized_text = self.text_preprocessor.tokenize_text(text)
                tokenized_text = {
                    "input_ids": tokenized_text["input_ids"].squeeze(0).tolist(),
                    "attention_mask": tokenized_text["attention_mask"].squeeze(0).tolist(),
                }
            else:
                raise KeyError(f"No transcription found for {file_name_without_ext}")

            # Get label
            if file_name_without_ext in self.labels_dict:
                label = self.labels_dict[file_name_without_ext]
            else:
                raise KeyError(f"No label found for {file_name_without_ext}")

            # Append data
            all_audio.append(audio_features.squeeze(0))
            all_text.append(tokenized_text)
            all_label.append(label)
            # processed_data.append({
            #     "audio_features": audio_features.tolist(),
            #     "tokenized_text": tokenized_text,
            #     "label": label
            # })

        return all_audio, all_text, all_label

