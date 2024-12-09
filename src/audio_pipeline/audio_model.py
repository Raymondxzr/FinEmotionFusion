# src/audio_pipeline/audio_model.py
import torch
from transformers import Wav2Vec2Model
import torch.nn as nn

class Wav2VecWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.wav2vec.gradient_checkpointing_enable()  # Enable gradient checkpointing

    def forward(self, audio_inputs):
        outputs = self.wav2vec(audio_inputs).last_hidden_state
        return outputs  # Shape: [batch_size, time_steps, feature_dim]


class AudioEmotionModel(nn.Module):
    def __init__(self, wav2vec_model, num_classes):
        super().__init__()
        self.wav2vec = wav2vec_model
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec.wav2vec.config.hidden_size, 256),  # Intermediate hidden layer
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Output layer for emotion classes
        )
    
    def forward(self, audio_inputs):
        # Get Wav2Vec embeddings
        features = self.wav2vec(audio_inputs).mean(dim=1)  # Mean pooling
        logits = self.classifier(features)
        return logits