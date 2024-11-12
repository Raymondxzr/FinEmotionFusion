# src/audio_pipeline/audio_model.py
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

class Wav2VecWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name)
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model_name)

    def forward(self, audio_inputs):
        outputs = self.wav2vec(audio_inputs).last_hidden_state
        return outputs  # Shape: [batch_size, time_steps, feature_dim]
