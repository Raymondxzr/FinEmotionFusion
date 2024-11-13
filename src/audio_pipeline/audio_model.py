# src/audio_pipeline/audio_model.py
import torch
from transformers import Wav2Vec2Model

class Wav2VecWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        self.wav2vec.gradient_checkpointing_enable()  # Enable gradient checkpointing


    def forward(self, audio_inputs):
        outputs = self.wav2vec(audio_inputs).last_hidden_state
        return outputs  # Shape: [batch_size, time_steps, feature_dim]
