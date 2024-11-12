# src/models/multimodal_pipeline.py
import torch.nn as nn

class MultimodalPipeline(nn.Module):
    def __init__(self, wav2vec_model, finbert_model, cross_attention, classifier):
        super().__init__()
        self.wav2vec_model = wav2vec_model
        self.finbert_model = finbert_model
        self.cross_attention = cross_attention
        self.classifier = classifier

    def forward(self, audio_inputs, text_inputs):
        # Extract features
        audio_features = self.wav2vec_model(audio_inputs)
        text_features = self.finbert_model(text_inputs)
        
        # Fuse features
        fused_features = self.cross_attention(audio_features, text_features)

        # Classification
        logits = self.classifier(fused_features.mean(dim=1))  # Pool over time
        return logits
