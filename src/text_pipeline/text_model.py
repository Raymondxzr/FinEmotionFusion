# src/text_pipeline/text_model.py
import torch
from transformers import BertModel

class FinBERTWrapper(torch.nn.Module):
    def __init__(self, pretrained_model_name="yiyanghkust/finbert-tone"):
        super().__init__()
        self.finbert = BertModel.from_pretrained(pretrained_model_name)
        self.finbert.gradient_checkpointing_enable()  # Enable gradient checkpointing

    def forward(self, text_inputs):
        outputs = self.finbert(**text_inputs).last_hidden_state
        return outputs  # Shape: [batch_size, seq_len, feature_dim]

