import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from src.audio_pipeline.audio_model import Wav2VecWrapper
from src.text_pipeline.text_model import FinBERTWrapper
from src.fusion.alignment import CrossAttentionFusion
from src.fusion.classifier import FusionClassifierHead
from src.fusion.fusion import MultimodalPipeline

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Two modality
wav2vec_model = Wav2VecWrapper(pretrained_model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition").to(device)
finbert_model = FinBERTWrapper(pretrained_model_name="yiyanghkust/finbert-tone").to(device)

# Cross-Attention Fusion: TODO
audio_dim = 768  # Wav2Vec feature dimension
text_dim = 768   # FinBERT feature dimension
fused_dim = 768  # Dimension after fusion
hidden_dim = 128 # Dimension of hidden layer in classifier
cross_attention_fusion = CrossAttentionFusion(audio_dim, text_dim, fused_dim).to(device)

# Classifier Head
num_classes = 8  # Number of emotion classes
classifier_head = FusionClassifierHead(input_dim=fused_dim, hidden_dim=hidden_dim, num_classes=num_classes).to(device)

# Multimodal Pipeline
model = MultimodalPipeline(
    wav2vec_model=wav2vec_model,
    finbert_model=finbert_model,
    cross_attention=cross_attention_fusion,
    classifier=classifier_head
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        for batch in dataloader:
            # Unpack batch
            audio_inputs, text_inputs, labels = batch
            audio_inputs, text_inputs, labels = (
                audio_inputs.to(device),
                {k: v.to(device) for k, v in text_inputs.items()},  # FinBERT input as dict
                labels.to(device),
            )

            # Forward pass
            optimizer.zero_grad()
            outputs = model(audio_inputs, text_inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Log epoch progress
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Train the model: TODO
# Bobby: write the dataloader logic, write preprocess with audio in audio_pipeline/preprocess_audio.py
# During training: audio_inputs, text_inputs, labels = batch
# Remember to tokenize + preprocess the text, and do whatever preprocess with audio
# Sherry: write the tokenize + preprocess in text_pipeline/preprocess_text.py so that Bobby can use it here.
dataloader = ...
print("Training started...")
# train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)
print("Training completed.")
