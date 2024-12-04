import torch
import json
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from src.audio_pipeline.audio_model import Wav2VecWrapper
from src.text_pipeline.text_model import FinBERTWrapper
from src.fusion.alignment import CrossAttentionFusion
from src.fusion.classifier import FusionClassifierHead
from src.fusion.fusion import MultimodalPipeline
from src.data_pipeline.dataloader import *
from src.audio_pipeline.preprocess_audio import *
from src.text_pipeline.preprocess_text import *
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Two modality
wav2vec_model = Wav2VecWrapper(pretrained_model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition").to(device)
finbert_model = FinBERTWrapper(pretrained_model_name="yiyanghkust/finbert-tone").to(device)


# Cross-Attention Fusion: TODO
audio_dim = 1024  # Wav2Vec feature dimension
text_dim = 768   # FinBERT feature dimension
fused_dim = 512  # Dimension after fusion
hidden_dim = 256 # Dimension of hidden layer in classifier
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
        correct_predictions = 0
        total_samples = 0
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
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, dim=1)  # Get predicted classes
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        # Calculate average loss and accuracy for this epoch
        avg_epoch_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        train_losses.append(avg_epoch_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.4f}")


audio_dir_path = 'data/data_1000'
labels_dir_path = 'data/labels.json'
transcripts_path = "data/transcripts.json"

with open(labels_dir_path, "r") as json_file:
    labels_dict = json.load(json_file)


preprocessor_text = TextPreprocessor(transcripts_path)
preprocessor_audio = AudioPreprocessor(audio_dir_path, preprocessor_text, labels_dict)
processed_audio, processed_text, labels = preprocessor_audio.preprocess()
# Create dataset and dataloader
dataset = EmoDataset(processed_audio, processed_text, labels)
subset_dataset = torch.utils.data.Subset(dataset, indices=range(4))

dataloader = DataLoader(subset_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

print("Training started...")
train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)
print("Training completed.")


