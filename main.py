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
import matplotlib.pyplot as plt
import numpy as np


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
    train_losses, accuracies = [], []
    all_predictions, all_labels = [], []
    
    model.train()
    for epoch in tqdm(range(num_epochs)):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        epoch_predictions, epoch_labels = [], []
        
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
            # TODO: why is outputs len 4?

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Calculate accuracy and store predictions/labels
            _, predicted = torch.max(outputs, dim=1)  # Get predicted classes
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # print(outputs.shape)

            epoch_predictions.extend(predicted.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
        
        # Append for full dataset tracking
        all_predictions.extend(epoch_predictions)
        all_labels.extend(epoch_labels)

        # Calculate average loss and accuracy for this epoch
        avg_epoch_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        train_losses.append(avg_epoch_loss)
        accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save and plot results
    save_and_plot_results(all_predictions, all_labels, train_losses, accuracies)

def save_and_plot_results(predictions, labels, train_losses, accuracies):
    # Save arrays
    np.save("predicted_labels.npy", predictions)
    np.save("actual_labels.npy", labels)

    # Calculate and plot distribution of predicted labels
    unique, counts = np.unique(predictions, return_counts=True)
    label_distribution = dict(zip(unique, counts))
    print("Predicted Label Distribution:", label_distribution)

    # Plot distribution
    plt.figure(figsize=(10, 5))
    plt.bar(label_distribution.keys(), label_distribution.values(), align="center")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Distribution of Predicted Labels")
    plt.grid(axis='y')
    plt.show()

    # Plot Loss and Accuracy
    save_metrics(train_losses, accuracies)

def save_metrics(train_losses, accuracies, save_dir='results/'):
    epochs = range(1, len(train_losses) + 1)

    # Plot and save training loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()

    # Plot and save training accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, "training_accuracy.png"))
    plt.close()




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
# subset_dataset = torch.utils.data.Subset(dataset, indices=range(4))

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

print("Training started...")
train_model(model, dataloader, criterion, optimizer, device, num_epochs=10)
print("Training completed.")


