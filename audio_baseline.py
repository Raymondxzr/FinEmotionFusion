import torch
import json
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from src.audio_pipeline.audio_model import Wav2VecWrapper,AudioEmotionModel
from src.data_pipeline.dataloader_baseline import *
from src.audio_pipeline.preprocess_audio_baseline import *
import matplotlib.pyplot as plt
import numpy as np
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_classes = 8
wav2vec_model = Wav2VecWrapper(pretrained_model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition").to(device)
model = AudioEmotionModel(wav2vec_model, num_classes).to(device)


audio_dim = 1024  # Wav2Vec feature dimension
hidden_dim = 256 # Dimension of hidden layer in classifier

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.classifier.parameters(), lr=0.001)

for param in model.wav2vec.parameters():
    param.requires_grad = False
    
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
            audio_inputs, labels = batch
            audio_inputs, labels = (
                audio_inputs.to(device),
                labels.to(device),
            )

            # Forward pass
            optimizer.zero_grad()
            outputs = model(audio_inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Calculate accuracy
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
with open(labels_dir_path, "r") as json_file:
    labels_dict = json.load(json_file)
    
preprocessor_audio = AudioPreprocessor_Baseline(audio_dir_path, labels_dict)
processed_audio, labels = preprocessor_audio.preprocess()
# Create dataset and dataloader
dataset = EmoDataset_Baseline(processed_audio, labels)
subset_dataset = torch.utils.data.Subset(dataset, indices=range(24))

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn,num_workers=4)

print("Training started...")
train_model(model, dataloader, criterion, optimizer, device, num_epochs=3)
print("Training completed.")


