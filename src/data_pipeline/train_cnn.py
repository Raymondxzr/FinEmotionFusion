
import torch
import torch.nn as nn
import torch.nn.functional as F
from emotion_cnn import EmotionCNN
from torch.utils.data import DataLoader


num_classes = 8  # Replace with actual number of classes
model = EmotionCNN(num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

device =  "cuda" if torch.cuda.is_available() else "cpu" 
model = model.to(device)

data_path = '../../data/audio/audio_features_tensor.pt'  # Adjust the path as needed
labels_path = '../../data/audio/audio_labels_tensor.pt'  

# Load the tensor
features_tensor = torch.load(data_path)
labels_tensor = torch.load(labels_path)

# select the emotions represented in https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio?resource=download
emotion_labels = labels_tensor[:, 2] - 1

# Define training loop parameters
num_epochs = 60  # Set to the number of epochs you wish to train
batch_size = 16  # Set the batch size
train_losses, accuracies = [], []
train_loader = DataLoader(list(zip(features_tensor, emotion_labels)), batch_size=batch_size, shuffle=True)

# Training loop
# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for features, labels in train_loader:
        # Reshape features to [batch_size, 1, 180]
        features = features.view(-1, 1, 180).to(device)
        labels = labels.to(device).long()  # Ensure labels are LongTensor
        
        # Forward pass
        outputs = model(features)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, dim=1)  # Get predicted classes
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
    
    # Calculate average loss and accuracy for this epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    train_losses.append(avg_epoch_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

print("Training completed!")

model_save_path = '../../data/audio/emotion_cnn_model.pth'
torch.save(model.state_dict(), model_save_path)

"""
TODO:
- 5 fold validation (choose the best epoch and batch size)
- predict on business data, which ever one we find (then we also need to preprocess)
"""
