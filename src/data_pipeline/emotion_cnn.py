import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.fc_input_size = 64 # default value
        
        # # First convolutional layer with 256 filters, kernel size 5, stride 1
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, stride=1)
        
        # # Second convolutional layer with 128 filters, kernel size 5, stride 1
        # self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1)
        
        # # Max pooling layer with window size 8
        # self.pool = nn.MaxPool1d(kernel_size=8)
        
        # # Third convolutional layer (output channels to be determined by input size)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1)
        
        
        # Convolutional layers with BatchNorm and ReLU
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, stride=1, padding=2)
        # self.bn1 = nn.BatchNorm1d(256)
        
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2)
        # self.bn2 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(kernel_size=8)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2)
        # self.bn3 = nn.BatchNorm1d(64)
        
        # Dropout layer with dropout rate 0.2
        self.dropout = nn.Dropout(0.35)
        
        # Fully connected layer
        #1408, 2176
        self.fc = nn.Linear(1408, num_classes)  # Adjust in_features based on flatten size after pooling
        
    def forward(self, x):
        # Pass through first conv layer
        x = F.relu(self.conv1(x))
        
        # Pass through second conv layer
        x = F.relu(self.conv2(x))
        
        # Max pooling
        x = self.pool(x)
        
        # Pass through third conv layer
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        self.fc_input_size = x.shape[-1]
        
        # Fully connected layer
        x = self.fc(x)
        
        # Dropout
        x = self.dropout(x)
        
        # print(f"after dropout {x.shape}")
        
        # Softmax layer
        x = F.log_softmax(x, dim=1)
        
        return x
    

