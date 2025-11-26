import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):

    def __init__(self, num_classes=3, n_mels=64):
        super(AudioCNN, self).__init__()

        # input (batch, 1, n_mels, time)

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2,2)
        #self.dropout1 = nn.Dropout(0.25)
        self.dropout1 = nn.Dropout2d(0.025) # all 0.1

        # Conv block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,5), padding=(1,2))   # kernel_size=3, padding=1   wider now to see more time
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2,2)
        #self.dropout2 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout2d(0.05)

        # Conv block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2,1)
        #self.dropout3 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout2d(0.05)

        # Conv block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(1,2)
        #self.dropout4 = nn.Dropout(0.25)
        self.dropout4 = nn.Dropout2d(0.10)

        # global avg pooling
        self.GAP = nn.AdaptiveAvgPool2d((1,1))

        # fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Global Average Pooling
        x = self.GAP(x)
        x = torch.flatten(x, 1)     # x = x.view(x.size(0), -1)
        
        # Fully Connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x