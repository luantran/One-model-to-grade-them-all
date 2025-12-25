# Define the neural network
import numpy as np
from torch import nn
import torch

class Classifier(nn.Module):
    """Feedforward neural network for CEFR classification (5 classes)."""
    def __init__(self, embedding_dim, hidden_dim=128, num_classes=5, dropout=0.3):
        super().__init__()

        # Store config as attributes for later saving
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DeepClassifier(nn.Module):
    """
    Deeper feedforward neural network for CEFR classification.

    Architecture:
    - 3 hidden layers with decreasing dimensions (embedding_dim → 256 → 128 → 64)
    - Batch normalization after each layer for training stability
    - Dropout for regularization
    - ReLU activations
    - Final layer outputs 5 classes (A1, A2, B1, B2, C1/C2)
    """

    def __init__(self, embedding_dim, hidden_dim1=256, hidden_dim2=128,
                 hidden_dim3=64, num_classes=5, dropout_rate=0.3):
        super().__init__()

        # Layer 1: embedding_dim → hidden_dim1 (300 to 256)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layer 2: hidden_dim1 → hidden_dim2 (56 to 128)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        # Layer 3: hidden_dim2 → hidden_dim3 (128 to 64)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.bn3 = nn.BatchNorm1d(hidden_dim3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer: hidden_dim3 → num_classes (64 to 5)
        self.fc_out = nn.Linear(hidden_dim3, num_classes)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        # Output layer (no activation - CrossEntropyLoss handles softmax)
        x = self.fc_out(x)

        return x
