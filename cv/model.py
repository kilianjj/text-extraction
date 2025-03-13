"""
PyTorch model for classifying characters.
Includes methods for loading model, predicting characters, etc.
"""

import numpy as np
import torch
import torch.nn as nn
import config

class Conv(nn.Module):

    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 62)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class Classifier:

    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = Conv()
        self.model.load_state_dict(torch.load(config.model_path))
        self.model.eval()

    def predict(self, img):
        img = torch.tensor(np.array([img]), dtype=torch.float32).unsqueeze(0)
        # if np.max(img) > 1:
        #     img /= 255
        img /= 255
        img = (img - 0.5) / 0.5
        return self.model(img).argmax().item()
