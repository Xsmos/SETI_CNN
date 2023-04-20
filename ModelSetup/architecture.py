import torch
import torch.nn as nn
import numpy as np
import sys, os, time
import optuna
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, activation, dr1, dr2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.dropout1 = nn.Dropout2d(dr1)
        self.dropout2 = nn.Dropout2d(dr2)
        self.fc1 = nn.Linear(3350528, 128)
        self.fc2 = nn.Linear(128,2)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x,1)
        x = self.activation(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.sigmoid(x)
        return output
    
