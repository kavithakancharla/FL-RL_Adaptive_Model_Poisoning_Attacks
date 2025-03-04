from fl.nn_models import *
import torch
import torch.nn as nn
import torch.nn.functional as F

#  define CIFAR-100 NN Model
class CIFAR100_NN(nn.Module):
    def init(self):
        super(CIFAR100_NN, self).init()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # CIFAR-100 images have 3 channels (RGB)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 256)  # Corrected input size for CIFAR-100
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 100)  # CIFAR-100 has 100 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the image
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x