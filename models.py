import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(F.tanh(self.conv1(x)))
        x = self.pool(F.tanh(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the architecture
class ImageAttacker(nn.Module):
    def __init__(self):
        super(ImageAttacker, self).__init__()
        data = torch.normal(0, 0.02, (28, 28))
        self.weights = nn.Parameter(data)

    @torch.no_grad()
    def preCompute(self, maxDist, distType):
        if distType == 1:
            self.weights[self.weights < -maxDist] = -maxDist
            self.weights[self.weights > maxDist] = maxDist
        elif distType == 2:
            weiNorm = self.weights.norm()
            if weiNorm > maxDist:
                normalizeVal = maxDist / weiNorm
                self.weights *= normalizeVal

    def forward(self, x):
        return x + self.weights
