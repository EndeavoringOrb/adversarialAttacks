import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import io
import random
from tqdm import tqdm
import numpy as np
import torchvision.transforms.v2 as transforms
import math

epochs = 100
batchSize = 8192


# Define the CNN architecture
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


def main():
    # Load and preprocess the data
    splits = {
        "train": "train-00000-of-00001.parquet",
        "test": "test-00000-of-00001.parquet",
    }
    df = pd.read_parquet("mnist/" + splits["train"])
    tensor_image = transforms.PILToTensor()

    dataset = []
    for idx, data in tqdm(df.iterrows(), desc="Loading data", total=len(df)):
        dataset.append(
            (
                tensor_image(Image.open(io.BytesIO(data["image"]["bytes"])))
                * (1.0 / 255.0),
                torch.tensor(data["label"], dtype=torch.int64),
            )
        )

    # Initialize the network
    print("Initializing model")
    model = CNN()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Model has {sum([p.numel() for p in model.parameters()]):,} parameters")

    # Init tracker stuff
    batchesPerEpoch = math.ceil(
        (len(dataset) / batchSize) + (1 if len(dataset) % batchSize > 0 else 0)
    )

    # Train the network
    print("Training")
    for epoch in range(epochs):  # loop over the dataset multiple times
        random.shuffle(dataset)
        inputs = []
        labels = []
        for idx, data in enumerate(dataset):
            inputs.append(data[0])
            labels.append(data[1])

            if len(inputs) == batchSize or idx == len(dataset) - 1:
                outputs = model(torch.stack(inputs, 0))
                loss = criterion(outputs, torch.stack(labels, 0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{idx + 1:5d}/{len(dataset)}] loss: {loss:.3e}"
                )

                inputs = []
                labels = []

        # Save the model
        torch.save(model, f"models/classifier.pt")

    print("Finished Training")


if __name__ == "__main__":
    main()
