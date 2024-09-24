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
from classifier import CNN

epochs = 1000000
batchSize = 2**14
maxDist = 10.0 / 255.0
distType = 1


# Define the CNN architecture
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


def main():
    # Load and preprocess the data
    print(f"Batch Size: {batchSize:,}")
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
    model = ImageAttacker()
    classifierModel: CNN = torch.load("models/classifier.pt", weights_only=False)
    for p in classifierModel.parameters():
        p.requires_grad_(False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"Model has {sum([p.numel() for p in model.parameters()]):,} parameters")

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
                model.preCompute(maxDist, distType)
                newInputs = model(torch.stack(inputs, 0))
                outputs = classifierModel(newInputs)
                loss = -criterion(outputs, torch.stack(labels, 0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{idx + 1:5d}/{len(dataset)}] loss: {loss:.3e}"
                )

                inputs = []
                labels = []

        # Save the model
        model.preCompute(maxDist, distType)
        torch.save(model, f"models/attacker.pt")

    print("Finished Training")


if __name__ == "__main__":
    main()
