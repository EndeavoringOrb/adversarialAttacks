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

from classifier import CNN
from attacker import ImageAttacker

epochs = 1000
batchSize = 4096
maxDist = 10


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
    classifierModel = CNN()
    attackerModel = ImageAttacker()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimClassifier = optim.Adam(classifierModel.parameters(), lr=0.001)
    optimAttacker = optim.Adam(attackerModel.parameters(), lr=0.001)
    print(
        f"Classifier Model has {sum([p.numel() for p in classifierModel.parameters()]):,} parameters"
    )
    print(
        f"Attacker Model has {sum([p.numel() for p in attackerModel.parameters()]):,} parameters"
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
                attackerModel.preCompute(maxDist)
                newInputs = attackerModel(torch.stack(inputs, 0))
                outputs = classifierModel(newInputs)
                atkLoss = -criterion(outputs, torch.stack(labels, 0))

                optimAttacker.zero_grad()
                atkLoss.backward()
                optimAttacker.step()

                attackerModel.preCompute(maxDist)
                with torch.no_grad():
                    newInputs = attackerModel(torch.stack(inputs, 0))
                outputs = classifierModel(newInputs)
                classifierLoss = criterion(outputs, torch.stack(labels, 0))

                optimClassifier.zero_grad()
                classifierLoss.backward()
                optimClassifier.step()

                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{idx + 1:5d}/{len(dataset)}], Atk loss: {atkLoss:.3e}, Classifier Loss: {classifierLoss:.3e}"
                )

                inputs = []
                labels = []

        # Save the models
        torch.save(classifierModel, f"models/betterClassifier.pt")
        torch.save(attackerModel, f"models/betterAttacker.pt")

    print("Finished Training")


if __name__ == "__main__":
    main()
