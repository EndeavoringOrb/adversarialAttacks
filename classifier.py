import torch
import random
import torch.nn as nn
import torch.optim as optim

from models import CNN
from data import loadDataset

epochs = 1000
batchSize = 8192


def main():
    # Load data
    dataset = loadDataset("train")

    # Init network
    print("Initializing model")
    model = CNN()

    # Init loss function and optimizer
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
