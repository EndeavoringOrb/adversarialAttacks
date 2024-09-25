import torch
import random
import torch.nn as nn
import torch.optim as optim

from models import CNN, ImageAttacker
from data import loadDataset

epochs = 1000
batchSize = 16384
maxDist = 2
distType = 2


def main():
    # Ask which model to test
    modelType = input("Train against better classifier? [y/n]: ").lower() == "y"

    # Load data
    dataset = loadDataset("train")

    # Init network
    print("Initializing model")
    model = ImageAttacker()
    classifierModel: CNN = torch.load(f"models/{'betterClassifier' if modelType else 'classifier'}.pt", weights_only=False)
    for p in classifierModel.parameters():  # Freeze classifier model
        p.requires_grad_(False)

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
                model.preCompute(maxDist, distType)
                newInputs = model(torch.stack(inputs, 0))
                outputs = classifierModel(newInputs)
                loss = -criterion(outputs, torch.stack(labels, 0)) # Negative loss

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
