import torch
import random
import torch.nn as nn
import torch.optim as optim

from data import loadDataset
from models import CNN, ImageAttacker

epochs = 1000
batchSize = 4096
maxDist = 10.0 / 255.0
distType = 1


def main():
    # Load data
    dataset = loadDataset("train")

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
                attackerModel.preCompute(maxDist, distType)
                newInputs = attackerModel(torch.stack(inputs, 0))
                outputs = classifierModel(newInputs)
                atkLoss = -criterion(outputs, torch.stack(labels, 0))

                optimAttacker.zero_grad()
                atkLoss.backward()
                optimAttacker.step()

                attackerModel.preCompute(maxDist, distType)
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
