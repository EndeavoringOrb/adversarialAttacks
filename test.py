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
from attacker import ImageAttacker

# Load models
model: ImageAttacker = torch.load("models/attacker.pt", weights_only=False)
classifierModel: CNN = torch.load("models/classifier.pt", weights_only=False)
criterion = nn.CrossEntropyLoss()

# Load and preprocess the data
splits = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
df = pd.read_parquet("mnist/" + splits["test"])
tensor_image = transforms.PILToTensor()

dataset = []
for idx, data in tqdm(df.iterrows(), desc="Loading data", total=len(df)):
    dataset.append(
        (
            tensor_image(Image.open(io.BytesIO(data["image"]["bytes"]))).to(torch.float32)
            * (1.0 / 255.0),
            torch.tensor(data["label"], dtype=torch.int64),
        )
    )

imgMax = dataset[0][0].max().item()
imgMin = dataset[0][0].min().item()
imgArr = ((dataset[0][0].cpu().detach().squeeze().numpy() + imgMin) * (255.0 / (imgMax - imgMin))).astype(np.uint8).clip(0, 255)
Image.fromarray(imgArr).save("img.png")
imgMax = model.weights.max().item()
imgMin = model.weights.min().item()
imgArr = ((model.weights.cpu().detach().squeeze().numpy() + imgMin) * (255.0 / (imgMax - imgMin))).astype(np.uint8).clip(0, 255)
Image.fromarray(imgArr).save("weights.png")
newIm = model(dataset[0][0])
imgMax = newIm.max().item()
imgMin = newIm.min().item()
imgArr = ((newIm.cpu().detach().squeeze().numpy() + imgMin) * (255.0 / (imgMax - imgMin))).astype(np.uint8).clip(0, 255)
Image.fromarray(imgArr).save("img_and_weights.png")

totalLoss = 0
totalAccuracy = 0
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset, desc="Evaluating Classifier")):
        outputs = classifierModel(data[0])
        totalLoss += criterion(outputs, data[1].unsqueeze(0)).item()
        topPred = outputs.argmax()
        totalAccuracy += (topPred == data[1]).item()

totalLoss /= float(len(dataset))
totalAccuracy /= float(len(dataset))

print(f"Classification Loss: {totalLoss:.2e}")
print(f"Classification Accuracy: {100*totalAccuracy:.2f}%")

totalLoss = 0
totalAccuracy = 0
totalPixelDist = model.weights.abs().mean().item() * 255.0
with torch.no_grad():
    for idx, data in enumerate(tqdm(dataset, desc="Evaluating Attacker")):
        newInput = model(data[0])
        outputs = classifierModel(newInput)
        totalLoss += criterion(outputs, data[1].unsqueeze(0)).item()
        topPred = outputs.argmax()
        totalAccuracy += (topPred == data[1]).item()

totalLoss /= float(len(dataset))
totalAccuracy /= float(len(dataset))

print(f"Classification Loss: {totalLoss:.2e}")
print(f"Classification Accuracy: {100*totalAccuracy:.2f}%")
print(f"Avg. Pixel Dist: {totalPixelDist:.2f}")