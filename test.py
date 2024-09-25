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

from data import loadDataset
from models import CNN, ImageAttacker

# Ask which models to test
modelType = input("Test better models? [y/n]: ").lower() == "y"

# Load models
model: ImageAttacker = torch.load(f"models/{'betterAttacker' if modelType else 'attacker'}.pt", weights_only=False)
classifierModel: CNN = torch.load(f"models/{'betterClassifier' if modelType else 'classifier'}.pt", weights_only=False)

# Init loss function
criterion = nn.CrossEntropyLoss()

# Load data
dataset = loadDataset("test")

# Visualize image, weights, image+weights
imgMax = dataset[0][0].max().item()
imgMin = dataset[0][0].min().item()
imgArr = ((dataset[0][0].cpu().detach().squeeze().numpy() + imgMin) * (255.0 / (imgMax - imgMin))).astype(np.uint8).clip(0, 255)
Image.fromarray(imgArr).save("images/img.png")
imgMax = model.weights.max().item()
imgMin = model.weights.min().item()
imgArr = ((model.weights.cpu().detach().squeeze().numpy() + imgMin) * (255.0 / (imgMax - imgMin))).astype(np.uint8).clip(0, 255)
Image.fromarray(imgArr).save("images/weights.png")
newIm = model(dataset[0][0])
imgMax = newIm.max().item()
imgMin = newIm.min().item()
imgArr = ((newIm.cpu().detach().squeeze().numpy() + imgMin) * (255.0 / (imgMax - imgMin))).astype(np.uint8).clip(0, 255)
Image.fromarray(imgArr).save("images/img_and_weights.png")

# Test classifier without attackss
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

# Test classifier with attacks
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