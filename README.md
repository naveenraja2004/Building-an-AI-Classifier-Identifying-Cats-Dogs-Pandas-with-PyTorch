# Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch
## Identifying Cats, Dogs & Pandas with PyTorch

## Overview

This project implements an image classification model using PyTorch to identify Cats, Dogs, and Pandas.
It uses a Convolutional Neural Network (CNN) trained on a custom dataset, and can classify individual images or run predictions in real-time using a webcam.

## Objectives

Build a CNN-based classifier for 3 animal categories
Train, validate, and test the model using PyTorch
Preprocess image datasets using torchvision.transforms
Predict classes for new images
(Optional) Real-time webcam-based animal detection
## Project Structure

Identifying-Cats-Dogs-Pandas-with-PyTorch/
│── ws_05.ipynb                   #Main notebook
│── dataset/
│      ├── train/
│      │     ├── cats/
│      │     ├── dogs/
│      │     └── pandas/
│      ├── val/
│      └── test/
│── models/
│      └── animal_classifier.pth  # Saved model
│── utils/
│      └── transforms.py
│── README.md

## Requirements
Install dependencies:

pip install torch torchvision numpy matplotlib pillow opencv-python

## Check GPU availability:

import torch
torch.cuda.is_available()


## Dataset Information

Dataset is organized into subdirectories:

dataset/
├── train/
├── val/
└── test/

Each folder contains:

/cats
/dogs
/pandas
Images are automatically labeled using folder names.


## Image Transformations

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

## Model Architecture (CNN)

import torch.nn as nn
import torch.nn.functional as F

class AnimalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

## Training Loop


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.4f}")

### Prediction on Single Image

from PIL import Image

img = Image.open("test/panda1.jpg")
img_t = transform(img).unsqueeze(0)

model.eval()
with torch.no_grad():
    pred = model(img_t)
    class_id = torch.argmax(pred).item()

classes = ["Cat", "Dog", "Panda"]
print("Prediction:", classes[class_id])


## Real-Time Webcam Detection (Optional)

import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transform(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        pred = model(img_t)
        class_name = classes[torch.argmax(pred)]

    cv2.putText(frame, class_name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Animal Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


### Results

Training Loss and Accuracy Visualization
Validation accuracy
Sample predictions for Cats, Dogs & Pandas
Screenshot examples of real-time webcam detection
ssification
