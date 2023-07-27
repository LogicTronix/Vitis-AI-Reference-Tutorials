import cv2
import torch

import numpy as np
import os
import pandas as pd

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import torch.optim as optim

from tqdm import tqdm
import time

from model import YOLOv3
from utils import YOLODataset, YoloLoss, check_class_accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 4
BATCH_SIZE = 32
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

IMG_DIR = "500images"
LABEL_DIR = "500labels"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]


PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

transform = transforms.Compose([transforms.Resize((416, 416)), transforms.ToTensor()])

train_dataset = YOLODataset(
    csv_file = "500examples.csv",
    transform=transform,
    S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
    anchors=ANCHORS,
)

test_dataset = YOLODataset(
    csv_file = "500examples.csv",
    transform=transform,
    S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
    img_dir=IMG_DIR,
    label_dir=LABEL_DIR,
    anchors=ANCHORS,
)

if subset_len:
    train_dataset = torch.utils.data.Subset(
        train_dataset, random.sample(range(0, len(train_dataset)), subset_len))
    test_dataset = torch.utils.data.Subset(
        test_dataset, random.sample(range(0, len(test_dataset)), subset_len))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
else:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Training"""
# Instantiate the model
model = YOLOv3(num_classes=NUM_CLASSES).to(DEVICE)

# Compile the model
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)
loss_fn = YoloLoss()

# Scaler
scaler = torch.cuda.amp.GradScaler()

# Anchors
scaled_anchors = (
    torch.tensor(ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
).to(DEVICE)

# Training Loop
for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
  model.train()

  losses = []

  start_time = time.time() # Start time of the epoch

  for batch_idx, (x,y) in enumerate(train_loader):
    x = x.to(DEVICE)
    y0, y1, y2 = (y[0].to(DEVICE),
                  y[1].to(DEVICE),
                  y[2].to(DEVICE))

    # context manager is used in PyTorch to automatically handle mixed-precision computations on CUDA-enabled GPUs
    with torch.cuda.amp.autocast():
      out = model(x)
      loss = (
          loss_fn(out[0], y0, scaled_anchors[0])
          + loss_fn(out[1], y1, scaled_anchors[1])
          + loss_fn(out[2], y2, scaled_anchors[2])
      )

    losses.append(loss.item())

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

  end_time = time.time()  # End time of the epoch
  epoch_duration = end_time - start_time  # Duration of the epoch

  if (epoch+1) % 2 == 0:
    # Print the epoch duration
    tqdm.write(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

    # Print the loss and accuracy for training and validation data
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
          f"Loss: {sum(losses)/len(losses):.4f}")

    # save the model after every 10 epoch
    torch.save(model.state_dict(), f'Yolov3_epoch{epoch+1}.pth')

check_class_accuracy(model, train_loader, threshold=CONF_THRESHOLD)

"""Testing"""
losses = []

with torch.no_grad(): # Disabling gradient calculation, since testing doesnot require weight update
model.eval() # Set the model to evaluation mode

for inputs, y in val_loader:
    # Move the inputs and labels to the selected device
    inputs = inputs.to(device)
    y0, y1, y2 = (y[0].to(device),
                    y[1].to(device),
                    y[2].to(device))

    # Forward pass
    outputs = model(inputs)
    loss = (
        loss_fn(outputs[0], y0, scaled_anchors[0])
        + loss_fn(outputs[1], y1, scaled_anchors[1])
        + loss_fn(outputs[2], y2, scaled_anchors[2])
    )
    
    losses.append(loss.item())

Loss = sum(losses)/len(losses) 
print(f"Loss: {sum(losses)/len(losses):.4f}")

check_class_accuracy(model, train_loader, threshold=CONF_THRESHOLD)