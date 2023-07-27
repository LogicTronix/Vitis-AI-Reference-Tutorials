import argparse
import os
import shutil
import time
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional
from pytorch_nndct import QatProcessor

from model_qat import YOLOv3_qat
from utils import YOLODataset, YoloLoss, check_class_accuracy

parser = argparse.ArgumentParser()

parser.add_argument(
    '--batch_size', default=1, type=int, help='Batch size for training.')
parser.add_argument(
    '--mode',
    default='train',
    choices=['train', 'deploy'],
    help='Running mode.')
parser.add_argument(
    '--subset_len',
    default=1,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--output_dir', default='qat_result', help='Directory to save qat result.')
args, _ = parser.parse_known_args()


"""# Configure Hyperparameters"""
NUM_WORKERS = 4
IMAGE_SIZE = 416
NUM_CLASSES = 20
LEARNING_RATE = 1e-5
NUM_EPOCHS = 1
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Anchors
scaled_anchors = (
  torch.tensor(ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
).to(device)


def train(model, train_loader, criterion, device):
  # Compile the model
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  # Scaler
  scaler = torch.cuda.amp.GradScaler()

  num_epochs = 1 # For less computation

  for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()

    losses = []

    start_time = time.time() # Start time of the epoch

    for batch_idx, (x,y) in enumerate(train_loader):
      x = x.to(device)
      y0, y1, y2 = (y[0].to(device),
                    y[1].to(device),
                    y[2].to(device))

      # context manager is used in PyTorch to automatically handle mixed-precision computations on CUDA-enabled GPUs
      with torch.cuda.amp.autocast():
        out = model(x)
        loss = (
            criterion(out[0], y0, scaled_anchors[0])
            + criterion(out[1], y1, scaled_anchors[1])
            + criterion(out[2], y2, scaled_anchors[2])
        )

      losses.append(loss.item())

      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

    end_time = time.time()  # End time of the epoch
    epoch_duration = end_time - start_time  # Duration of the epoch

    # Print the epoch duration
    tqdm.write(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

    # Print the loss and accuracy for training and validation data
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Loss: {sum(losses)/len(losses):.4f}")

  return model


def main():
  print('Used arguments:', args)
  subset_len = args.subset_len
  batch_size = args.batch_size

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
  
  """# Load the model"""
  model = YOLOv3_qat(in_channels=3, num_classes=20)
  model.load_state_dict(torch.load("Yolov3_epoch20.pth", map_location=torch.device('cpu')))
  model = model.to(device)

  # define loss function (criterion)
  criterion = YoloLoss().to(device)

  inputs = torch.randn([batch_size, 3, 416, 416],
                       dtype=torch.float32,
                       device=device)

  # Instantiate QatProcessor Object
  qat_processor = QatProcessor(model, inputs)

  if args.mode == 'train':
    # Step 1: Get quantized model and train it.
    quantized_model = qat_processor.trainable_model(allow_reused_module=True)

    # Train quantized model and save the trained model
    qat_model = train(quantized_model, train_loader, criterion, device)
    torch.save(qat_model.state_dict(), "Yolov3_QAT.pth")

    # Evaluate the trained quantized model
    check_class_accuracy(qat_model, val_loader, threshold=CONF_THRESHOLD)

    # Step 2: Get deployable model and test it.
    # There may be some slight differences in accuracy with the quantized model.
    quantized_model.load_state_dict(torch.load("Yolov3_QAT.pth", map_location=torch.device('cpu')))
    deployable_model = qat_processor.to_deployable(quantized_model,
                                                   args.output_dir)
    check_class_accuracy(qat_model, val_loader, threshold=CONF_THRESHOLD)
  elif args.mode == 'deploy':
    # Step 3: Export xmodel from deployable model.
    deployable_model = qat_processor.deployable_model(
        args.output_dir, used_for_xmodel=True)
    val_subset = torch.utils.data.Subset(test_dataset, list(range(1)))
    subset_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False)
    # Must forward deployable model at least 1 iteration with batch_size=1
    for images, _ in subset_loader:
      deployable_model(images)
    qat_processor.export_torch_script(args.output_dir)
  else:
    raise ValueError('mode must be one of ["train", "deploy"]')

if __name__ == '__main__':
  main()