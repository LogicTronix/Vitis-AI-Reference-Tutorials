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

from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional
from pytorch_nndct import QatProcessor

parser = argparse.ArgumentParser()

parser.add_argument(
    '--batch_size', default=4, type=int, help='Batch size for training.')
parser.add_argument(
    '--mode',
    default='train',
    choices=['train', 'deploy'],
    help='Running mode.')
parser.add_argument(
    '--subset_len',
    default=128,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--output_dir', default='qat_result', help='Directory to save qat result.')
args, _ = parser.parse_known_args()


class block(nn.Module):
  # Identity downsample for conv layer for projection shortcut
  def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
    super(block,self).__init__()
    self.expansion = 4

    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
    self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)

    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.relu3 = nn.ReLU()

    self.identity_downsample = identity_downsample

  def forward(self, x):
    identity = x

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)

    x = self.conv3(x)
    x = self.bn3(x)

    if self.identity_downsample is not None:
      identity = self.identity_downsample(identity)

    x += identity
    x = self.relu3(x)
    return x


class ResNet(nn.Module):
  def __init__(self, block, layers, image_channels, num_classes):
    super(ResNet, self).__init__()
    self.in_channels = 64 # Since, in resnet the input channel starts from 64

    # For Conv1 (image_channels in start is 3(RGB) --> after conv layer 64)
    self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU()

    # For Maxpool (channel 64 remains constant, spatial dimension decreases in half)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # ResNet layers (In ResNet101, there are 4 layers where, in each layer the blocks are repeated [3,4,23,3] times)
    self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
    self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
    self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
    self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

    # For Avg. Pooling such that the output is 2048x1x1 so Adaptive Pooling applied accordingly
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    # For final fully connected layer
    self.fc = nn.Linear(2048*1*1, num_classes)

    # Quantization and De-Quantization
    self.quant_stub = nndct_nn.QuantStub()
    self.dequant_stub = nndct_nn.DeQuantStub()

  def forward(self,x):
    x = self.quant_stub(x)

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.reshape(x.shape[0],-1)
    x = self.fc(x)

    x = self.dequant_stub(x)
    return x

  def _make_layer(self, block, num_residual_blocks, out_channels, stride):
    identity_downsample = None
    layers = []

    if stride!=1 or self.in_channels != out_channels*4 :
      identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,
                                                    stride=stride),
                                          nn.BatchNorm2d(out_channels*4))

    layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
    self.in_channels = out_channels*4

    for i in range(num_residual_blocks - 1):
      layers.append(block(self.in_channels, out_channels))

    return nn.Sequential(*layers)


def ResNet101(image_channels=3, num_classes=3):
  return ResNet(block, [3,4,23,3], image_channels, num_classes)


def validate(val_loader, model, device):
  correct = 0
  total = 0

  with torch.no_grad(): # Disabling gradient calculation, since testing doesnot require weight update
      model.eval() # Set the model to evaluation mode

      for inputs, labels in val_loader:
        # Move the inputs and labels to the selected device
          inputs = inputs.to(device)
          labels = labels.to(device)

          # Forward pass
          outputs = model(inputs)
          _, predicted = torch.max(outputs, 1)

          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  accuracy = 100 * correct / total
  print(f"Accuracy: {accuracy:.4f}%")


def train(model, train_loader, criterion, device, train_dataset):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
  num_epochs = 1 # For less computation

  for epoch in tqdm(range(num_epochs), desc='Epochs'):
      model.train()  # Set the model to training mode

      start_time = time.time() # Start time of the epoch

      running_loss = 0.0
      running_corrects = 0

      # Iterate over the training data in batches
      for inputs, labels in train_loader:
          # Move the inputs and labels to the selected device
          inputs = inputs.to(device)
          labels = labels.to(device)

          # Forward pass
          outputs = model(inputs)

          torch.cuda.empty_cache() # Limit GPU memory growth

          # Calculate the loss
          loss = criterion(outputs, labels)

          # Backward pass and optimization
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          torch.cuda.empty_cache() # Limit GPU memory growth

          # Update running loss and accuracy
          running_loss += loss.item() * inputs.size(0)
          _, preds = torch.max(outputs, 1) # the maximum value and the index of that maximum value along dimension 1
          running_corrects += torch.sum(preds == labels.data) # labels.data gives access to underlying labels tensor

      end_time = time.time()  # End time of the epoch
      epoch_duration = end_time - start_time  # Duration of the epoch

      # Print the epoch duration
      tqdm.write(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

      # Calculate epoch loss and accuracy for training data
      epoch_loss = running_loss / len(train_dataset)
      epoch_acc = running_corrects.double() / len(train_dataset)

      # Print the loss and accuracy for training and validation data
      print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

  return model


def main():
  print('Used arguments:', args)
  subset_len = args.subset_len
  batch_size = args.batch_size

  """# Load Filtered Dataset"""
  filtered_dataset = torch.load("filtered_dataset.pth")

  """# Random split filtered dataset""" 
  from torch.utils.data import random_split

  # Define the proportions for train and test split
  train_ratio = 0.8
  test_ratio = 1 - train_ratio

  # Calculate the sizes for train and test datasets
  train_size = int(train_ratio * len(filtered_dataset))
  test_size = len(filtered_dataset) - train_size

  # Split the dataset into train and test subsets
  train_dataset, test_dataset = random_split(filtered_dataset, [train_size, test_size])

  # Verify the sizes of the train and test datasets
  print("Train dataset size:", len(train_dataset))
  print("Test dataset size:", len(test_dataset))

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
  model = ResNet101(image_channels=3, num_classes=3)
  model.load_state_dict(torch.load("ResNet101.pth", map_location=torch.device(device)))
  model = model.to(device)

  # define loss function (criterion)
  criterion = nn.CrossEntropyLoss()

  inputs = torch.randn([batch_size, 3, 224, 224],
                       dtype=torch.float32,
                       device=device)

  # Instantiate QatProcessor Object
  qat_processor = QatProcessor(model, inputs)

  if args.mode == 'train':
    # Step 1: Get quantized model and train it.
    quantized_model = qat_processor.trainable_model()

    # Train quantized model and save the trained model
    qat_model = train(quantized_model, train_loader, criterion, device, train_dataset)
    torch.save(qat_model.state_dict(), "ResNet101_QAT_trained.pth")

    # Evaluate the trained quantized model
    validate(val_loader, qat_model, device)

    # Step 2: Get deployable model and test it.
    # There may be some slight differences in accuracy with the quantized model.
    quantized_model.load_state_dict(torch.load("ResNet101_QAT_trained.pth", map_location=torch.device(device)))
    deployable_model = qat_processor.to_deployable(quantized_model,
                                                   args.output_dir)
    validate(val_loader, deployable_model, device)
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
    qat_processor.export_xmodel(args.output_dir)
    qat_processor.export_torch_script(args.output_dir)
  else:
    raise ValueError('mode must be one of ["train", "deploy"]')

if __name__ == '__main__':
  main()