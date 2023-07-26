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

# Imports
import torch
import torch.nn as nn
from math import ceil

# Specifying base model for MBConv
base_model = [
    # Expand_ratio, channels, repeats, stride, kernel_size
    [1,16,1,1,3],
    [6,24,2,2,3],
    [6,40,2,2,5],
    [6,80,3,2,3],
    [6,112,3,1,5],
    [6,192,4,2,5],
    [6,320,1,1,3],
]

phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0":(0, 224, 0.2),
    "b1":(0.5, 240, 0.2),
    "b2":(1, 260, 0.3),
    "b3":(2, 300, 0.3),
    "b4":(3, 380, 0.4),
    "b5":(4, 456, 0.4),
    "b6":(5, 528, 0.5),
    "b7":(6, 600, 0.5),
}

class CNNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
    super(CNNBlock, self).__init__()

    self.cnn = nn.Conv2d(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         groups=groups, # 'groups' parameter for depth-wise convolution i.e for each channel independently
                         bias=False)

    self.bn = nn.BatchNorm2d(out_channels)
    self.hard_swish = nn.Hardswish()

  def forward(self, x):
    return self.hard_swish(self.bn(self.cnn(x)))

 
class SqueezeExcitation(nn.Module):
  def __init__(self, in_channels, reduced_dim):
    super(SqueezeExcitation, self).__init__()

    self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), # CxHxW --> Cx1x1
                            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
                            nn.Hardswish(),
                            nn.Conv2d(reduced_dim, in_channels, kernel_size=1), # Brings back the original channels
                            nn.Sigmoid()) # Reduce the channels

  def forward(self, x):
    return x * self.se(x) # Each channel is multiplied to the value/attention scores which determines the priority


class InvertedResidualBlock(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride,
               padding,
               expand_ratio, # Expands i/p to higher number of channels --> depth-wise conv --> back to same shape of i/p
               reduction=4,# For squeeze excitation
               dropout_index=None
               ):
    super(InvertedResidualBlock, self).__init__()

    self.use_residual = in_channels == out_channels and stride == 1 # Use residual connection if condition satisfies
    self.dropout_index = dropout_index

    hidden_dim = in_channels * expand_ratio
    self.expand = in_channels != hidden_dim # In same stage -- no expansion , when switching stages -- expansion

    reduced_dim = int(in_channels / reduction)

    if self.expand:
      self.expand_conv = CNNBlock(
          in_channels, hidden_dim, kernel_size=1, stride=1, padding=0,
      )

    self.conv = nn.Sequential(
        CNNBlock(
            hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
        ),
        SqueezeExcitation(hidden_dim, reduced_dim),
        nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels))

    # Use a functional module to replace ‘+’
    self.skip_add = functional.Add()

  # For dropout
  def dropout_rate(self, x):
    if not self.training:
      return x

    dropout_rates = [0.025, 0.05, 0.075, 0.0875, 0.1125, 0.125, 0.15, 0.1625, 0.175]

    dropout_rate = dropout_rates[self.dropout_index] # Get dropout rate from the list
    dropout_layer = nn.Dropout(dropout_rate) # Create dropout layer
    return dropout_layer(x)


  def forward(self, inputs):
    x = self.expand_conv(inputs) if self.expand else inputs

    if self.use_residual:
      # Use a functional module to replace ‘+’
      return self.skip_add(self.dropout_rate(self.conv(x)), inputs)
      # return self.dropout_rate(self.conv(x)) + inputs
    else:
      return self.conv(x)


class EfficientNet(nn.Module):
  def __init__(self, version, num_classes):
    super(EfficientNet, self).__init__()

    width_factor, depth_factor, dropout_rate = self.calculate_factors(version)
    last_channels = ceil(1280 * width_factor)

    self.pool = nn.AdaptiveAvgPool2d(1)

    self.features = self.create_features(width_factor, depth_factor, last_channels)
    self.classifier = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(last_channels, num_classes),
    )

    # Quantization and De-Quantization
    self.quant_stub = nndct_nn.QuantStub()
    self.dequant_stub = nndct_nn.DeQuantStub()

  def calculate_factors(self, version, alpha=1.2, beta=1.1):
    phi, resolution, drop_rate = phi_values[version]
    depth_factor = alpha ** phi
    width_factor = beta ** phi
    return width_factor, depth_factor, drop_rate

  def create_features(self, width_factor, depth_factor, last_channels):
    channels = int(32 * width_factor) # From initial conv layer, there are 32 channels which increases between stages
    features = [CNNBlock(3, channels, 3, stride=2, padding=1)] # Since image has in_channels = 3
    in_channels = channels # Update in_channels for next layer

    dropout_index = 0  # Initialize the dropout index

    # Now iterate through all stages of base model
    for expand_ratio, channels, repeats, stride, kernel_size in base_model:

      # Since during Squeeze Excitation, we reduce the channels by 4, so making sure the out_channels is divisible by 4
      out_channels = 4 * ceil(int(channels * width_factor) / 4)

      layers_repeats = ceil(repeats * depth_factor)

      for layer in range(layers_repeats):
        if layer != 0: # Apply dropout
          features.append(
              InvertedResidualBlock(
                  in_channels,
                  out_channels,
                  expand_ratio = expand_ratio,
                  stride = stride if layer == 0 else 1,
                  kernel_size = kernel_size,
                  padding = kernel_size//2, # if k=1:pad=0, k=3:pad=1, k=5:pad=2
                  dropout_index = dropout_index
              )
          )
          dropout_index += dropout_index # Increment the dropout index
        else:
          features.append(
                    InvertedResidualBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
        in_channels = out_channels


    features.append(
        CNNBlock(in_channels, last_channels, kernel_size=1, stride=1, padding=0)
    )

    return nn.Sequential(*features)

  def forward(self, x):
    x = self.quant_stub(x)

    x = self.pool(self.features(x))
    x = self.classifier(x.view(x.shape[0], -1))

    x = self.dequant_stub(x)
    return x


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
  model = EfficientNet(version='b0', num_classes=3)
  model.load_state_dict(torch.load("EfficientNetV1.pth", map_location=torch.device('cpu')))
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
    torch.save(qat_model.state_dict(), "EfficientNetV1_QAT_trained.pth")

    # Evaluate the trained quantized model
    validate(val_loader, qat_model, device)

    # Step 2: Get deployable model and test it.
    # There may be some slight differences in accuracy with the quantized model.
    quantized_model.load_state_dict(torch.load("EfficientNetV1_QAT_trained.pth", map_location=torch.device('cpu')))
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


