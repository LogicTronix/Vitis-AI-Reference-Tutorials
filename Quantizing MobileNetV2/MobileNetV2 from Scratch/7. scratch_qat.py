import argparse
import os
import shutil
import time
import random
import math
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


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        # Use functional.Add instead of +/torch.add
        self.skip_add = functional.Add()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add(x , self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=3, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building adaptive average pool
        self.adp_avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1)) 

        # building dropout
        self.dropout = nn.Dropout(p=0.2, inplace=False)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)

        # Quantization and De-Quantization
        self.quant_stub = nndct_nn.QuantStub()
        self.dequant_stub = nndct_nn.DeQuantStub()

        self._initialize_weights()

    def forward(self, x):
        x = self.quant_stub(x)

        x = self.features(x)
        x = self.adp_avgpool(x) # Replaced x.mean(3) with Adaptive AvgPool 
        x = x.reshape(x.size(0), -1) # Replaced x.mean(2) with Reshape
        x = self.dropout(x)
        x = self.classifier(x)

        x = self.dequant_stub(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


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
  # Instantiate MobileNetV2 model
  model = MobileNetV2()

  # Load the model
  model.load_state_dict(torch.load('MobileNetV2_scratch.pth', map_location=torch.device('cpu')))
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
    torch.save(qat_model.state_dict(), "MobileNetV2_QAT.pth")

    # Evaluate the trained quantized model
    validate(val_loader, qat_model, device)

    # Step 2: Get deployable model and test it.
    # There may be some slight differences in accuracy with the quantized model.
    quantized_model.load_state_dict(torch.load("MobileNetV2_QAT.pth", map_location=torch.device('cpu')))
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
