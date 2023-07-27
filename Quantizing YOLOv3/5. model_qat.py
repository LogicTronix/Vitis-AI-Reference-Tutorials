# Imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

import torch

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn

from pytorch_nndct import nn as nndct_nn
from pytorch_nndct.nn.modules import functional

"""
Architecture config:
- Tuple --> (filters, kernel_size, stride)
- List --> ['B', num_repeats] where 'B' is residual block
- 'S' --> scale prediction block. Also for computing yolo loss
- 'U' --> upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53

    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
  def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
    super(CNNBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs) # If batchnorm layer(bn_act) is true, then bias is False
    self.bn = nn.BatchNorm2d(out_channels)
    self.leaky = nn.LeakyReLU(0.1015625)
    self.use_bn_act = bn_act

  def forward(self, x):
    if self.use_bn_act:
      return self.leaky(self.bn(self.conv(x)))
    else:
      return self.conv(x)


class ResidualBlock(nn.Module):
  def __init__(self, channels, use_residual=True, num_repeats=1):
    super(ResidualBlock, self).__init__()
    self.layers = nn.ModuleList() # Like regular python list, but is container for pytorch nn modules

    for repeat in range(num_repeats):
      self.layers += [
          nn.Sequential(
            CNNBlock(channels, channels//2, kernel_size=1),
            CNNBlock(channels//2, channels, kernel_size=3, padding=1)
          )
      ]

    self.use_residual = use_residual
    self.num_repeats = num_repeats

    # Use a functional module to replace ‘+’
    self.skip_add = functional.Add()

  def forward(self, x):
    for layer in self.layers:
      if self.use_residual:
        forwarded_x = x
        x = layer(x)
        x = self.skip_add(x, forwarded_x)
      else:
        x = layer(x)

    return x


class ScalePrediction(nn.Module):
  def __init__(self, in_channels, num_classes):
    super(ScalePrediction, self).__init__()
    self.pred = nn.Sequential(
        CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
        CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1), # (num_classes + 5) * 3 --> (20+5) for each anchor box which in total is 3
    )
    self.num_classes = num_classes

  def forward(self, x):
    batch_size, _, grid_h, grid_w = x.shape
    # Pass through the prediction function
    pred_result = self.pred(x)
    # Combine reshape and permute operations using view
    fused_result = pred_result.view(batch_size, 3, grid_h, grid_w, self.num_classes + 5) # [batch_size, anchor_boxes, grid_h, grid_w, prediction(25)]
    
    return fused_result


class YOLOv3_qat(nn.Module):
  def __init__(self, in_channels=3, num_classes=20):
    super(YOLOv3_qat, self).__init__()
    self.num_classes = num_classes
    self.in_channels = in_channels
    self.layers = self._create_conv_layers()

    # Use a functional module to replace ‘torch.cat’
    self.concat = functional.Cat()

    # Quantization and De-Quantization
    self.quant_stub = nndct_nn.QuantStub()
    self.dequant_stub = nndct_nn.DeQuantStub()

  def forward(self, x):
    x = self.quant_stub(x)
    
    outputs = [] # store the output predictions from each ScalePrediction layer
    route_connections = [] # store the intermediate outputs from the layers with ResidualBlock and nn.Upsample.

    for layer in self.layers:
      if isinstance(layer, ScalePrediction): # checks if each layer is an instance of the class ScalePrediction 
        outputs.append(layer(x))
        continue

      x = layer(x)

      if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
        route_connections.append(x)

      elif isinstance(layer, nn.Upsample):
        x = self.concat([x, route_connections[-1]], dim=1)
        # x = torch.cat([x, route_connections[-1]], dim=1)
        route_connections.pop()
    
    x = self.dequant_stub(x)

    return outputs


  def _create_conv_layers(self):
    layers = nn.ModuleList()
    in_channels = self.in_channels

    for module in config:
      if isinstance(module, tuple):
        out_channels, kernel_size, stride = module
        layers.append(CNNBlock(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1 if kernel_size == 3 else 0
        ))
        in_channels = out_channels

      elif isinstance(module, list):
        num_repeats = module[1]
        layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

      elif isinstance(module, str):
        if module == "S":
          layers += [
              ResidualBlock(in_channels, use_residual=False, num_repeats=1),
              CNNBlock(in_channels, in_channels//2, kernel_size=1),
              ScalePrediction(in_channels//2, num_classes = self.num_classes)
          ]
          in_channels = in_channels // 2

        elif module == "U":
          layers.append(nn.Upsample(scale_factor=2))
          in_channels = in_channels * 3

    return layers
