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
      return self.dropout_rate(self.conv(x)) + inputs
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
    x = self.pool(self.features(x))
    return self.classifier(x.view(x.shape[0], -1))

