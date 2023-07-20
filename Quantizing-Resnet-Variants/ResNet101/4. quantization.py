import os
import re
import sys
import argparse
import time

import random
from pytorch_nndct.apis import torch_quantizer
import torch
import torchvision
import torchvision.transforms as transforms

from model import ResNet101

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=8,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')
parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')

args, _ = parser.parse_known_args()

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


def load_data(train=True,
              batch_size=8,
              subset_len=None,
              sample_method='random',
              model_name='resnet101',
              **kwargs):

  #prepare data
  # random.seed(12345)

  train_sampler = None

  if train:
    dataset = train_dataset
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        sampler=train_sampler,
        **kwargs)
  else:
    dataset = test_dataset
    if subset_len:
      assert subset_len <= len(dataset)
      if sample_method == 'random':
        dataset = torch.utils.data.Subset(
            dataset, random.sample(range(0, len(dataset)), subset_len))
      else:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, **kwargs)
  return data_loader, train_sampler


def evaluate(model, val_loader, loss_fn):
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
  return accuracy

def quantization(title='optimize',
                 model_name='', 
                 file_path=''): 

  quant_mode = args.quant_mode
  finetune = args.fast_finetune
  deploy = args.deploy
  batch_size = args.batch_size
  subset_len = args.subset_len
  config_file = args.config_file
  target = args.target

  # Assertions
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1 or subset_len != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1
    subset_len = 1

  model = ResNet101(image_channels=3, num_classes=3)
  model.load_state_dict(torch.load(file_path,  map_location=torch.device('cpu')))
  model = model.to(device)

  input = torch.randn([batch_size, 3, 224, 224])
  if quant_mode == 'float': # Evaluate the un-quantized model
    quant_model = model      
  else: # For calib and test --> Quantize the model
    ## new api
    ####################################################################################
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device, quant_config_file=config_file, target=target)

    quant_model = quantizer.quant_model
    #####################################################################################

  # to get loss value after evaluation
  loss_fn = torch.nn.CrossEntropyLoss().to(device)

  val_loader, _ = load_data(
      subset_len=subset_len,
      train=False,
      batch_size=batch_size,
      sample_method='random',
      model_name=model_name)

  # fast finetune model or load finetuned parameter before test
  if finetune == True:
      ft_loader, _ = load_data(
          subset_len=8,
          train=False,
          batch_size=batch_size,
          sample_method='random',
          model_name=model_name)
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
   
  # Evaluate -- Get training accuracy
  accuracy = evaluate(quant_model, val_loader, loss_fn)
  print(f"Accuracy: {accuracy:.4f}%")

  # handle quantization result
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if deploy:
    quantizer.export_torch_script()


if __name__ == '__main__':

  model_name = 'resnet101'
  file_path = 'ResNet101.pth'

  feature_test = ' float model evaluation'
  if args.quant_mode != 'float':
    feature_test = ' quantization'
    # force to merge BN with CONV for better quantization accuracy
    args.optimize = 1
    feature_test += ' with optimization'
  else:
    feature_test = ' float model evaluation'
  title = model_name + feature_test

  print("-------- Start {} test ".format(model_name))

  # calibration or evaluation
  quantization(
      title=title,
      model_name=model_name,
      file_path=file_path)

  print("-------- End of {} test ".format(model_name))
