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
from torch.utils.data import Dataset, DataLoader

from model import YOLOv3
from utils import YOLODataset, YoloLoss, check_class_accuracy

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

"""# Configure Hyperparameters"""
import cv2
import torch

NUM_WORKERS = 4
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

# Anchors
scaled_anchors = (
    torch.tensor(ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
).to(device)

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

def load_data(train=True,
              batch_size=16,
              subset_len=None,
              sample_method='random',
              model_name='YoloV3',
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

  return Loss


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

  model = YOLOv3(in_channels=3, num_classes=20)
  model.load_state_dict(torch.load("Yolov3_epoch20.pth", map_location=torch.device('cpu')))
  model = model.to(device)

  input = torch.randn([batch_size, 3, 416, 416])
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
  loss_fn = YoloLoss().to(device)

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
   
  # Evaluate -- Get training accuracy and loss
  check_class_accuracy(quant_model, val_loader, threshold=CONF_THRESHOLD)
  print()
  loss = evaluate(quant_model, val_loader, loss_fn)
  print(f"Loss: {loss:.4f}")

  # handle quantization result
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if deploy:
    quantizer.export_torch_script()


if __name__ == '__main__':

  model_name = 'YoloV3'
  file_path = 'Yolov3_epoch20.pth'

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