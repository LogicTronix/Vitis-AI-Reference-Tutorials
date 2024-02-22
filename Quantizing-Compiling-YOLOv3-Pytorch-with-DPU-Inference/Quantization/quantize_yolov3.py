import argparse
import importlib

import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer

import sys
import os

# Add the project directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..'))

from nets.model_main import ModelMain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')
parser.add_argument(
    '--batch_size',
    default=8,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')

args, _ = parser.parse_known_args()


def quantization(title='optimize',
                 model_name='', 
                 file_path=''): 
  quant_mode = args.quant_mode
  deploy = args.deploy
  batch_size = args.batch_size
  config_file = args.config_file

  # Assertions
  if quant_mode != 'test' and deploy:
    deploy = False
    print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
  if deploy and (batch_size != 1):
    print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
    batch_size = 1

  # Load the model
  params_path = 'params.py'
  config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS

  model = ModelMain(config, is_training=False)
  model.train(False)

  # Set data parallel
  model = nn.DataParallel(model)

  state_dict = torch.load(config["pretrain_snapshot"], map_location=torch.device(device))
  model.load_state_dict(state_dict)
  model = model.to(device)
  print(model)

  # Quantization
  input = torch.randn([batch_size, 3, 416, 416])
  if quant_mode == 'float': 
    quant_model = model      
  else: 
    ## new api
    ####################################################################################
    quantizer = torch_quantizer(
        quant_mode, model, (input), device=device, quant_config_file=config_file)

    quant_model = quantizer.quant_model
    #####################################################################################

  # Forward -- Dry Run
  input_data = torch.randn([batch_size, 3, 416, 416]).to(device)
  quant_model(input_data)

  # Handle quantization result
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_torch_script(verbose=False)
    quantizer.export_xmodel(deploy_check=True, dynamic_batch=True)


if __name__ == '__main__':

  model_name = 'YoloV3'
  file_path = '../weights/official_yolov3_weights_pytorch.pth'

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
