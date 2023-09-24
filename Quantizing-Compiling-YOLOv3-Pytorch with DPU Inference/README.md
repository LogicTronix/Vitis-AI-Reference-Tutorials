## 0. Overview

Welcome to the "Quantizing-Compiling-YOLOv3-Pytorch with DPU Inference" project within the "Vitis-AI-Reference-Tutorials" repository. This project focuses on the implementation and optimization of YOLOv3 object detection using PyTorch, quantization for DPU (Deep Processing Unit) inference, and various related tasks. Whether you are new to the topic or looking to deepen your understanding, this repository offers a comprehensive set of resources and tutorials.

### Getting Started
If you're new to this project, we recommend starting with the README.md file, which provides an introduction and instructions for getting started. Explore the various directories to access tutorials, scripts, and resources relevant to your needs.

### Project Structure
- **Compiled:** This directory contains the compiled model outputs generated during the compilation process. These models are optimized for DPU inference.

- **DPU Inference:** Here, you will find scripts and resources related to DPU inference. This is where the magic happens when you run your optimized models on Xilinx's Deep Processing Unit.

- **GPU Inference:** This directory covers GPU inference scripts and modules. It's useful for those who want to leverage GPU acceleration for their YOLOv3 model.

- **Quantization:** Dive into quantization techniques and tools. This is where we optimize our models for deployment on DPU hardware.

- **Quantized Inference:** Learn how to perform inference with quantized models using Torch Script.

- **Common:** Find common utilities and scripts that are shared across different parts of the project.

- **Data, Evaluate, Nets, Test, Training, Weights:** These directories are the building blocks for training and evaluating your YOLOv3 models.

## 1. GPU inference
* Run **gpu_inference.py** script

├── Quantizing-Compiling-YOLOv3-Pytorch with DPU Inference\
│   ├── GPU inference\
│   │   ├── _ _ init _ _.py\
│   │   ├── **gpu_inference.py**\
│   │   ├── gpu_utils.py\
│   │   └── params.py
  
```bash
python gpu_inference.py
```
*Note:* Run the above script on Vitis AI pytorch environment to avoid errors. 

## 2. Quantization 
* Run **quantize_yolov3.py** script

├── Quantizing-Compiling-YOLOv3-Pytorch with DPU Inference\
│   ├── Quantization\
│   │   ├── quantize_result\
│   │   │   ├── ...\
│   │   ├── _ _ init _ _.py\
│   │   ├── params.py\
│   │   ├── **quantize_yolov3.py**\
│   │   └── quantized_model.txt

* For "calib"
```bash
python gpu_inference.py --quant_mode calib
```
* For "test"
```bash
python gpu_inference.py --quant_mode test --batch_size 1 --deploy
```
*Note:* Run the above script on Vitis AI pytorch environment to avoid errors. 

## 3. Inference of Quantized Torch Script
* Run **quantized_inference.py** script

├── Quantizing-Compiling-YOLOv3-Pytorch with DPU Inference\
│   ├── Quantized inference\
│   │   ├── params.py\
│   │   ├── **quantized_inference.py**\
│   │   └── quantized_utils.py

```bash
python quantized_inference.py
```
*Note:* Run the above script on Vitis AI pytorch environment to avoid errors. 

## 4. Compile
Compile quantized XIR formatted model using **VAI_C** from Vitis AI. 

```bash
vai_c_xir --xmodel ./Quantization/quantize_result/ModelMain_int.xmodel --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json --net_name Yolooooov3 --output_dir ./Compiled_Yolov3
```

* --xmodel : Path to quantized XIR formatted model.
* -- arch : Target to compile for. Here, KV260.
* -- net_name : Name of compiled XIR formatted model.
* --output_dir : Directory of compiled model.

After compilation, the Compiled (output directory) should look something like this:

├── Quantizing-Compiling-YOLOv3-Pytorch with DPU Inference\
│   ├── Compiled\
│   │   ├── Yolooooov3.xmodel\
│   │   ├── md5sum.txt\
│   │   └── meta.json

*Note:* Run the above commands on Vitis AI pytorch environment to avoid errors. 

## 5. DPU Inference
* Run **dpu_inference.py** script

├── Quantizing-Compiling-YOLOv3-Pytorch with DPU Inference\
│   ├── DPU inference\
│   │   ├── **dpu_inference.py**\
│   │   ├── dpu_utils.py\
│   │   └── params.py

```bash
python dpu_inference.py <compiled_xmodel_file> <test_image_path>
```
* <compiled_xmodel_file> : The compiled xmodel. Here, ../Compiled/Yolooooov3.xmodel
* <test_image_path> : The image you want to inference on. Here, ../test/images/test1.jpg

## Contribution
We welcome contributions from the community. If you have ideas, improvements, or fixes to share, please feel free to open issues or submit pull requests. Together, we can make this resource even more valuable to the community.
