## 0. Overview

This documentation outlines the steps for quantizing, compiling, and performing inference with YOLOv3-Pytorch using DPU Inference. It covers GPU inference, quantization, quantized Torch Script inference, Compilation, and DPU inference on KV260 board to be specific.

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
