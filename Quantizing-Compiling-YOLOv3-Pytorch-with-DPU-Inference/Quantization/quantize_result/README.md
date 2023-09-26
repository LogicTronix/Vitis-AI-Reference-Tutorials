> Since the quantized models are over 200Mb, GitHub doesn't allow pushing large files. But, here are the steps to get quantized result (Quantized torch script model and Quantized xmodel) on your own and by the end I have also added a google drive link where you can directly download the quantized model. 

## Steps to generate Quantized Result on your own:

### Step 1: Run 'quantize_yolov3.py' script - "calib"
```python
python quantize_yolov3.py --deploy_mode calib
``` 

### Step 2: Run 'quantize_yolov3.py' script - "test"
```python
python quantize_yolov3.py --deploy_mode test --batch_size 1 --deploy
```

*Note:* Run the script inside the Vitis AI pytorch environment because of certain APIs that Vitis AI provides and supports.

**You should see quantized_result directory in your workspace containing configuration .json file along with Quantized torch script (.pt) and Quantized XIR format model (.xmodel).**

## Directly download the Quantized Result
Get the quantized torch script and xmodel from the google drive link below: 

> [Google Drive](https://drive.google.com/drive/folders/1HON_lcEEscUbTDkRjFEapefXXrPVGJIC?usp=sharing)
