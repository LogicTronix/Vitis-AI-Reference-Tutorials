## MobileNetV2 Transfer Learning and Quantization Guide

### Overview:
This guide walks you through the process of training a MobileNetV2 model, performing quantization using Post Training Quantization (PTQ), and compiling it into a model. Additionally, it provides links to download the results of quantization for direct use.

**Note:** Quantization Aware Training (QAT) is not feasible for MobileNetV2 in a transfer learning scenario die to following reasons:
1. **Pre-trained Model Architecture:**
   - MobileNetV2 models used in transfer learning are typically imported from `torchvision.models.mobilenet_v2`.
   - These pre-trained models have fixed architectural configurations that cannot be modified during transfer learning.

2. **Lack of Architecture Manipulation:**
   - In transfer learning, the architecture of the pre-trained model remains unchanged.
   - There is no provision to insert additional layers, such as quantization-aware layers, into the architecture of pre-trained models.

3. **Quantization Aware Training (QAT) Requirements:**
   - QAT involves inserting fake quantization layers during training to simulate the effects of quantization.
   - MobileNetV2 models imported from torchvision do not support the insertion of such quantization-aware layers: `quant_stub` and `dequant_stub`.


### Steps:

1. **Clone Repository:**
   ```bash
   git clone https://github.com/LogicTronix/Vitis-AI-Reference-Tutorials.git
   ```
2. **Navigate to ResNet50 Directory:**
   ```bash
   cd 'Vitis-AI-Reference-Tutorials/Quantizing MobileNetV2/MobileNetV2 Transfer Learning'/
   ```

3. **Open Directory in Code Editor (Visual Studio Code):**
   ```bash
   code .
   ```

<hr>

4. **Download dataset:**
    - Download the dataset used in this repository from the following link: [Dataset - Google Drive](https://drive.google.com/file/d/1cB6OLCdHq0iMBBuqWRAA4SaQF6jxiz1z/view?usp=drive_link)
    - This dataset is necessary for training the MobileNetV2 model. Ensure it is available in the appropriate directory within the project.

<hr>

5. **Train Float Model:**
   - To train the MobileNetV2 model with floating-point precision, execute the jupiter notebook `MobileNetV2_transfer_learning.ipynb`.

   **Optional**
   - Alternatively, you can access the pre-trained floating-point model `MobileNetV2_transfer_learning.pth` directly from the directory within the project.

<hr>

6. **Inference Float Model:**
   - Perform inference using the trained MobileNetV2 model with the Python script: `transfer_learning_trained_model_inference.py`.
   ```bash
   python transfer_learning_trained_model_inference.py
   ```

<hr>

7. **Inspect Model:**
   - Utilize `transfer_learning_inspection.py` to inspect the trained and quantized models.
   ```bash
   python transfer_learning_inspection.py
   ```
   The result of the inspector is stored in the `transfer_learning_inspect` folder.

<hr>

8. **Post Training Quantization (PTQ):**
   - Run `transfer_learning_quant.py` to perform Post Training Quantization (PTQ) on the trained model.

   **Calib step**
   ```bash
   python transfer_learning_quant.py --quant_mode calib --subset_len 20
   ```
   **Test step**
   ```bash
   python transfer_learning_quant.py --quant_mode test --subset_len 20
   ```

   **Deploy step**
   ```bash 
    python transfer_learning_quant.py --quant_mode test --subset_len 1 --batch_size 1 --deploy
    ```
    > Note: In Calib and Test steps `subset_len` parameter has been set to small value for quicker execution of the script. This allows you to observe results rapidly, making it suitable for demonstration purposes or quick evaluation.

    <br>

   **Optional**
   - Alternatively, you can access the results of Post Training Quantization (PTQ) from `quantize_result` folder directly from the directory within the project.
   - Access the PTQ results (`quantize_result/MobileNetV2_int.pt` and `quantize_result/MobileNetV2_int.xmodel`) files in the project directory.

<hr>

9. **Inference:**
   - For inference with the PTQ torch script model, use `transfer_learning_quantized_inference.py`.
   ```bash
   python transfer_learning_quantized_inference.py
   ```

<hr>

10. Compile (For Kv260)
    - For compiling PTQ xmodel:
    ```bash
    vai_c_xir --xmodel ./quantize_result/MobileNetV2_int.xmodel --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json --net_name MobileNetV2 --output_dir ./Compiled
    ```

<hr>

**Note:** Ensure that necessary data and files are accessible before running the scripts. Adjust paths or configurations as needed based on your environment.
