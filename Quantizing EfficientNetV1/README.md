## EfficientNetV1 Model Training and Quantization Guide

### Overview:
This guide walks you through the process of training a EfficientNetV1 model, performing quantization using Post Training Quantization (PTQ) and Quantization Aware Training (QAT), and compiling it into a model. Additionally, it provides links to download the results of quantization and Quantization Aware Training (QAT) for direct use.


### Requirements for PTQ and QAT Quantization, Compile
For Post-Training Quantization (PTQ), Quantization-Aware Training (QAT), and model compilation, it is necessary to utilize the Vitis-AI PyTorch Docker image. Please ensure that you are running this repository on the appropriate environment with compatibility for the Vitis AI 3.0 GitHub branch and the latest Vitis-AI PyTorch Docker environment.


### Steps:

1. **Clone Repository:**
   ```bash
   git clone https://github.com/LogicTronix/Vitis-AI-Reference-Tutorials.git
   ```
2. **Navigate to EfficientNetV1 Directory:**
   ```bash
   cd 'Vitis-AI-Reference-Tutorials/Quantizing EfficientNetV1'/
   ```

3. **Open Directory in Code Editor (Visual Studio Code):**
   ```bash
   code .
   ```

<hr>

4. **Download dataset:**
    - Download the dataset used in this repository from the following link: [Dataset - Google Drive](https://drive.google.com/file/d/167L5B9ORegMzydr1SAT8EGDKjXgYaEim/view?usp=drive_link)
    - This dataset is necessary for training the EfficientNetV1 model. Ensure it is available in the appropriate directory within the project.

<hr>

5. **Train Float Model:**
   - To train the EfficientNetV1 model with floating-point precision, execute `model_train.py`.
   ```bash
   python model_train.py
   ```
   **Optional**
   - Alternatively, you can access the pre-trained floating-point model `EfficientNetV1.pth` directly from the directory within the project.

<hr>

6. **Inspect Model:**
   - Utilize `inspection.py` to inspect the trained and quantized models.
   ```bash
   python inspection.py
   ```
   The result of the inspector is stored in the `inspect` folder.

<hr>

7. **Post Training Quantization (PTQ):**
   - Run `quant.py` to perform Post Training Quantization (PTQ) on the trained model.

   **Calib step**
   ```bash
   python quant.py --quant_mode calib --subset_len 20
   ```
   **Test step**
   ```bash
   python quant.py --quant_mode test --subset_len 20
   ```

   **Deploy step**
   ```bash 
    python quant.py --quant_mode test --subset_len 1 --batch_size 1 --deploy
    ```
    > Note: In Calib and Test steps `subset_len` parameter has been set to small value for quicker execution of the script. This allows you to observe results rapidly, making it suitable for demonstration purposes or quick evaluation.

    <br>

   **Optional**
   - Alternatively, you can access the results of Post Training Quantization (PTQ) from `quantize_result` folder directly from the directory within the project.
   - Access the PTQ results (`quantize_result/EfficientNet_int.pt` and `quantize_result/EfficientNet_int.xmodel`) files in the project directory.

<hr>

8. **Quantization Aware Training (QAT):**
   - Run `qat.py` to perform Quantization Aware Training (QAT) on the trained model.

   **Train step**
   ```bash
   python qat.py --subset_len 4 --batch_size 2
   ```

   **Deploy step**
   ```bash
   python qat.py --mode deploy --subset_len 4 --batch_size 2
   ```

   > Note: The `subset_len` and `batch_size` parameters have been set to small values for quicker execution of the script. This allows you to observe results rapidly, making it suitable for demonstration purposes or quick evaluation.

    <br>

   **Optional**
   - Alternatively, you can access the results of Quantization Aware Training (QAT) from `qat_result` folder directly from the directory within the project.
   - Access the QAT results (`qat_result/EfficientNet_int.pt` and `qat_result/EfficientNet_0_int.xmodel`) files in the project directory.

<hr>

9. **Inference:**
   - For inference with the QAT torch script model, use `qat_inference.py`.
   ```bash
   python qat_inference.py
   ```
   - For inference with the PTQ torch script model, use `quant_inference.py`.
   ```bash
   python quant_inference.py
   ```

<hr>

10. Compile (For Kv260)
    - For compiling PTQ xmodel:
    ```bash
    vai_c_xir --xmodel ./quantize_result/EfficientNet_int.xmodel --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json --net_name EfficientNetV1 --output_dir ./Compiled
    ```
    - For compiling QAT xmodel:
    ```bash
    vai_c_xir --xmodel ./qat_result/EfficientNet_0_int.xmodel --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json --net_name EfficientNetV1 --output_dir ./Compiled_QAT
    ```

<hr>

**Note:** Ensure that necessary data and files are accessible before running the scripts. Adjust paths or configurations as needed based on your environment.
