# VART ResNet50 DPU Implementation Documentation
This code and flow explanation document is based on VART Resnet50 github example: https://github.com/Xilinx/Vitis-AI/tree/3.0/examples/vai_runtime/resnet50_mt_py
This ADAS Detection source is same for Vitis AI 3.5 and 3.0 repo.

## 1. Introduction

### ResNet50 Algorithm

ResNet50 is a sophisticated deep convolutional neural network (CNN) architecture, introduced by Kaiming He and colleagues in their 2015 paper "Deep Residual Learning for Image Recognition". The primary innovation of ResNet, short for Residual Network, is its use of residual connections or "skip connections" that create shortcuts between layers. This design addresses the vanishing gradient problem, which is prevalent in deep networks, by allowing gradients to be directly backpropagated to earlier layers. This not only makes the training of deeper networks feasible but also enhances the overall performance and accuracy of the model.

ResNet50, a specific variant of the ResNet architecture, is composed of 50 layers, including:
- **Convolutional Layers**: These layers perform the convolution operations that extract features from the input images.
- **Batch Normalization Layers**: These layers help in stabilizing and accelerating the training process by normalizing the inputs of each layer.
- **ReLU Activation Functions**: These functions introduce non-linearity into the model, enabling it to learn complex patterns.
- **Fully Connected Layers**: These layers are used for the final classification tasks.
- **Identity and Convolutional Shortcuts**: These shortcuts are used to implement the residual learning by skipping one or more layers.

![Resnet Architecture](https://miro.medium.com/v2/resize:fit:1400/0*9LqUp7XyEx1QNc6A.png)

**Fig. The architecture of ResNet-50 model**

The architecture follows a specific pattern where the convolutional layers are grouped into blocks, and each block contains multiple convolutional layers along with shortcuts that bypass these layers. This design allows the network to train deeper models without degradation, achieving state-of-the-art performance on various benchmark datasets such as ImageNet.

---

### Implementation Overview

The primary objective of this work is to implement the ResNet50 model using Xilinx's Deep Learning Processing Unit (DPU), a specialized hardware accelerator designed for the efficient execution of deep learning models. The DPU significantly enhances the performance of deep learning inference tasks by offloading the computationally intensive operations from the CPU or GPU.

The provided Python script demonstrates the end-to-end process of implementing and running the ResNet50 model on the DPU. The script encompasses several key components:

1. **Softmax Calculation**: 
   - Function: `CPUCalcSoftmax`
   - Description: This function computes the softmax probabilities from the raw output of the DPU. Softmax is an activation function that converts the output logits into probabilities, which sum up to 1. This is crucial for interpreting the model's output as probabilities for each class.

2. **Top-K Results Extraction**: 
   - Function: `TopK`
   - Description: This function identifies the top-K predictions from the softmax probabilities. It sorts the probabilities in descending order and selects the top K values, which correspond to the most likely predictions of the model.

3. **Image Preprocessing**: 
   - Function: `preprocess_one_image_fn`
   - Description: This function preprocesses the input images to the format required by the ResNet50 model. The preprocessing steps include resizing the image, subtracting mean values for normalization, and scaling the pixel values.

4. **DPU Execution**: 
   - Function: `runResnet50`
   - Description: This function handles the execution of the ResNet50 model on the DPU. It prepares the input data, runs the model on the DPU, and processes the output data. The function supports batch processing to maximize the efficiency of the DPU.

5. **Multithreading**: 
   - Description: To fully utilize the DPU's capabilities, the script employs multithreading. Multiple threads are created to run DPU inference in parallel, which significantly improves the throughput and overall performance.

6. **Main Function**: 
   - Function: `main`
   - Description: This is the entry point of the script. It orchestrates the entire process, from reading the input images, setting up the DPU runners, preprocessing the images, running the inference, and reporting the performance in frames per second (FPS).



## 2. Algorithm Overview

This section provides a detailed explanation of the algorithm implemented in the provided Python script. It covers the key components and steps involved in deploying the ResNet50 model on the Xilinx DPU, including pseudocode, a flowchart, and relevant equations.

### Pseudocode

The pseudocode below outlines the main steps of the algorithm:

#### Pseudocode for the Main Function

The `main` function orchestrates the overall process of loading and executing the ResNet50 model on the DPU. It handles setting up the environment, initializing the DPU runners, preprocessing the images, and managing multithreading for efficient execution.

```
FUNCTION main(argv):
    SET threadnum TO integer value of argv[1]
    SET calib_image_dir TO script directory + "/../images/"
    LIST listimage WITH files from calib_image_dir
    SET runTotall TO length of listimage
    LOAD graph FROM file argv[2]
    GET subgraphs USING get_child_subgraph_dpu(graph)
    ASSERT length of subgraphs IS 1
    LIST all_dpu_runners
    
    FOR i FROM 0 TO threadnum-1:
        APPEND vart.Runner.create_runner(subgraphs[0], "run") TO all_dpu_runners
    
    GET input_fixpos FROM all_dpu_runners[0]
    SET input_scale TO 2 RAISED_TO input_fixpos
    
    LIST img
    FOR each image IN listimage:
        SET path TO calib_image_dir + "/" + image
        APPEND preprocess_one_image_fn(path, input_scale) TO img
    
    SET cnt TO 360
    SET time_start TO current time
    
    LIST threadAll
    FOR i FROM 0 TO threadnum-1:
        CREATE thread t1 TO run runResnet50 WITH args (all_dpu_runners[i], img, cnt)
        APPEND t1 TO threadAll
    
    FOR each thread x IN threadAll:
        START x
    
    FOR each thread x IN threadAll:
        JOIN x
    
    DELETE all_dpu_runners
    SET time_end TO current time
    SET timetotal TO time_end - time_start
    SET total_frames TO cnt * threadnum
    SET fps TO total_frames DIVIDED_BY timetotal
    
    PRINT "FPS=%.2f, total frames = %.2f , time=%.6f seconds" % (fps, total_frames, timetotal)

END FUNCTION
```

**Explanation:**

1. **Initialize Variables and Directories**:
   - The function begins by extracting the number of threads from the command-line arguments.
   - It sets the directory where calibration images are stored.

2. **Load Images**:
   - Lists all images in the specified directory.
   - Sets the total number of images to be processed.

3. **Load DPU Graph**:
   - Deserializes the DPU graph from the provided file path.
   - Extracts subgraphs suitable for DPU execution.

4. **Create DPU Runners**:
   - Initializes DPU runners, one for each thread, and appends them to a list.

5. **Calculate Input Scale**:
   - Obtains the fixed-point position of the input tensor and calculates the scaling factor.

6. **Preprocess Images**:
   - Preprocesses each image using the `preprocess_one_image_fn` function and stores the results in a list.

7. **Set Execution Parameters**:
   - Defines the number of iterations each thread will run (`cnt`).
   - Records the start time for performance measurement.

8. **Create and Start Threads**:
   - Creates threads for running the `runResnet50` function.
   - Starts all threads.

9. **Join Threads**:
   - Waits for all threads to complete execution.

10. **Cleanup and Performance Measurement**:
    - Deletes the DPU runners to free up resources.
    - Calculates the total execution time and frames per second (FPS).
    - Prints the performance metrics.

---

#### Pseudocode for Preprocessing Function

The `preprocess_one_image_fn` function preprocesses input images to the required format for ResNet50. This involves resizing the image, normalizing it by subtracting mean values, scaling the pixel values, and converting the image to an appropriate data type.

```
FUNCTION preprocess_one_image_fn(image_path, fix_scale, width=224, height=224):
    SET means TO [104.0, 107.0, 123.0]
    SET scales TO [1.0, 1.0, 1.0]
    LOAD image USING cv2.imread(image_path)
    RESIZE image TO (width, height)
    SPLIT image INTO B, G, R channels
    FOR each channel IN [B, G, R]:
        SUBTRACT mean value FROM channel
        MULTIPLY channel BY scale
        MULTIPLY channel BY fix_scale
    MERGE channels INTO image
    CONVERT image TO np.int8
    RETURN image
END FUNCTION
```

**Explanation:**

1. **Set Mean and Scale Values**:
   - Defines mean values and scales for each color channel (B, G, R).

2. **Load and Resize Image**:
   - Reads the image from the specified path using OpenCV.
   - Resizes the image to the required dimensions (224x224).

3. **Normalize and Scale Channels**:
   - Splits the image into its B, G, and R channels.
   - For each channel, subtracts the corresponding mean value, multiplies by the scale factor, and applies the fixed-point scale.

4. **Merge Channels and Convert Type**:
   - Merges the normalized channels back into a single image.
   - Converts the image data type to `np.int8`.

5. **Return Preprocessed Image**:
   - Returns the preprocessed image, ready for DPU inference.

---

#### Pseudocode for DPU Execution Function

The `runResnet50` function is responsible for executing the ResNet50 model on the DPU. It prepares the input data, runs the model, and processes the output.

```
FUNCTION runResnet50(runner, img, cnt):
    GET inputTensors FROM runner
    GET outputTensors FROM runner
    SET input_ndim TO dimensions OF inputTensors[0]
    SET pre_output_size TO size OF outputTensors[0] DIVIDED_BY input_ndim[0]
    SET output_ndim TO dimensions OF outputTensors[0]
    GET output_fixpos FROM outputTensors[0]
    SET output_scale TO 1 DIVIDED_BY (2 RAISED_TO output_fixpos)
    SET n_of_images TO length OF img
    SET count TO 0
    
    WHILE count < cnt:
        SET runSize TO input_ndim[0]
        CREATE inputData AS empty array WITH shape input_ndim AND dtype np.int8
        CREATE outputData AS empty array WITH shape output_ndim AND dtype np.int8
        
        FOR j FROM 0 TO runSize-1:
            SET imageRun TO inputData[0]
            ASSIGN img[(count + j) % n_of_images].reshape(input_ndim[1:]) TO imageRun[j, ...]
        
        EXECUTE runner WITH inputData AND outputData ASYNCHRONOUSLY
        WAIT FOR completion OF execution
        
        INCREMENT count BY runSize
    
END FUNCTION
```

**Explanation:**

1. **Retrieve Tensors and Dimensions**:
   - Obtains input and output tensors from the DPU runner.
   - Determines the dimensions of the input and output tensors.
   - Calculates the size of the output data per image.

2. **Calculate Output Scale**:
   - Gets the fixed-point position of the output tensor.
   - Computes the scaling factor for the output data.

3. **Initialize Variables**:
   - Sets the number of images to be processed and initializes a counter.

4. **Run DPU Inference**:
   - Enters a loop that runs until the specified count (`cnt`) is reached.
   - For each iteration:
     - Prepares input and output data arrays.
     - Loads images into the input data buffer, reshaping them as required.
     - Executes the model on the DPU asynchronously and waits for completion.
     - Increments the count by the batch size (`runSize`).

By following these steps, the `runResnet50` function efficiently handles the DPU inference for the ResNet50 model, ensuring high performance through batch processing and multithreading.

---

#### Pseudocode for Efficient ResNet-50 Inference Using Xilinx DPU in Python

The steps for implementing and running the ResNet-50 model using the DPU (Deep Learning Processing Unit) with Python. It includes environment initialization, function definitions, and the main execution logic.

```
Load the pre-trained ResNet-50 model (xmodel file)
Create a graph object from the loaded model
Get the DPU subgraph from the graph object
Create DPU runners for the DPU subgraph
Get the directory containing input images
Preprocess the input images (resize, mean subtraction, and scaling)
for each batch of input images:
    Prepare input and output tensors for the batch
    Initialize input tensors with preprocessed images
    Execute the model on the DPU runners with input tensors
    (Optional) Perform softmax and top-k classification on the output tensors
Measure the FPS performance of the DPU runners
```

**Explanation:**

1. **Initialize Environment:**

    ```python
    import necessary libraries
    define constants for mean and scale values for image preprocessing
    ```
    - Essential libraries like `cv2` for image processing, `numpy` for numerical operations, and `xir` and `vart` for DPU operations are imported.
    - Constants for mean values and scale factors for the B, G, and R channels are defined to preprocess images correctly before feeding them into the DPU.

2. **Define Functions:**

    ```python
    def CPUCalcSoftmax(data, size, scale):
        compute softmax values for the given data

    def TopK(datain, size, filePath):
        determine top K predictions from the softmax output

    def preprocess_one_image_fn(image_path, fix_scale, width=224, height=224):
        preprocess a single image by resizing and normalizing it

    def get_script_directory():
        return the current script directory

    def runResnet50(runner, img, cnt):
        run the ResNet50 model on a batch of images using a DPU runner

    def get_child_subgraph_dpu(graph):
        get the DPU subgraph from the model graph
    ```

    - Computes the softmax values for the given data, normalizing the outputs into a probability distribution.
    - Determines the top K predictions from the softmax output and prints them, using labels from a provided file path.
    - Preprocesses a single image by resizing it to 224x224 pixels, normalizing it by subtracting mean values and scaling, and converting it to the required data type
    - Returns the directory where the current script is located.
    - Runs the ResNet50 model on a batch of images using a DPU runner. It prepares input and output tensors, executes the model, and (optionally) performs softmax and top-k classification.
    - Extracts the DPU subgraph from the model graph, ensuring that the subgraph is compatible with DPU execution.

3. **Main Function:**

    ```python
    parse command line arguments
    initialize variables and load images
    deserialize the model graph and obtain DPU subgraph
    create DPU runners
    preprocess images
    run inference using multiple threads
    measure and print performance metrics
    ```

    - Extracts the number of threads and the path to the ResNet50 xmodel file from the command-line arguments.
    - Initializes necessary variables, such as the number of threads and the directory containing input images. Loads the list of images from the specified directory.
    - Deserializes the ResNet50 model graph from the xmodel file and extracts the DPU subgraph using the get_child_subgraph_dpu function.
    - Creates DPU runners for the extracted subgraph, one for each thread specified.
    - Preprocesses each image in the input list using the `preprocess_one_image_fn` function, scaling them appropriately.
    - Spawns multiple threads to run the `runResnet50` function, distributing the workload across the DPU runners to improve efficiency and performance.
    - Measures the total time taken for inference, calculates the frames per second (FPS), and prints the performance metrics.

---

### Flowchart

The flowchart below visualizes the main steps of the algorithm:

```plaintext
 ┌─────────────────────────┐
 │ Load Pre-trained Model  │
 └─────────┬─────────────┬─┘
           │             │
 ┌─────────▼─────────┐  ┌▼─────────────────────┐
 │ Create Graph Obj. │  │ Get Input Image Dir. │
 └─────────┬─────────┘  └───────────┬──────────┘
           │                        │
 ┌─────────▼─────────┐              │
 │ Get DPU Subgraph  │              │
 └─────────┬─────────┘              │
           │                        │
 ┌─────────▼─────────┐  ┌───────────▼──────────┐
 │ Create DPU Runners│  │ Preprocess Input Imgs│
 └─────────┬─────────┘  └───────────┬──────────┘
           │                        │
           └─────────────────┬──────┘
                             │
                   ┌─────────▼─────────┐
                   │  Batch Input Imgs │
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │ Prepare Input/    │
                   │ Output Tensors    │
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │ Execute Model on  │
                   │ DPU Runners       │
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │ (Optional) Softmax│
                   │ & Top-K           │
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │ Measure FPS       │
                   └─────────────────┬─┘
                                     │
                           ┌─────────▼─────────┐
                           │     Output FPS    │
                           └───────────────────┘
```

This flowchart outlines the detailed steps involved in implementing and running the ResNet-50 model inference using the DPU (Deep Learning Processing Unit) with Python. Each block in the flowchart represents a specific step in the process. Here's an explanation of each step:

1. **Load Pre-trained Model**:
    - The first step involves loading the pre-trained ResNet-50 model, specifically the xmodel file that contains the trained parameters and architecture of the model.

2. **Create Graph Object**:
    - After loading the model, create a graph object from the loaded xmodel file. This graph object represents the entire computational graph of the model.

3. **Get Input Image Directory**:
    - Identify and set the directory where the input images are stored. This directory will be used to load and preprocess the images for inference.

4. **Get DPU Subgraph**:
    - Extract the DPU subgraph from the created graph object. This subgraph is the part of the model that will be executed on the DPU.

5. **Create DPU Runners**:
    - Create DPU runners for the extracted DPU subgraph. These runners are responsible for executing the model on the DPU hardware.

6. **Preprocess Input Images**:
    - Preprocess the input images from the specified directory. This includes resizing the images to the required dimensions, subtracting mean values, and scaling.

7. **Batch Input Images**:
    - Group the preprocessed images into batches. Batch processing helps in efficiently utilizing the DPU for inference.

8. **Prepare Input/Output Tensors**:
    - Prepare the input and output tensors required for the DPU runners. These tensors will hold the data that will be fed into and received from the DPU.

9. **Execute Model on DPU Runners**:
    - Execute the ResNet-50 model on the DPU runners using the prepared input tensors. This step involves running the inference on the DPU hardware.

10. **(Optional) Softmax & Top-K**:
    - Optionally, perform softmax and top-k classification on the output tensors. This step converts the raw output scores into probabilities and identifies the top predictions.

11. **Measure FPS**:
    - Measure the frames per second (FPS) performance of the DPU runners. This involves calculating the number of images processed per second during inference.

12. **Output FPS**:
    - Output the calculated FPS value. This gives an indication of the performance of the DPU in terms of how many images it can process per second.

This flowchart and the corresponding explanation provide a clear and detailed process for implementing ResNet-50 inference on the Xilinx DPU using Python. Each step is crucial for ensuring that the model runs efficiently on the DPU, from loading the model and preprocessing the images to executing the inference and measuring performance.

---

### Equations

#### 1. Softmax Calculation

The softmax function is used to convert the raw logits (output of the DPU) into probabilities. The equation for the softmax function is:

$\text{softmax}(z_i)$ = $\frac{e^{z_i}}{\sum_{j} e^{z_j}}$ 

Where $z_i$ is the raw score (logit) for the $i$-th class, and the denominator is the sum of the exponentials of all logits.

#### 2. Input Scaling

The input to the DPU is scaled using a fixed-point scaling factor. The equation for scaling the input data is:

$\text{ScaledInput}$ = $(\text{input} - \text{mean}) \times \text{scale} \times \text{fixScale}$

Where:
- $input$ is the pixel value.
- $mean$ is the mean value for the corresponding color channel.
- $scale$ is the predefined scale factor (1.0 in this case).
- $fixScale$ is $2^{\text{InputFixpos}}$, derived from the fixed-point position of the input tensor.

#### 3. FPS Calculation

The frames per second (FPS) is calculated to measure the performance of the DPU inference. The equation for FPS is:

$\text{FPS}$ = $\frac{\text{totalframes}}{\text{timetotal}}$

Where:
- $\text{totalframes}$ is the total number of frames processed (calculated as $\text{cnt} \times \text{threadnum} $).
- $\text{timetotal}$ is the total time taken for processing all frames.



## 3. Implementation Detail

1. ### **Load Pretrained Model**:
    The pretrained ResNet50 model is loaded from an XMODEL file, which is a Xilinx-specific format. The code deserializes the model using the `xir.Graph.deserialize()` function.

    ```python
    g = xir.Graph.deserialize(argv[2])
    ```

    The `argv[2]` argument is expected to be the path to the XMODEL file containing the pretrained ResNet50 model.

2. ### **Create Graph Object**:
    The deserialized model is represented as a graph object `g`. This object represents the computational graph of the neural network, including its layers and connections.

3. ### **Get Input Image Directory**:
    The script assumes that the input images are located in a directory called `images` relative to the script's directory. The directory path is obtained using the `get_script_directory()` function.

    ```python
    def get_script_directory():
        path = os.getcwd()
        return path

    SCRIPT_DIR = get_script_directory()
    calib_image_dir = SCRIPT_DIR + "/../images/"
    ```

    The `get_script_directory()` function retrieves the current working directory using `os.getcwd()`. The `SCRIPT_DIR` variable stores the path to the script's directory, and `calib_image_dir` is constructed by appending `"/../images/"` to `SCRIPT_DIR`, assuming that the `images` directory is located one level above the script's directory.

4. ### **Get DPU Subgraph**:
    The graph object `g` may contain multiple subgraphs, and the function `get_child_subgraph_dpu()` is used to extract the subgraph(s) that are intended to be executed on the DPU. It is assumed that there is only one such subgraph.

    ```python
    def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
        assert graph is not None, "'graph' should not be None."
        root_subgraph = graph.get_root_subgraph()
        assert (root_subgraph
                is not None), "Failed to get root subgraph of input Graph object."
        if root_subgraph.is_leaf:
            return []
        child_subgraphs = root_subgraph.toposort_child_subgraph()
        assert child_subgraphs is not None and len(child_subgraphs) > 0
        return [
            cs for cs in child_subgraphs
            if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
        ]

    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1  # only one DPU kernel
    ```

    The `get_child_subgraph_dpu()` function first checks if the input `graph` is not `None`. It then retrieves the root subgraph of the graph using `graph.get_root_subgraph()`. If the root subgraph is a leaf (i.e., it has no child subgraphs), the function returns an empty list. Otherwise, it retrieves the list of child subgraphs using `root_subgraph.toposort_child_subgraph()`.

    The function then filters the list of child subgraphs to include only those subgraphs that have the attribute "device" set to "DPU" (case-insensitive). The filtered list of DPU subgraphs is returned.

    In the code, `subgraphs` is assigned the result of `get_child_subgraph_dpu(g)`, and it is asserted that there is only one DPU subgraph.

5. ### **Create DPU Runners**:
    For each thread specified in the command-line argument, a DPU runner is created using `vart.Runner.create_runner()`. These runners will be used to execute the DPU subgraph.

    ```python
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    ```

    The `threadnum` variable is expected to be provided as the first command-line argument (`argv[1]`). It specifies the number of threads (and hence, DPU runners) to be created. The `vart.Runner.create_runner()` function creates a runner object for the given DPU subgraph (`subgraphs[0]`) with the name "run". These runner objects are appended to the `all_dpu_runners` list.

6. ### **Preprocess Input Images**:
    The script assumes that the input images are in the format expected by the ResNet50 model (i.e., RGB images with specific mean and scale values). The `preprocess_one_image_fn()` function is used to preprocess each input image by resizing it, subtracting the mean, and scaling the pixel values.

    ```python
    _B_MEAN = 104.0
    _G_MEAN = 107.0
    _R_MEAN = 123.0
    MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
    SCALES = [1.0, 1.0, 1.0]

    def preprocess_one_image_fn(image_path, fix_scale, width=224, height=224):
        means = MEANS
        scales = SCALES
        image = cv2.imread(image_path)
        image = cv2.resize(image, (width, height))
        B, G, R = cv2.split(image)
        B = (B - means[0]) * scales[0] * fix_scale
        G = (G - means[1]) * scales[1] * fix_scale
        R = (R - means[2]) * scales[2] * fix_scale
        image = cv2.merge([B, G, R])
        image = image.astype(np.int8)
        return image

    listimage = os.listdir(calib_image_dir)
    img = []
    for i in range(runTotall):
        path = os.path.join(calib_image_dir, listimage[i])
        img.append(preprocess_one_image_fn(path, input_scale))
    ```

    The `preprocess_one_image_fn()` function takes an image path, a fixed scale value (`fix_scale`), and optional width and height parameters (defaulting to 224). It first reads the image using `cv2.imread()` and resizes it to the specified width and height using `cv2.resize()`.

    The function then splits the image into its blue, green, and red channels using `cv2.split()`. It subtracts the corresponding mean values (`MEANS`) from each channel and scales the result by the corresponding scale values (`SCALES`) and the `fix_scale` value.

    The scaled channels are then merged back into a single image using `cv2.merge()`, and the resulting image is converted to `np.int8` data type before being returned.

    The code then lists all the image files in the `calib_image_dir` directory using `os.listdir()`. An empty list `img` is created, and a loop iterates over the number of images (`runTotall`). For each image, the file path is constructed by joining the `calib_image_dir` and the image filename (`listimage[i]`). The `preprocess_one_image_fn()` function is called with the image path and the `input_scale` value (obtained from the DPU runner's input tensor attributes), and the preprocessed image is appended to the `img` list.

7. ### **Prepare Input/Output Tensors**:
    Before executing the DPU runners, input and output tensors are created to hold the input images and output predictions, respectively. The input tensor dimensions are obtained from the DPU runner's input tensors, and the output tensor dimensions are obtained from the DPU runner's output tensors.

    ```python
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    pre_output_size = int(outputTensors[0].get_data_size() / input_ndim[0])

    output_ndim = tuple(outputTensors[0].dims)
    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)
    ```

    First, the script retrieves the input tensors and output tensors from the DPU runner using `runner.get_input_tensors()` and `runner.get_output_tensors()`, respectively. These tensors represent the expected input and output shapes of the ResNet50 model.

    The `input_ndim` variable is a tuple containing the dimensions of the input tensor, obtained by accessing the `dims` attribute of the first input tensor (`inputTensors[0].dims`). Similarly, the `output_ndim` variable is a tuple containing the dimensions of the output tensor, obtained from `outputTensors[0].dims`.

    The `pre_output_size` variable stores the size of the output tensor before the batch dimension. This is calculated by dividing the total data size of the output tensor (`outputTensors[0].get_data_size()`) by the batch size (`input_ndim[0]`).

    The `output_fixpos` variable retrieves the fixed-point position attribute of the output tensor using `outputTensors[0].get_attr("fix_point")`. This attribute represents the number of fractional bits in the fixed-point representation of the output data.

    The `output_scale` variable is calculated as `1 / (2**output_fixpos)`. This scale factor is used to convert the fixed-point output values to floating-point values during post-processing (e.g., softmax calculation).

    By retrieving these tensor dimensions and attributes, the script can properly allocate memory for the input and output tensors, as well as perform any necessary post-processing on the output tensor after executing the DPU runner.

    The `input_ndim`, `output_ndim`, and `output_scale` variables are used in subsequent steps of the code, such as batching the input images (step 8) and performing softmax and TopK calculations (step 10, if uncommented).

8. ### **Batch Input Images**:
    To improve performance, the script batches multiple input images together and processes them in a single DPU runner execution. The input images are reshaped and copied into the input tensor.

    ```python
    runSize = input_ndim[0]
    inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
    outputData = [np.empty(output_ndim, dtype=np.int8, order="C")]

    for j in range(runSize):
        imageRun = inputData[0]
        imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
    ```

    The `runSize` variable is set to the batch size, which is the first dimension of the input tensor (`input_ndim[0]`). An input tensor (`inputData`) and an output tensor (`outputData`) are created using `np.empty()` with the appropriate dimensions and data types.

    The code then loops over the `runSize` and copies the preprocessed input images from the `img` list into the input tensor. The `imageRun` variable refers to the first (and only) element of the `inputData` list, which is the input tensor. The line `imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])` copies the preprocessed image at index `(count + j) % n_of_images` from the `img` list into the `j`-th batch of the input tensor. The image is reshaped to match the dimensions of the input tensor, excluding the batch dimension.

9. ### **Execute Model on DPU Runner**:
    The input tensor(s) and output tensor(s) are passed to the DPU runner's `execute_async()` function, which schedules the execution on the DPU accelerator. The script then waits for the execution to complete using the `wait()` function.

    ```python
    job_id = runner.execute_async(inputData, outputData)
    runner.wait(job_id)
    ```

    The `execute_async()` function takes the input tensor(s) (`inputData`) and output tensor(s) (`outputData`) as arguments and returns a `job_id`. This `job_id` is then passed to the `wait()` function, which blocks until the execution is complete.

10. ### **Softmax and TopK**:
    The script includes code to perform softmax and TopK operations on the output tensor, but it is commented out by default. The `CPUCalcSoftmax()` function calculates the softmax of the output tensor, and the `TopK()` function finds the top-k predictions and their corresponding labels.

    ```python
    def CPUCalcSoftmax(data, size, scale):
        sum = 0.0
        result = [0 for i in range(size)]
        for i in range(size):
            result[i] = math.exp(data[i] * scale)
            sum += result[i]
        for i in range(size):
            result[i] /= sum
        return result

    def TopK(datain, size, filePath):
        cnt = [i for i in range(size)]
        pair = zip(datain, cnt)
        pair = sorted(pair, reverse=True)
        softmax_new, cnt_new = zip(*pair)
        fp = open(filePath, "r")
        data1 = fp.readlines()
        fp.close()
        for i in range(5):
            idx = 0
            for line in data1:
                if idx == cnt_new[i]:
                    print("Top[%d] %d %s" % (i, idx, (line.strip)("\n")))
                idx = idx + 1
    ```

    The `CPUCalcSoftmax()` function takes a data array, its size, and a scale value as input. It calculates the exponential of each element in the data array, scaled by the provided scale value. The exponentials are then summed up, and each element is divided by the sum to obtain the softmax values. The resulting softmax array is returned.

    The `TopK()` function takes the softmax array (`datain`), its size (`size`), and a file path (`filePath`) as input. It first creates a list `cnt` with indices corresponding to the elements in the softmax array. The `zip()` function is used to pair the softmax values with their indices, and the `sorted()` function sorts these pairs in descending order of the softmax values.

    The sorted pairs are then unpacked into two separate lists, `softmax_new` and `cnt_new`, using the `zip(*pair)` syntax. The `filePath` is assumed to contain a newline-separated list of labels corresponding to the output classes. The function opens this file, reads its contents into `data1`, and closes the file.

    Finally, a loop iterates over the top 5 indices from `cnt_new`, and for each index, it finds the corresponding label in `data1` and prints the top-k index, the label index, and the label text.

    These functions are not called by default in the provided code but can be uncommented if desired.

11. ### **Measure FPS**:
    After executing the DPU runners for all input images, the script measures the overall frames per second (FPS) by dividing the total number of processed frames by the total execution time.

    ```python
    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = cnt * int(threadnum)
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )
    ```

    The `time_end` variable stores the current time after the execution using `time.time()`. The `timetotal` variable calculates the total execution time by subtracting `time_start` (captured before the execution) from `time_end`.

    The `total_frames` variable stores the total number of frames processed by multiplying the `cnt` (the number of times each DPU runner executed) with the number of threads (`threadnum`).

    The frames per second (`fps`) is calculated by dividing `total_frames` by `timetotal`.

    Finally, the FPS, total frames, and total execution time are printed to the console using the `print` statement.

The implementation utilizes multithreading to improve performance by creating multiple DPU runners and processing input images in parallel. Each thread executes the `runResnet50()` function, which handles the batching, execution, and (optionally) softmax and TopK operations for a subset of the input images.

---

### Multi-threaded Inference

This implementation leverages multithreading to perform inference on the DPU in parallel, which can improve the overall performance and throughput. Here's how the multithreading is implemented:

1. **Create Multiple DPU Runners**:

    The number of threads is determined by the command-line argument `threadnum`. For each thread, a separate DPU runner is created using `vart.Runner.create_runner()`.

    ```python
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    ```

    Each DPU runner is an independent instance that can execute the DPU subgraph concurrently with other runners.

2. **Spawn Multiple Threads**:

    After creating the DPU runners, the script spawns multiple threads, each executing the `runResnet50()` function with one of the DPU runners and a subset of the input images.

    ```python
    threadAll = []
    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runResnet50, args=(all_dpu_runners[i], img, cnt))
        threadAll.append(t1)

    for x in threadAll:
        x.start()

    for x in threadAll:
        x.join()
    ```

    The `threading.Thread` class is used to create a new thread, with the `target` parameter specifying the function to be executed (`runResnet50`), and the `args` parameter providing the arguments to that function (the DPU runner, the list of input images, and the `cnt` value). Each thread is appended to the `threadAll` list.

    The `start()` method is called on each thread in the `threadAll` list to start their execution. The script then waits for all threads to complete by calling `join()` on each thread.

3. **Parallel Inference**:

    Inside the `runResnet50()` function, each thread performs the following steps in parallel:

    - Prepare batches of input images (`inputData`)
    - Execute the DPU runner with the input batch (`runner.execute_async()` and `runner.wait()`)
    - Optionally perform softmax and TopK calculations on the output batch

    Since each thread has its own DPU runner instance, they can execute concurrently on the DPU hardware, effectively parallelizing the inference process.

4. **Performance Measurement**:

    After all threads have completed, the script measures the overall performance by calculating the frames per second (FPS) based on the total number of frames processed and the total execution time.

    ```python
    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = cnt * int(threadnum)
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )
    ```

    The `total_frames` is calculated by multiplying the `cnt` (the number of times each DPU runner executed) by the number of threads (`threadnum`). This represents the total number of frames processed by all threads combined.

    The FPS is then calculated by dividing `total_frames` by `timetotal`, which is the total execution time across all threads.

By leveraging multithreading and executing multiple DPU runners in parallel, the implementation can take advantage of the available hardware resources (e.g., multiple DPU cores) and potentially achieve higher throughput and better performance compared to a single-threaded implementation.

It's important to note that the performance improvement from multithreading depends on various factors, such as the number of available hardware resources, the workload distribution among threads, and the overhead of thread management and synchronization.



## 4. Issues Faced and Solutions

During the implementation and execution of this ResNet50 DPU inference code, several issues may arise. Here are some potential issues and their solutions:

1. ### **Resource Contention and Synchronization**:
   - **Issue**: When multiple threads are executing concurrently, they may compete for shared resources like memory or hardware accelerators, leading to potential race conditions or performance bottlenecks.
   - **Solution**: Proper synchronization mechanisms, such as locks or semaphores, can be employed to ensure thread safety and prevent race conditions. Additionally, careful management of resource allocation and utilization is crucial to avoid oversubscription and ensure optimal performance.

2. ### **Input/Output Tensor Management**:
   - **Issue**: Improper handling of input and output tensors can lead to memory leaks, data corruption, or incorrect tensor shapes, which can result in erroneous or inconsistent results.
   - **Solution**: Implement robust tensor management techniques, such as proper memory allocation and deallocation, tensor shape validation, and error handling mechanisms. Ensure that input tensors are properly preprocessed and output tensors are correctly interpreted according to the model's specifications.

3. ### **Model Compatibility and Optimization**:
   - **Issue**: The ResNet50 model might have been trained or optimized for a different hardware platform or framework, leading to potential compatibility issues or suboptimal performance on the DPU accelerator.
   - **Solution**: Model quantization, calibration, and optimization techniques can be applied to ensure compatibility and optimal performance on the DPU accelerator. Additionally, thorough testing and benchmarking should be conducted to identify and address any performance bottlenecks or compatibility issues.

4. ### **Limited Hardware Resources**:
   - **Issue**: The available hardware resources, such as the number of DPU cores or memory capacity, might be limited, constraining the achievable performance or preventing the processing of larger input batches or models.
   - **Solution**: Implementing dynamic batch sizing or model partitioning strategies can help optimize resource utilization and enable processing of larger workloads within the available hardware constraints. Additionally, careful consideration should be given to the trade-off between performance and resource utilization when configuring the number of threads and batch sizes.

5. ### **Preprocessing and Post-processing Overhead**:
   - **Issue**: The preprocessing of input images and post-processing of output tensors (e.g., softmax and TopK calculations) can introduce additional computational overhead, potentially impacting the overall performance.
   - **Solution**: Optimize the preprocessing and post-processing steps by leveraging hardware acceleration (e.g., using optimized libraries or DPU kernels) or offloading these operations to separate threads or processes to reduce the impact on the inference pipeline.

Addressing these issues through careful design, optimization, and testing is crucial for achieving optimal performance, reliability, and scalability in real-world deployments.



## 5. Test Images

The provided implementation includes a set of test images to evaluate the performance of the ResNet50 model running on the Xilinx DPU. The images cover various object categories, such as airplanes, automobiles, and birds. Here are the details of the test images and their corresponding predictions:

### Test 1 - Class: Airplane
![airplane](Img_airplane.png)

Predicted class: airplane \
Probability: 0.9942418336868286

The first test image depicts an airplane, and the ResNet50 model accurately classifies it as an "airplane" with a high probability of 0.9942 (or 99.42%).

---

### Test 2 - Class: Automobile
![automobile](img_automobile.png)

Predicted class: automobile \
Probability: 0.9876630902290344

The second test image shows an automobile, and the model correctly identifies it as an "automobile" with a probability of 0.9877 (or 98.77%).

---

### Test 3 - Class: Bird
![bird](img_bird.png)

Predicted class: bird \
Probability: 0.6628021001815796

The third test image is of a bird, and the ResNet50 model classifies it as a "bird" with a relatively lower probability of 0.6628 (or 66.28%).

---

These test images serve as a validation set to assess the model's performance and accuracy on various object categories. The high probabilities for the "airplane" and "automobile" categories demonstrate the model's effectiveness in recognizing these objects, while the lower probability for the "bird" category highlights potential areas for improvement or further fine-tuning.

The provided test images and their corresponding predictions serve as a starting point for assessing the ResNet50 model's performance and can be extended with additional test cases or evaluation metrics as per the specific requirements of the application.

## 6. Reference Links

### 6.1 Directly Followed Links

- [Vitis AI Resnet50 Example](https://github.com/Xilinx/Vitis-AI/blob/3.0/examples/vai_runtime/resnet50_mt_py/resnet50.py): This is the GitHub link to the source code for the ResNet50 example running on the Xilinx DPU.
- [Xilinx Vitis-AI Repository](https://github.com/Xilinx/Vitis-AI/tree/3.0): This is the GitHub repository containing the example code and documentation for running ResNet50 and other deep learning model on the Xilinx DPU.

### 6.2 Referred Links for Supportive Tutorials, Books, or Similar

- [Vitis AI User Guide](https://docs.amd.com/r/3.0-English/ug1414-vitis-ai/Vitis-AI-Overview): The official user guide for Xilinx Vitis AI, which provides comprehensive information about the Vitis AI stack, including the DPU architecture, programming models, and development flows.
- [Vitis AI Tutorial](https://github.com/Xilinx/Vitis-AI-Tutorials): A collection of tutorials covering various aspects of Vitis AI, including DPU programming, quantization, and deployment.
- [Xilinx Vitis AI Video Tutorials](https://www.xilinx.com/video/vitis-ai.html): The official link to Vitis AI provided by Xilinx, covering various topics related to Vitis AI and DPU programming.
- [Threading in Python](https://docs.python.org/3/library/threading.html): Official Python documentation on the threading module, which provides a way to create and manage threads in Python.
- [ResNet50 Paper](https://arxiv.org/abs/1512.03385): The original paper by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, introducing the ResNet50 deep neural network architecture.
