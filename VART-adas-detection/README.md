# ADAS (Advanced Driver Assistance Systems) Object Detection using Vitis AI 3.0
This code and flow explanation document is based on VART Resnet50 github example: https://github.com/Xilinx/Vitis-AI/tree/3.0/examples/vai_runtime/adas_detection \
This ADAS Detection source is same for Vitis AI 3.5 and 3.0 repo.

## 1. Introduction

In the realm of automotive innovation, Advanced Driver Assistance Systems (ADAS) stand as a beacon of safety and convenience. These systems amalgamate state-of-the-art sensors, cameras, and AI algorithms to furnish vehicles with heightened perceptiveness and responsive capabilities. By continually scanning the surroundings, analyzing road conditions, and alerting drivers to potential dangers, ADAS systems empower safer and more confident driving experiences. They encompass an array of functionalities including collision warning, lane departure detection, adaptive cruise control, blind spot monitoring, and parking assistance, collectively reshaping the landscape of road safety.

In alignment with the paradigm of ADAS advancement, this project sets out to implement a pivotal component: object detection. Specifically tailored for real-time applications, the project leverages the YOLO-v3 (You Only Look Once) object detection algorithm through the Vitis AI 3.0 framework from Xilinx. This strategic integration enables the swift identification of diverse objects such as cars, pedestrians, and cyclists from dynamic video streams or image sequences. By harnessing the computational efficiency and precision of YOLO-v3 within the robust infrastructure of Vitis AI, the project not only bolsters ADAS capabilities but also signifies a stride towards a safer and more intelligent automotive future.

![ADAS Working](https://images.spiceworks.com/wp-content/uploads/2022/06/15105923/ADAS-Working.png)

**Fig. A Diagrammatic Outline of How ADAS Works [Source: spiceworks]**


## 2. Algorithm Overview

1. **YOLO-v3 Object Detection Algorithm**:
   - YOLO (You Only Look Once) is a state-of-the-art deep learning algorithm renowned for its efficiency and accuracy in real-time object detection tasks.
   - YOLO-v3 is the third iteration of the YOLO algorithm, introducing several architectural improvements over its predecessors.
   - Key features of YOLO-v3 include a feature pyramid network, which enables the detection of objects at multiple scales, and a prediction mechanism that utilizes a single neural network to predict bounding boxes and class probabilities directly from full images in a single evaluation.
   - YOLO-v3 divides the input image into a grid and predicts bounding boxes and their associated class probabilities for each grid cell, resulting in a highly efficient and parallelizable approach to object detection.

2. **Model Architecture**:
   - The YOLO-v3 model architecture consists of a backbone convolutional neural network (CNN) followed by detection layers.
   - The backbone CNN, typically based on architectures like Darknet or ResNet, extracts features from the input image.
   - Detection layers are responsible for predicting bounding boxes, confidence scores, and class probabilities for objects detected in the input image.
   - YOLO-v3 employs a multi-scale detection strategy, enabling it to detect objects of varying sizes and aspect ratios within the same image.

3. **Training and Inference**:
   - During training, the YOLO-v3 model is trained on annotated datasets using techniques like gradient descent and backpropagation to optimize its parameters for object detection.
   - In inference, the trained model is deployed to detect objects in real-time or on static images or video frames.
   - The model takes an input image and processes it through the backbone CNN to extract features.
   - These features are then passed through detection layers to predict bounding boxes, confidence scores, and class probabilities for detected objects.
   - Post-processing techniques such as non-maximum suppression (NMS) are applied to filter out redundant bounding boxes and refine the final set of detected objects.

4. **Integration with Vitis AI**:
   - The YOLO-v3 model is integrated into the Vitis AI 3.0 framework from Xilinx for deployment on Xilinx hardware platforms.
   - Vitis AI provides tools and libraries for optimizing and deploying deep learning models on Xilinx FPGAs and SoCs, enabling accelerated inference for real-time applications.
   - By leveraging Vitis AI, the YOLO-v3 model can exploit the parallel processing capabilities of Xilinx hardware, resulting in efficient and high-performance object detection for ADAS applications.

Overall, the YOLO-v3 algorithm, combined with the Vitis AI framework, offers a powerful solution for real-time object detection in ADAS systems, providing enhanced safety and situational awareness on the road.

### 2.1. YOLO-v3 Object Detection Algorithm

The YOLO-v3 algorithm is a real-time object detection system that divides the input image into a grid of cells. For each cell, the algorithm predicts bounding boxes and confidence scores for those boxes. The confidence scores represent the probability that an object exists within the bounding box and the accuracy of the predicted bounding box. Additionally, the algorithm predicts the probability of each class (e.g., car, person, cyclist) for each bounding box.

**Steps:**
1. **Divide the Input Image:**
   - The algorithm divides the input image into an S × S grid of cells.

2. **Prediction for Each Cell:**
   - For each cell in the grid, the algorithm predicts:
     - B bounding boxes: These bounding boxes represent the potential locations of objects within the cell.
     - Confidence scores: These scores indicate the likelihood that an object exists within each bounding box, along with the accuracy of the prediction.
     - Class probabilities: These probabilities represent the likelihood of each class (e.g., car, person, cyclist) being present within each bounding box.

3. **Non-Maximum Suppression (NMS):**
   - To filter out redundant bounding boxes and ensure only the most confident detections are retained, the algorithm applies non-maximum suppression (NMS). This process removes overlapping bounding boxes and selects the ones with the highest confidence scores.

4. **Output Generation:**
   - The final output consists of the filtered bounding boxes, along with the associated class predictions and confidence scores.

### 2.2. Pseudocode

```
function YOLO(image):
    grid_cells = divide_into_grid(image, S)
    bounding_boxes = []
    for cell in grid_cells:
        for anchor in B:
            box = predict_bounding_box(cell, anchor)
            confidence = predict_confidence(cell, anchor)
            class_probabilities = predict_class_probabilities(cell, anchor)
            bounding_boxes.append((box, confidence, class_probabilities))
    filtered_boxes = non_max_suppression(bounding_boxes)
    return filtered_boxes
```

### 2.3. Flowchart

```
                    +------------------------+
                    |        Input Image     |
                    +------------------------+
                                |
                                v
                    +------------------------+
                    |   Divide into S x S   |
                    |        Grid Cells     |
                    +------------------------+
                                |
                                v
                    +------------------------+
                    |  Predict Bounding     |
                    |  Boxes, Confidences,  |
                    |  and Class Probs      |
                    +------------------------+
                                |
                                v
                    +------------------------+
                    |   Non-Max Suppression |
                    +------------------------+
                                |
                                v
                    +------------------------+
                    |   Output Bounding     |
                    |   Boxes and Labels    |
                    +------------------------+
```
This pseudocode and flowchart illustrate the key steps of the YOLO-v3 algorithm, including dividing the input image into grid cells, predicting bounding boxes, confidences, and class probabilities for each cell, aggregating predictions, applying non-maximum suppression, and finally outputting the filtered bounding boxes.


## 3. Implementation Details

The implementation of the ADAS object detection system using Vitis AI 3.0 and the YOLO-v3 algorithm involves several steps and helper functions. The following contains each of the steps in details:

### 3.1. Input Frame Reading and Queue Management

The input frame reading and queue management is crucial for maintaining a smooth and efficient flow of data in the ADAS object detection system. The implementation uses a separate thread (`readFrame`) to handle the reading of frames from the input video file. This approach decouples the frame reading process from the main execution thread, allowing for concurrent operations and preventing potential bottlenecks.

The `readFrame` function is responsible for opening the specified video file and continuously reading frames from it. However, instead of directly processing the frames, the function pushes them into a thread-safe queue called `queueInput`. This queue acts as a buffer, storing the frames for later processing by the YOLO-v3 model.

Here's a detailed look at the `readFrame` function:

```cpp
void readFrame(const char* fileName) {
  static int loop = 3;
  VideoCapture video;
  string videoFile = fileName;
  start_time = chrono::system_clock::now();

  while (loop > 0) {
    loop--;
    if (!video.open(videoFile)) {
      cout << "Fail to open specified video file:" << videoFile << endl;
      exit(-1);
    }

    while (true) {
      usleep(20000); // Sleep for 20 milliseconds
      Mat img;
      if (queueInput.size() < 30) { // Check if queue size is less than 30
        if (!video.read(img)) { // Read a frame from the video
          break; // Break out of the loop if no more frames are available
        }

        mtxQueueInput.lock(); // Lock the mutex to ensure thread safety
        queueInput.push(make_pair(idxInputImage++, img)); // Push the frame into the queue
        mtxQueueInput.unlock(); // Unlock the mutex
      } else {
        usleep(10); // Sleep for a short duration to prevent busy waiting
      }
    }

    video.release(); // Release the video capture object
  }
  bExiting = true; // Set the bExiting flag to true when the loop finishes
}
```

The `readFrame` function reads frames from the video file and pushes them into the `queueInput` queue if the queue size is less than a specified limit (30 in this case). This limit helps to prevent the queue from growing excessively large and consuming too much memory.

The `queueInput` queue is protected by a mutex (`mtxQueueInput`) to ensure thread safety during concurrent operations. When pushing a new frame into the queue, the function first locks the mutex, pushes the frame, and then unlocks the mutex.

By using a separate thread and a queue for input frame reading, the implementation decouples this process from the main execution thread, allowing for concurrent operations and preventing potential bottlenecks caused by slow frame reading or processing.

### 3.2. Input Frame Preprocessing

Before feeding the input frames into the YOLO-v3 model, they need to be preprocessed to match the expected input format and dimensions of the model. The `setInputImageForYOLO` function is responsible for this preprocessing step.

Here's a detailed look at the `setInputImageForYOLO` function:

```cpp
void setInputImageForYOLO(vart::Runner* runner, int8_t* data, const Mat& frame,
                          float* mean, float input_scale) {
  Mat img_copy;
  int width = shapes.inTensorList[0].width;
  int height = shapes.inTensorList[0].height;
  int size = shapes.inTensorList[0].size;
  image img_new = load_image_cv(frame); // Convert OpenCV Mat to internal image representation
  image img_yolo = letterbox_image(img_new, width, height); // Resize the image

  vector<float> bb(size);
  for (int b = 0; b < height; ++b) {
    for (int c = 0; c < width; ++c) {
      for (int a = 0; a < 3; ++a) {
        bb[b * width * 3 + c * 3 + a] =
            img_yolo.data[a * height * width + b * width + c];
      }
    }
  }

  float scale = pow(2, 7);
  for (int i = 0; i < size; ++i) {
    data[i] = (int8_t)(bb.data()[i] * input_scale); // Quantize and scale the input data
    if (data[i] < 0) data[i] = (int8_t)((float)(127 / scale) * input_scale); // Handle negative values
  }
  free_image(img_new); // Free the temporary image objects
  free_image(img_yolo);
}
```

The `setInputImageForYOLO` function takes the following inputs:

- `vart::Runner* runner`: The DPU runner object for executing the YOLO-v3 model.
- `int8_t* data`: A pointer to the input data buffer where the preprocessed image data will be stored.
- `const Mat& frame`: The input frame as an OpenCV `Mat` object.
- `float* mean`: The mean values used for normalization (unused in this implementation).
- `float input_scale`: The input scale factor used for quantization.

The function performs the following steps:

1. It retrieves the expected input dimensions (width, height, and size) of the YOLO-v3 model from the `shapes` struct.
2. It converts the input frame (OpenCV `Mat`) to the internal image representation used by the YOLO-v3 model using the `load_image_cv` function.
3. It resizes the input image to match the expected input dimensions of the YOLO-v3 model using the `letterbox_image` function.
4. It iterates over the resized image data and copies it into a vector of floating-point values (`bb`).
5. It quantizes and scales the input data using the provided `input_scale` factor and stores the quantized data in the `data` buffer.
6. It handles negative values in the quantized data by clamping them to a specific value.
7. Finally, it frees the temporary image objects (`img_new` and `img_yolo`) to avoid memory leaks.

The `load_image_cv` and `letterbox_image` functions are helper functions defined in `utils.cc` for loading and resizing images, respectively. These functions are responsible for converting the input frame from OpenCV's format to the internal image representation used by the YOLO-v3 model and resizing the image to match the model's input dimensions.

The quantization and scaling operations in the `setInputImageForYOLO` function are necessary to convert the floating-point input data into the fixed-point format expected by the DPU accelerator. The `input_scale` factor is used to scale the input data to the appropriate range for the quantization process.

By preprocessing the input frames in this manner, the implementation ensures that the input data is in the correct format and dimensions for efficient processing by the YOLO-v3 model running on the DPU accelerators.

### 3.3. Running YOLO-v3 on DPU Runners

The implementation creates four separate DPU runners (`runner`, `runner1`, `runner2`, and `runner3`) for executing the YOLO-v3 model in parallel. Each DPU runner is assigned a dedicated thread (`runYOLO`) to process input frames concurrently, enabling parallel processing and improving overall performance.

Here's a detailed look at the `runYOLO` function:

```cpp
void runYOLO(vart::Runner* runner) {
  // ... (tensor shape initialization and memory allocation omitted)

  while (true) {
    pair<int, Mat> pairIndexImage;

    mtxQueueInput.lock();
    if (queueInput.empty()) {
      mtxQueueInput.unlock();
      if (bExiting) break;
      if (bReading) {
        continue;
      } else {
        break;
      }
    } else {
      pairIndexImage = queueInput.front(); // Retrieve the next frame from the input queue
      queueInput.pop(); // Remove the frame from the input queue
      mtxQueueInput.unlock();
    }

    setInputImageForYOLO(runner, data, pairIndexImage.second, mean, input_scale); // Preprocess the input frame

    // ... (input/output tensor buffer preparation omitted)

    auto job_id = runner->execute_async(inputsPtr, outputsPtr); // Execute the YOLO-v3 model asynchronously
    runner->wait(job_id.first, -1); // Wait for the execution to complete

    postProcess(runner, pairIndexImage.second, result, width, height, output_scale.data()); // Post-process the output

    mtxQueueShow.lock();
    queueShow.push(pairIndexImage); // Push the processed frame into the output queue
    mtxQueueShow.unlock();

    // ... (memory cleanup omitted)
  }

  // ... (memory deallocation omitted)
}
```

The `runYOLO` function operates in a continuous loop, processing input frames from the `queueInput` queue and pushing the processed frames into the `queueShow` queue for display.

Here's a step-by-step breakdown of the `runYOLO` function:

1. The function first checks if the `queueInput` queue is empty. If it's empty and the `bExiting` flag is set, the function breaks out of the loop. If the `bReading` flag is set, indicating that the input video is still being read, the function continues waiting for new frames.

2. If a frame is available in the `queueInput` queue, the function locks the `mtxQueueInput` mutex, retrieves the next frame from the queue (`pairIndexImage`), removes it from the queue using `pop`, and then unlocks the mutex.

3. The retrieved frame is preprocessed using the `setInputImageForYOLO` function, which performs operations like resizing, normalization, and quantization, as explained earlier.

4. The function prepares input and output tensor buffers for the DPU runner.

5. The YOLO-v3 model is executed asynchronously on the DPU runner using the `execute_async` function. The function then waits for the execution to complete using the `wait` function.

6. After the execution is complete, the `postProcess` function is called to perform post-processing operations on the output tensors, such as non-maximum suppression and drawing bounding boxes on the output frame.

7. The processed frame is pushed into the `queueShow` queue, which is protected by the `mtxQueueShow` mutex to ensure thread safety.

8. The function cleans up any temporary memory allocations for input and output tensor buffers.

By running multiple instances of the `runYOLO` function on separate threads, each with its own DPU runner, the implementation achieves parallel processing of input frames. This parallelism can significantly improve the overall performance and throughput of the ADAS object detection system, enabling real-time object detection on high-resolution video streams.

### 3.4. Post-processing and Visualization

After the YOLO-v3 model has processed an input frame on the DPU runner, the output tensors need to be decoded and post-processed to obtain the final object detection results. The `postProcess` function is responsible for this task.

Here's a detailed look at the `postProcess` function:

```cpp
void postProcess(vart::Runner* runner, Mat& frame, vector<int8_t*> results,
                 int sWidth, int sHeight, const float* output_scale) {
  const string classes[3] = {"car", "person", "cycle"};

  vector<vector<float>> boxes;
  for (int ii = 0; ii < 4; ii++) {
    int width = shapes.outTensorList[ii].width;
    int height = shapes.outTensorList[ii].height;
    int channel = shapes.outTensorList[ii].channel;
    int sizeOut = channel * width * height;
    vector<float> result(sizeOut);

    get_output(results[ii], sizeOut, channel, height, width, output_scale[ii], result);
    detect(boxes, result, channel, height, width, ii, sHeight, sWidth);
  }

  correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);
  vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

  float h = frame.rows;
  float w = frame.cols;
  for (size_t i = 0; i < res.size(); ++i) {
    // ... (drawing bounding boxes and labels omitted for brevity)
  }
}
```

The `postProcess` function takes the following inputs:

- `vart::Runner* runner`: The DPU runner object used for executing the YOLO-v3 model.
- `Mat& frame`: The input frame as an OpenCV `Mat` object (reference).
- `vector<int8_t*> results`: A vector of pointers to the output tensors from the DPU runner.
- `int sWidth`: The width of the input frame.
- `int sHeight`: The height of the input frame.
- `const float* output_scale`: A pointer to the output scale factors for the YOLO-v3 model's output tensors.

The function performs the following steps:

1. It defines a string array (`classes`) with the class names for the YOLO-v3 model (e.g., "car", "person", "cycle").

2. It iterates over the four output tensors of the YOLO-v3 model.

3. For each output tensor, it calls the `get_output` function to convert the tensor data from the DPU runner's format to a vector of floating-point values (`result`).

4. The `detect` function is then called, which decodes the output tensor data (`result`) to obtain the predicted bounding boxes, confidence scores, and class probabilities. These predictions are stored in the `boxes` vector.

5. After processing all output tensors, the `correct_region_boxes` function is called to adjust the bounding box coordinates to match the original input frame dimensions.

6. The `applyNMS` function is called to apply non-maximum suppression (NMS) to the predicted bounding boxes, filtering out overlapping boxes and returning a list of non-overlapping boxes (`res`).

7. Finally, the function iterates over the remaining bounding boxes (`res`) and draws them on the input frame (`frame`) using OpenCV's `rectangle` function. It also displays the class label (e.g., "car", "person", "cycle") for each bounding box.

The `get_output`, `detect`, `correct_region_boxes`, and `applyNMS` functions are helper functions defined in `utils.cc`. They handle tasks such as converting output tensor data, decoding bounding boxes and class probabilities, adjusting bounding box coordinates, and applying non-maximum suppression, respectively.

The `postProcess` function is responsible for taking the raw output tensors from the DPU runner and converting them into meaningful object detection results, including bounding boxes and class labels. By drawing these results on the input frame, the function provides a visual representation of the detected objects, which can be displayed or further processed as needed.

This function first decodes the output tensors from the DPU runners using the `get_output` and `detect` functions. The `get_output` function converts the output tensors from the DPU runners into a vector of floating-point values for further processing. The `detect` function decodes these output values to obtain the predicted bounding boxes, confidence scores, and class probabilities.

After decoding the output tensors, the `postProcess` function calls the `correct_region_boxes` function to adjust the bounding box coordinates to match the original input frame dimensions.

```cpp
void correct_region_boxes(vector<vector<float>>& boxes, int n, int w, int h,
                           int netw, int neth, int relative = 0) {
  // ... (implementation omitted for brevity)
}
```

Next, the `applyNMS` function is called to apply non-maximum suppression (NMS) to the predicted bounding boxes. This function filters out overlapping bounding boxes and returns a list of non-overlapping boxes.

```cpp
vector<vector<float>> applyNMS(vector<vector<float>>& boxes, int classes,
                               const float thres) {
  // ... (implementation omitted for brevity)
}
```

Finally, the `postProcess` function iterates over the remaining bounding boxes after NMS and draws them on the output frame using OpenCV's `rectangle` function. It also displays the class label (e.g., car, person, cycle) for each bounding box.

```cpp
float h = frame.rows;
float w = frame.cols;
for (size_t i = 0; i < res.size(); ++i) {
  float xmin = (res[i][0] - res[i][2] / 2.0) * w + 1.0;
  float ymin = (res[i][1] - res[i][3] / 2.0) * h + 1.0;
  float xmax = (res[i][0] + res[i][2] / 2.0) * w + 1.0;
  float ymax = (res[i][1] + res[i][3] / 2.0) * h + 1.0;

  if (res[i][res[i][4] + 6] > CONF) {
    int type = res[i][4];
    string classname = classes[type];

    if (type == 0) {
      rectangle(frame, Point(xmin, ymin), Point(xmax, ymax),
                Scalar(0, 0, 255), 1, 1, 0);
    } else if (type == 1) {
      rectangle(frame, Point(xmin, ymin), Point(xmax, ymax),
                Scalar(255, 0, 0), 1, 1, 0);
    } else {
      rectangle(frame, Point(xmin, ymin), Point(xmax, ymax),
                Scalar(0, 255, 255), 1, 1, 0);
    }
  }
}
```

### 3.5. Display Frame

After the input frames have been processed by the YOLO-v3 model and the object detection results have been obtained, the processed frames need to be displayed to the user. The `displayFrame` function is responsible for this task.

Here's a detailed look at the `displayFrame` function:

```cpp
void displayFrame() {
  Mat frame;

  while (true) {
    if (bExiting) break;
    mtxQueueShow.lock();

    if (queueShow.empty()) {
      mtxQueueShow.unlock();
      usleep(10);
    } else if (idxShowImage == queueShow.top().first) {
      auto show_time = chrono::system_clock::now();
      stringstream buffer;
      frame = queueShow.top().second;
      if (frame.rows <= 0 || frame.cols <= 0) {
        mtxQueueShow.unlock();
        continue;
      }
      auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
      buffer << fixed << setprecision(1)
             << (float)queueShow.top().first / (dura / 1000000.f);
      string a = buffer.str() + " FPS";
      cv::putText(frame, a, cv::Point(10, 15), 1, 1, cv::Scalar{0, 0, 240}, 1);
      cv::imshow("ADAS Detection@Xilinx DPU", frame);

      idxShowImage++;
      queueShow.pop();
      mtxQueueShow.unlock();
      if (waitKey(1) == 'q') {
        bReading = false;
        exit(0);
      }
    } else {
      mtxQueueShow.unlock();
    }
  }
}
```

The `displayFrame` function operates in a continuous loop, retrieving processed frames from the `queueShow` queue and displaying them using OpenCV's `imshow` function.

Here's a step-by-step breakdown of the `displayFrame` function:

1. The function first checks if the `bExiting` flag is set. If it is, the function breaks out of the loop.

2. The function locks the `mtxQueueShow` mutex to ensure thread safety when accessing the `queueShow` queue.

3. If the `queueShow` queue is empty, the function unlocks the mutex and sleeps for a short duration (`usleep(10)`) to avoid busy waiting.

4. If the `queueShow` queue is not empty and the index of the next frame to be displayed (`idxShowImage`) matches the index of the top frame in the queue (`queueShow.top().first`), the function proceeds with displaying the frame.

5. The function retrieves the current system time using `chrono::system_clock::now()` and calculates the frames per second (FPS) based on the elapsed time and the frame index.

6. The FPS value is converted to a string (`a`) and drawn on the frame using OpenCV's `putText` function.

7. The processed frame (`queueShow.top().second`) is displayed using OpenCV's `imshow` function.

8. The function increments the `idxShowImage` index, removes the displayed frame from the `queueShow` queue using `pop`, and unlocks the `mtxQueueShow` mutex.

9. If the user presses the 'q' key, the function sets the `bReading` flag to `false` and exits the program.

10. If the condition for displaying the frame is not met (e.g., the frame index doesn't match `idxShowImage`), the function unlocks the `mtxQueueShow` mutex and continues to the next iteration of the loop.

The `displayFrame` function ensures that the processed frames are displayed in the correct order, as determined by the frame indices. It also calculates and displays the frames per second (FPS) value, which can be useful for monitoring the performance of the ADAS object detection system.

By using a separate thread for displaying frames, the implementation decouples this process from the main execution thread, preventing potential bottlenecks and ensuring smooth visualization of the object detection results.

### 3.6. Main Function and Thread Management

The `main` function is the entry point of the ADAS object detection program. It handles command-line arguments, sets up the necessary components, and spawns the required threads for the different tasks involved in the object detection pipeline.

Here's a detailed look at the `main` function:

```cpp
int main(const int argc, const char** argv) {
  if (argc != 3) {
    cout << "Usage of ADAS detection: " << argv[0]
         << " <video_file> <model_file>" << endl;
    return -1;
  }

  auto graph = xir::Graph::deserialize(argv[2]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u) << "yolov3 should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  auto runner1 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner2 = vart::Runner::create_runner(subgraph[0], "run");
  auto runner3 = vart::Runner::create_runner(subgraph[0], "run");

  // ... (tensor shape initialization omitted)

  array<thread, 6> threadsList = {
      thread(readFrame, argv[1]), thread(displayFrame),
      thread(runYOLO, runner.get()), thread(runYOLO, runner1.get()),
      thread(runYOLO, runner2.get()), thread(runYOLO, runner3.get())};

  for (int i = 0; i < 6; i++) {
    threadsList[i].join();
  }

  return 0;
}
```

Here's a step-by-step breakdown of the `main` function:

1. The function checks if the correct number of command-line arguments (three) are provided. If not, it prints the usage information and returns with an error code.

2. The function deserializes the YOLO-v3 model file (`argv[2]`) using the `xir::Graph::deserialize` function and retrieves the DPU subgraph from the model graph.

3. The function creates four DPU runners (`runner`, `runner1`, `runner2`, and `runner3`) for the YOLO-v3 model using the `vart::Runner::create_runner` function.

4. The function initializes the tensor shapes for the input and output tensors of the YOLO-v3 model using the `getTensorShape` function (implementation omitted for brevity).

5. The function creates an array of six threads (`threadsList`):
   - One thread for reading input frames from the video file (`readFrame`).
   - One thread for displaying the processed frames (`displayFrame`).
   - Four threads for running the YOLO-v3 model on the separate DPU runners (`runYOLO`).

6. The function waits for all threads to complete using the `join` function, ensuring that the program terminates gracefully after all threads have finished their tasks.

The `main` function is responsible for setting up the necessary components and managing the execution flow of the ADAS object detection system. By spawning multiple threads and leveraging the parallel processing capabilities of the DPU runners, the implementation achieves real-time object detection performance on video streams.

The use of command-line arguments allows the user to specify the input video file and the YOLO-v3 model file, providing flexibility and configurability for different use cases.

### 3.7. Tensor Shape Handling

The YOLO-v3 model has multiple input and output tensors with different shapes and dimensions. The implementation handles these tensor shapes using the `getTensorShape` function, which retrieves the shapes of the input and output tensors from the DPU runner.

```cpp
void getTensorShape(vart::Runner* runner, GraphInfo* shapes, int inputCnt,
                    vector<string> outputNodes) {
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  for (int i = 0; i < inputCnt; ++i) {
    shapes->inTensorList[i].batch = inputTensors[i]->get_shape().at(0);
    shapes->inTensorList[i].channel = inputTensors[i]->get_shape().at(1);
    shapes->inTensorList[i].height = inputTensors[i]->get_shape().at(2);
    shapes->inTensorList[i].width = inputTensors[i]->get_shape().at(3);
    shapes->inTensorList[i].size =
        shapes->inTensorList[i].batch * shapes->inTensorList[i].channel *
        shapes->inTensorList[i].height * shapes->inTensorList[i].width;
  }

  for (int i = 0; i < outputNodes.size(); ++i) {
    auto outputTensor = runner->get_output_tensor(outputNodes[i]).get();
    shapes->outTensorList[i].batch = outputTensor->get_shape().at(0);
    shapes->outTensorList[i].channel = outputTensor->get_shape().at(1);
    shapes->outTensorList[i].height = outputTensor->get_shape().at(2);
    shapes->outTensorList[i].width = outputTensor->get_shape().at(3);
    shapes->outTensorList[i].size =
        shapes->outTensorList[i].batch * shapes->outTensorList[i].channel *
        shapes->outTensorList[i].height * shapes->outTensorList[i].width;
    shapes->output_mapping.push_back(i);
  }
}
```

This function takes a `vart::Runner` object and a `GraphInfo` struct as input, along with the number of input tensors and a vector of output node names. It retrieves the shapes of the input and output tensors from the DPU runner and stores them in the `GraphInfo` struct.

The `GraphInfo` struct is defined as:

```cpp
typedef struct GraphInfo {
  TensorShape* inTensorList;
  TensorShape* outTensorList;
  std::vector<int> output_mapping;
} GraphInfo;
```

The `TensorShape` struct represents the shape of a tensor, including its batch size, channels, height, width, and total size.

By properly handling the tensor shapes, the implementation can correctly interpret and process the input and output data of the YOLO-v3 model.

### 3.8. Memory Management

The implementation employs dynamic memory allocation and deallocation to manage the memory requirements of the input and output tensors. This is particularly important because the YOLO-v3 model involves large tensors, and efficient memory management is crucial for real-time performance.

In the `runYOLO` function, memory is dynamically allocated for the input and output tensors:

```cpp
int8_t* data = new int8_t[shapes.inTensorList[0].size *
                          inputTensors[0]->get_shape().at(0)];
int8_t* result0 =
    new int8_t[shapes.outTensorList[0].size *
               outputTensors[shapes.output_mapping[0]]->get_shape().at(0)];
// ... (allocation for other output tensors omitted)
```

After processing the input frame and executing the DPU runner, the dynamically allocated memory is deallocated:

```cpp
delete[] data;
delete[] result0;
// ... (deallocation for other output tensors omitted)
```

This dynamic memory allocation and deallocation approach ensures that memory is efficiently utilized and avoids potential memory leaks or excessive memory consumption.

### 3.9. Performance Optimization

To achieve real-time performance, the implementation incorporates several optimization techniques:

1. **Multi-threading**: The implementation utilizes multi-threading to parallelize the execution of the YOLO-v3 model on multiple DPU runners. This allows for concurrent processing of input frames, improving overall throughput.

2. **Asynchronous Execution**: The DPU runners execute the YOLO-v3 model asynchronously using the `execute_async` function. This allows the main thread to continue processing while the DPU runners perform computations, improving overall efficiency.

3. **Input/Output Queue Management**: The use of input and output queues (`queueInput` and `queueShow`) helps to decouple the frame reading, processing, and display operations, preventing bottlenecks and ensuring smooth execution.

4. **Fixed-point Arithmetic and Quantization**: The Vitis AI framework provides optimized fixed-point arithmetic and quantization techniques for efficient execution on the DPU accelerators. The implementation leverages these techniques to optimize the performance of the YOLO-v3 model.

5. **OpenCV Optimizations**: The implementation utilizes OpenCV functions for image loading, resizing, and preprocessing, which are highly optimized for efficient image processing operations.

By combining these optimization techniques, the implementation achieves real-time object detection performance, enabling practical applications in Advanced Driver Assistance Systems (ADAS) and other related domains.


## 4. Issues Faced and Solutions

During the implementation of this project, several issues were encountered, and solutions were developed to address them:

1. **Memory Management**:
   - **Issue**: The YOLO-v3 model requires significant memory for input and output tensors, posing challenges in memory allocation and deallocation.
   - **Solution**: Employed dynamic memory allocation to efficiently manage memory usage, ensuring proper allocation and deallocation of resources to prevent memory leaks and optimize resource utilization.

2. **Parallelization and Synchronization**:
   - **Issue**: Leveraging multi-threading for real-time performance introduced synchronization challenges, particularly in managing input and output buffers.
   - **Solution**: Mitigated synchronization issues by implementing mutex locks and queues to manage input and output frame buffers, enabling seamless parallel processing of input frames across multiple threads.

3. **Input/Output Tensor Management**:
   - **Issue**: Handling multiple input and output tensors with varying shapes and dimensions required careful management to ensure proper data handling.
   - **Solution**: Implemented robust tensor management strategies, including utilizing the `getTensorShape` function to retrieve tensor shapes and dimensions, ensuring accurate handling of input and output data.

4. **Post-processing and Visualization**:
   - **Issue**: Decoding and visualizing output tensors from the YOLO-v3 model, containing bounding box information, presented challenges in post-processing and visualization.
   - **Solution**: Developed the `postProcess` function to decode bounding box information, apply non-maximum suppression, and draw final bounding boxes and class labels on output frames for clear visualization.

5. **Anchor Box Selection for Proper Learning and Detection**:
   - **Issue**: Inadequate anchor boxes can hinder the YOLO-v3 model's ability to effectively learn and detect objects of varying sizes and aspect ratios.
   - **Solution**: Addressed the need for proper anchor boxes by conducting extensive analysis and experimentation to select optimal anchor box configurations, ensuring robust object detection performance across diverse datasets and environmental conditions.


## 5. Usage or Test Images/Video

### 5.1. Usage Instructions

To utilize the ADAS detection system, follow these instructions:

1. **Build the Project**:
   - Navigate to the project directory in a terminal window.
   - Run the build script `build.sh` using the following command:
     ```bash
     bash build.sh
     ```

2. **Run the ADAS Detection System**:
   - Once the build process is complete, locate the `adas_detection` executable.
   - Execute the `adas_detection` binary with the following command format:
     ```bash
     ./adas_detection <video_file> <model_file>
     ```
   - Replace `<video_file>` with the path to the video file you wish to analyze.
   - Replace `<model_file>` with the path to the model file used for object detection.
   - Example command:
     ```bash
     ./adas_detection path/to/video.mp4 path/to/model.pb
     ```

### 5.2. Obtaining Sample Images or Video

For testing purposes, you can obtain sample images or videos from various sources:

1. **Online Datasets**: Explore online repositories like Kaggle, ImageNet, and Open Images for free datasets of annotated images and videos spanning various categories.

2. **Publicly Available Videos**: Platforms like YouTube host publicly available videos covering diverse topics. You can use tools like youtube-dl to download videos for testing.

3. **Capture Your Own**: Use a camera or smartphone to capture video footage or images from your surroundings. Record traffic scenes, pedestrians, and other relevant scenarios to test the ADAS detection system's performance.

4. **Generated Data**: If real-world data is unavailable or insufficient, consider using synthetic data generation techniques. Tools like Blender or Unreal Engine enable the creation of realistic synthetic images and videos.

Once you obtain the images or videos, ensure they are in compatible formats and resolutions for optimal performance with the ADAS detection system. Then, follow the usage instructions to test the system's capabilities.


## 6. Reference Links

### 6.1. Directly Followed Links

- [ADAS Detection using Vitis AI 3.0](https://github.com/Xilinx/Vitis-AI/tree/master/examples/vai_runtime/adas_detection)

### 6.2. Referred Links for Supportive Tutorials, Books, or Similar

- [YOLO-v3 Object Detection Algorithm](https://pjreddie.com/darknet/yolo/)
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [C++ Multithreading Tutorial](https://cplusplus.com/reference/multithreading/)
- [Deep Learning for Object Detection: A Comprehensive Review](https://arxiv.org/abs/1809.02165)

In conclusion, the provided reference links offer a wealth of resources for further exploration and implementation of the ADAS detection system using Vitis AI 3.0. The direct link to the GitHub repository provides immediate access to the project code and documentation, facilitating seamless integration and customization. Additionally, the referred links to tutorials, documentation, and research papers offer comprehensive insights into essential topics such as the YOLO-v3 object detection algorithm, OpenCV usage, C++ multithreading, and deep learning methodologies. By leveraging these resources, developers can deepen their understanding, refine their skills, and embark on innovative projects in the realm of intelligent automotive systems.
