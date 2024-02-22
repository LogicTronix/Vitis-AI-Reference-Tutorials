import sys
import xir
import vart
import time
from typing import List
from ctypes import *
import random

import cv2
import importlib
import numpy as np

from dpu_utils import YOLOPost, non_max_suppression

def runYolo(dpu_runner_tfYolo, image, config, image_path):
    config = config

    inputTensors = dpu_runner_tfYolo.get_input_tensors()  #  get the model input tensor
    outputTensors = dpu_runner_tfYolo.get_output_tensors() # get the model ouput tensor
    
    outputHeight_0 = outputTensors[0].dims[1]
    outputWidth_0 = outputTensors[0].dims[2]
    outputChannel_0 = outputTensors[0].dims[3]

    outputHeight_1 = outputTensors[1].dims[1]
    outputWidth_1 = outputTensors[1].dims[2]
    outputChannel_1 = outputTensors[1].dims[3]

    outputHeight_2 = outputTensors[2].dims[1]
    outputWidth_2 = outputTensors[2].dims[2]
    outputChannel_2 = outputTensors[2].dims[3]    

    outputSize_0 = (outputHeight_0,outputWidth_0,outputChannel_0)
    outputSize_1 = (outputHeight_1,outputWidth_1,outputChannel_1)
    outputSize_2 = (outputHeight_2,outputWidth_2,outputChannel_2)


    runSize = 1
    shapeIn = (runSize,) + tuple([inputTensors[0].dims[i] for i in range(inputTensors[0].ndim)][1:])

    # InputTensor[0]: {name: 'ModelMain__input_0_fix', shape: [1, 416, 416, 3], type: 'xint8', attrs: {'location': 1, 'ddr_addr': 1264, 'bit_width': 8, 'round_mode': 'DPU_ROUND', 'reg_id': 2, 'fix_point': 4, 'if_signed': True}}

    '''prepare batch input/output '''
    outputData = []
    inputData = []
    outputData.append(np.empty((runSize,outputHeight_0,outputWidth_0,outputChannel_0), dtype = np.float32, order = 'C'))
    outputData.append(np.empty((runSize,outputHeight_1,outputWidth_1,outputChannel_1), dtype = np.float32, order = 'C'))
    outputData.append(np.empty((runSize,outputHeight_2,outputWidth_2,outputChannel_2), dtype = np.float32, order = 'C'))
    inputData.append(np.empty((shapeIn), dtype = np.float32, order = 'C'))

    '''init input image to input buffer '''
    imageRun = inputData[0]
    imageRun[0,...] = image.reshape(inputTensors[0].dims[1],inputTensors[0].dims[2],inputTensors[0].dims[3])

    '''Execute Async'''
    job_id = dpu_runner_tfYolo.execute_async(inputData,outputData)
    dpu_runner_tfYolo.wait(job_id)

    '''Transpose Output for consistency'''
    outputData[0] = np.transpose(outputData[0], (0,3,1,2))
    outputData[1] = np.transpose(outputData[1], (0,3,1,2))
    outputData[2] = np.transpose(outputData[2], (0,3,1,2))


    '''Post Processing'''
    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOPost(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    output_list = []
    for i in range(3):
        output_list.append(yolo_losses[i].forward(outputData[i]))
    output_con = np.concatenate(output_list, 1)


    # Perform Non Max Supression
    batch_detections = non_max_suppression(output_con, config["yolo"]["classes"],
                                                   conf_thres=config["confidence_threshold"],
                                                   nms_thres=0.45)
    print(batch_detections)


    '''Plot prediction with bounding box'''
    classes = open(config["classes_names_path"], "r").read().split("\n")[:-1]

    for idx, detections in enumerate(batch_detections):
        if detections is not None:
            im = cv2.imread(image_path)
            unique_labels = np.unique(detections[:, -1])
            n_cls_preds = len(unique_labels)
            bbox_colors = {int(cls_pred): (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls_pred in unique_labels}
            
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                color = bbox_colors[int(cls_pred)]

                # Rescale coordinates to original dimensions
                ori_h, ori_w, _ = im.shape
                pre_h, pre_w = config["img_h"], config["img_w"]
                box_h = ((y2 - y1) / pre_h) * ori_h
                box_w = ((x2 - x1) / pre_w) * ori_w
                y1 = (y1 / pre_h) * ori_h
                x1 = (x1 / pre_w) * ori_w

                # Create a Rectangle patch
                cv2.rectangle(im, (int(x1), int(y1)), (int(x1 + box_w), int(y1 + box_h)), color, 2)

                # Add label
                label = classes[int(cls_pred)]
                cv2.putText(im, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # # Save generated image with detections
        # output_path = 'prediction.jpg'
        # cv2.imwrite(output_path, im)

        # Display image
        cv2.imshow("Prediction", im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."

    root_subgraph = graph.get_root_subgraph() # Retrieves the root subgraph of the input 'graph'
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    
    if root_subgraph.is_leaf:
        return [] # If it is a leaf, it means there are no child subgraphs, so the function returns an empty list 
    
    child_subgraphs = root_subgraph.toposort_child_subgraph() # Retrieves a list of child subgraphs of the 'root_subgraph' in topological order
    assert child_subgraphs is not None and len(child_subgraphs) > 0

    return [
        # List comprehension that filters the child_subgraphs list to include only those subgraphs that represent DPUs
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def main(argv):
    
    g = xir.Graph.deserialize(argv[1]) # Deserialize the DPU graph
    subgraphs = get_child_subgraph_dpu(g) # Extract DPU subgraphs from the graph
    assert len(subgraphs) == 1  # only one DPU kernel

    """Creates DPU runner, associated with the DPU subgraph."""
    dpu_runners = vart.Runner.create_runner(subgraphs[0], "run")


    # Get config
    params_path = "params.py"
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS

    # Preprocessing 
    image_path = argv[2]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config["img_w"], config["img_h"]),
                                interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image /= 255.0

    # Measure time 
    time_start = time.time()

    """Assigns the runYolo function with corresponding arguments"""
    runYolo(dpu_runners, image, config, image_path)
    del dpu_runners

    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = 1 
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage : python3 dpu_inference.py <xmodel_file> <image_path>")
    else:
        main(sys.argv)