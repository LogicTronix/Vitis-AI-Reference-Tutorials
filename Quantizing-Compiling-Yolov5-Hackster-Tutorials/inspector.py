import torch
import os
from pytorch_nndct.apis import Inspector
from models.common import DetectMultiBackend


# Model load
# model = YOLO('models/yolov8m.pt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = 'runs/train/exp4/weights/epoch10.pt'
dnn = False
data = 'data/bdd.yaml'


model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)


# Target
target = "DPUCZDX8G_ISA1_B4096"

# Initialize inspector with target
inspector = Inspector(target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Random input
dummy_input = torch.randn(1,3,640,640)
# Start to inspect
inspector.inspect(model, (dummy_input,), device=device, output_dir="inspect", image_format="png")

