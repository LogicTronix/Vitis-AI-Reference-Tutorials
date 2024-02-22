import torch
import torch.nn as nn
from pytorch_nndct.apis import Inspector

from model_scratch import MobileNetV2

target = "DPUCZDX8G_ISA1_B4096"
# Initialize inspector with target
inspector = Inspector(target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate MobileNetV2 model
model = MobileNetV2()

# Load the model to be inspected
model.load_state_dict(torch.load("MobileNetV2_scratch.pth", map_location=torch.device(device)))

# Random input
dummy_input = torch.randn(1,3,224,224)

# Start to inspect
inspector.inspect(model, (dummy_input,), device=device, output_dir="inspect_scratch", image_format="png") 