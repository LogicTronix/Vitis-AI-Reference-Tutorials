import torch
import torch.nn as nn
from pytorch_nndct.apis import Inspector

from torchvision.models.mobilenet import mobilenet_v2

target = "DPUCZDX8G_ISA1_B4096"
# Initialize inspector with target
inspector = Inspector(target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate MobileNetV2 model
model = mobilenet_v2(pretrained=True)

# Access the classifier of the model
classifier = model.classifier

# Remove the last layer from the classifier
classifier = classifier[:-1]

# Add a new linear layer at the end of the classifier
new_linear_layer = nn.Linear(in_features=1280, out_features=3)
classifier.add_module('new_linear', new_linear_layer)

# Assign the modified classifier back to the model
model.classifier = classifier

# Load the model to be inspected
model.load_state_dict(torch.load("MobileNetV2_transfer_learning.pth", map_location=torch.device('cpu')))

# Random input
dummy_input = torch.randn(1,3,224,224)

# Start to inspect
inspector.inspect(model, (dummy_input,), device=device, output_dir="inspect", image_format="png") 