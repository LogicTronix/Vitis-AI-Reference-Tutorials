import torch
import torch.nn as nn
from torchvision import transforms

from PIL import Image

import pytorch_nndct

def predict(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load('quantize_result/EfficientNet_int.pt', map_location=torch.device(device))

    model.eval()
    model = model.to(device)

    transform = transforms.Compose([transforms.Resize((224,224)), #<-- Resize for ResNet input
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    input_image = transform(image).unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        input_image = input_image.to(device)
        output = model(input_image)

    _, predicted = torch.max(output.data, 1)

    # Get the predicted class probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    # Get the predicted class label
    predicted_class = torch.argmax(probabilities).item()
    
    return probabilities, predicted_class

# Test on airplane
image_path = "airplane.png"
image = Image.open(image_path)
image.show()

probabilities, predicted_class = predict(image)

class_labels = ['airplane', 'automobile', 'bird']

# Print the predicted class label and corresponding probability
print("Predicted class:", class_labels[predicted_class])
print("Probability:", probabilities[predicted_class].item())
print()


# Test on automobile
image_path = "automobile.png"
image = Image.open(image_path)
image.show()

probabilities, predicted_class = predict(image)

class_labels = ['airplane', 'automobile', 'bird']

# Print the predicted class label and corresponding probability
print("Predicted class:", class_labels[predicted_class])
print("Probability:", probabilities[predicted_class].item())
print()


# Test on bird
image_path = "bird.png"
image = Image.open(image_path)
image.show()

probabilities, predicted_class = predict(image)

class_labels = ['airplane', 'automobile', 'bird']

# Print the predicted class label and corresponding probability
print("Predicted class:", class_labels[predicted_class])
print("Probability:", probabilities[predicted_class].item())
print()