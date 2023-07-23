import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
import matplotlib.pyplot as plt

from torchvision.models.mobilenet import mobilenet_v2

def predict(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    model.load_state_dict(torch.load("MobileNetV2.pth", map_location=torch.device('cpu')))
    model.eval()
    model = model.to(device)

    transform = transforms.Compose([transforms.Resize((224,224)),
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