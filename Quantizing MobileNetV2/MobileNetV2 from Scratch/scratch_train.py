import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model_scratch import MobileNetV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""# Load Filtered Dataset"""
filtered_dataset = torch.load("filtered_dataset.pth")

"""# Random split filtered dataset""" 
from torch.utils.data import random_split

# Define the proportions for train and test split
train_ratio = 0.8
test_ratio = 1 - train_ratio

# Calculate the sizes for train and test datasets
train_size = int(train_ratio * len(filtered_dataset))
test_size = len(filtered_dataset) - train_size

# Split the dataset into train and test subsets
train_dataset, test_dataset = random_split(filtered_dataset, [train_size, test_size])

# Verify the sizes of the train and test datasets
print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))


"""DataLoader"""
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)


"""Train the model"""
# Instantiate the model
model = MobileNetV2().to(device)
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# Define the loss function
criterion = nn.CrossEntropyLoss()

from tqdm import tqdm
import time

num_epochs = 10 # For less computation

for epoch in tqdm(range(num_epochs), desc='Epochs'):
    model.train()  # Set the model to training mode

    start_time = time.time() # Start time of the epoch

    running_loss = 0.0
    running_corrects = 0

    # Iterate over the training data in batches
    for inputs, labels in train_loader:
        # Move the inputs and labels to the selected device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)

        torch.cuda.empty_cache() # Limit GPU memory growth

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache() # Limit GPU memory growth

        # Update running loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1) # the maximum value and the index of that maximum value along dimension 1
        running_corrects += torch.sum(preds == labels.data) # labels.data gives access to underlying labels tensor

    end_time = time.time()  # End time of the epoch
    epoch_duration = end_time - start_time  # Duration of the epoch

    # Print the epoch duration
    tqdm.write(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

    # Calculate epoch loss and accuracy for training data
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    # Print the loss and accuracy for training and validation data
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")


"""Testing the model"""
correct = 0
total = 0

with torch.no_grad(): # Disabling gradient calculation, since testing doesnot require weight update
    model.eval() # Set the model to evaluation mode

    for inputs, labels in test_loader:
      # Move the inputs and labels to the selected device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


"""Save the model"""
save_path = "MobileNetV2_scratch.pth"
torch.save(model.state_dict(), save_path)

print("Trained model saved at:", save_path)