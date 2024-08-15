import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import zlib
import pickle
import torch.quantization
import torch.nn.utils.prune as prune

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize images to 640x640 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model architecture (same as used during training)
class RockPaperScissorsModel(torch.nn.Module):
    def __init__(self):
        super(RockPaperScissorsModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)

        # Dummy tensor to calculate the size after convolutions and pooling
        x = torch.randn(1, 3, 640, 640)
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        self.flattened_size = x.view(-1).size(0)

        self.fc1 = torch.nn.Linear(self.flattened_size, 128)
        self.fc2 = torch.nn.Linear(128, 3)  # Output for 3 classes

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(-1, self.flattened_size)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Step 1: Initialize the model
model = RockPaperScissorsModel()

# Step 2: Apply quantization to the model
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# Step 3: Reapply pruning to the same layers
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)  # Adjust amount to match training

# Step 4: Load and decompress the state dictionary
with open('compressed_model.pt', 'rb') as f:
    compressed_state_dict = f.read()

state_dict = pickle.loads(zlib.decompress(compressed_state_dict))

# Step 5: Load the state dictionary into the model
model.load_state_dict(state_dict)

# Step 6: Set the model to evaluation mode
model.eval() 

"""
# Initialize the model
model = RockPaperScissorsModel()

# Load the trained weights
model.load_state_dict(torch.load("./rock_paper_scissors_model.pt"))
model.eval()  # Set the model to evaluation mode"""

def predict_image(image_stream):
    # Load the image from the stream
    image = Image.open(image_stream).convert('RGB')

    # Apply the transformations
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict the class
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.argmax(output, dim=1)

    # Convert prediction to class label
    label_mapping = {0: 'paper', 1: 'rock', 2: 'scissors'}
    predicted_class = label_mapping[prediction.item()]
    
    return predicted_class
