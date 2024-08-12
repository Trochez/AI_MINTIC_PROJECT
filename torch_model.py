import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


class RockPaperScissorsDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        
        # List all image files
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_name = os.path.splitext(image_file)[0]  # Get the base name without extension
        
        # Load image
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        # Load corresponding label
        label_file = os.path.join(self.labels_dir, f"{image_name}.txt")
        with open(label_file, 'r') as file:
            label_line = file.readline().strip()
            
        # Skip empty label files
        if not label_line:
            return None
        
        # Extract the first character before the first space
        label = label_line.split(' ')[0]
        
        # Convert label to an integer
        label_mapping = {'0': 0, '1': 1, '2': 2}  # Assuming '0' for Paper, '1' for Rock, '2' for Scissors
        label = label_mapping[label]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def custom_collate(batch):
    # Filter out None values from the batch
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize images to 640x640 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define directories
images_dir = 'project_git/Rock Paper Scissors SXSW.v14i.yolov8/train/images'
labels_dir = 'project_git/Rock Paper Scissors SXSW.v14i.yolov8/train/labels'

# Create dataset and dataloader
dataset = RockPaperScissorsDataset(images_dir=images_dir, labels_dir=labels_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

# Define the model architecture
# Define the model architecture (same as before)
class RockPaperScissorsModel(nn.Module):
    def __init__(self):
        super(RockPaperScissorsModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)

        # Dummy tensor to calculate the size after convolutions and pooling
        x = torch.randn(1, 3, 640, 640)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        self.flattened_size = x.view(-1).size(0)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 3)  # Output for 3 classes

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, self.flattened_size)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = RockPaperScissorsModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with verbose output
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Verbose output
        if i % 10 == 9:  # Print every 10 batches
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

print('Training completed!')

# Save the model
torch.save(model.state_dict(), 'rock_paper_scissors_model.pt')

# Instantiate the model and load the trained weights
model = RockPaperScissorsModel()
model.load_state_dict(torch.load('rock_paper_scissors_model.pt'))
model.eval()  # Set the model to evaluation mode


# Load the image
image_path = 'project_git/2024-08-10-204059.jpg'
image = Image.open(image_path).convert('RGB')

# Apply the transformations
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Predict the class
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1)

# Convert prediction to class label
label_mapping = {0: 'rock', 1: 'paper', 2: 'scissors'}
predicted_class = label_mapping[prediction.item()]

print(f'Predicted class: {predicted_class}')