import torch
from torchvision import transforms
from PIL import Image

# Assuming the model architecture matches the one used during training
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define layers here (should match the saved model's architecture)
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1)
        self.fc1 = torch.nn.Linear(16*26*26, 128)
        self.fc2 = torch.nn.Linear(128, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
#model = MyModel()
model = torch.load('./game_hands.pt')
model.eval()  # Set the model to evaluation mode

# Assuming you have an image to predict
image_path = './2024-08-10-204059.jpg'
image = Image.open(image_path)

# Preprocessing the image (ensure it matches the model's training preprocessing)
preprocess = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to match the input size
    transforms.ToTensor(),        # Convert image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Apply preprocessing
image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Model inference
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1)

# Print the prediction
print(f'Predicted class: {prediction.item()}')
