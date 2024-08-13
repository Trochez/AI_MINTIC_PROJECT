import torch.nn.utils.prune as prune
import torch
import zlib
import pickle
import torch.nn as nn
import torch.quantization


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
    
# Step 1: Initialize the model
model = RockPaperScissorsModel()
model.load_state_dict(torch.load('./rock_paper_scissors_model.pt'))


# Step 6: Set the model to evaluation mode
model.eval()

model_fp32 = model  # Assuming you have the pruned model here
model_int8 = torch.quantization.quantize_dynamic(model_fp32, {torch.nn.Linear}, dtype=torch.qint8)

# Apply pruning to convolutional and linear layers
for name, module in model_int8.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.2)  # Adjust amount as needed

model_int8 = model_int8.half()  # Convert the model to half precision (float16)


# Save the model's state dictionary in half precision
state_dict = model_int8.state_dict()

# Serialize and compress the state dictionary
compressed_state_dict = zlib.compress(pickle.dumps(state_dict), level=9)

# Save the compressed state dictionary to a file
with open('compressed_model.pt', 'wb') as f:
    f.write(compressed_state_dict)

