import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(23, 10)  # First hidden layer with 64 neurons
        #self.fc2 = nn.Linear(64, 128) # Second hidden layer with 128 neurons
        self.fc2 = nn.Linear(10, 95) # Output layer with 95 neurons
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax for multi-class classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc2(x)
        return self.softmax(x)  # Use softmax if classification

# Example usage
model = NeuralNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Use for classification

# Example input (batch of 10 samples, each with 23 features)
example_input = torch.randn(1, 23)
output = model(example_input)
print(output)
print(output.shape)  # Should be (1, 95)