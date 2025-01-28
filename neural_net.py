import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()  # Use for classification

# Example input (batch of 10 samples, each with 23 features)
example_input = torch.randn(1, 23)
output = model(example_input)
print(output)
print(output.shape)  # Should be (1, 95)

# Generate dummy dataset (replace with real data)
X_train = torch.randn(10, 23)  # 1000 samples, 23 features
y_train = torch.randint(0, 95, (10,))  # 1000 labels (for classification)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
num_epochs = 3  # Set number of epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")