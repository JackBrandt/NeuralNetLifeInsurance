import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(23, 10)  # First hidden layer with 64 neurons
        #self.fc2 = nn.Linear(64, 128) # Second hidden layer with 128 neurons
        self.fc2 = nn.Linear(10, 145) # Output layer with 95 neurons
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

# Load CSV file
df = pd.read_csv("data.csv", header=0)

# Extract target (y) and features (X)
empty=[0]*145
y_vals = df.iloc[:, 0].values  # First column is target
y_vals=[value-23 for value in y_vals]
y=[empty.copy() for _ in y_vals]
for i,y_val in enumerate(y_vals):
    y[i][y_val]=1
#print(y_vals[0])
#print(y[0][72])
X = df.iloc[:, 1:].copy()  # Everything else is features

# Convert categorical columns to numerical values
label_encoders = {}  # Store encoders for inverse transform later if needed
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])  # Convert categories to numbers
    label_encoders[col] = le  # Save encoder for future use

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Standardize input features

# Convert to PyTorch tensors
#print(y)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)  # Use long for classification

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create PyTorch DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#example_input = torch.randn(1, 23)
#output = model(example_input)
#print(output)
#print(output.shape)  # Should be (1, 95)

# Training loop
num_epochs = 1  # Set number of epochs
for epoch in range(num_epochs):
    running_loss = 0.0
    batch_counter=1
    for inputs, labels in train_loader:
        print(batch_counter)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()
        batch_counter+=1
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()  # Set to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        #print(outputs)
        #_, predicted = torch.max(outputs, 1)  # Get class with highest probability
        total += labels.size(0)
        correct += (outputs == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("Training complete!")