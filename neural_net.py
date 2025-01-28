import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_prep_data(file_path):
    '''Loads and preps data...
    Args:
        file_path (str):...
    Returns:
        X (array of arrays)
        y (array of arrays)
    '''
    # Load CSV file
    df = pd.read_csv(file_path, header=0)

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

    return X,y

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
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # Use for classification

X,y = load_prep_data('data.csv')

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


# Training loop
def neural_net_train(epoch=1):
    '''Trains the nn
    Args:
        epoch (int): number of times to run through the data'''
    model.train()
    num_epochs = 1  # Set number of epochs
    for epoch in range(num_epochs):
        running_loss = 0.0
        batch_counter=1
        for inputs, labels in train_loader:
            #print(batch_counter)
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()
            batch_counter+=1
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
    print("Training complete!")

# Evaluation
def neural_net_eval():
    model.eval()  # Set to evaluation mode
    sum_of_mean_absolute_errors=0
    with torch.no_grad():
        for inputs, labels in test_loader:
            #print(labels.shape)
            outputs=model(inputs)
            #print(outputs.shape)
            #print(outputs)
            #_, predicted = torch.max(outputs, 1)  # Get class with highest probability
            for i, output in enumerate(outputs):
                single_sum_errors=0
                #print(output.shape)
                #try: print(labels[i].shape)
                #except IndexError: print(i, labels)
                #print(output)
                for j, value in enumerate(output):
                    error=abs(value-labels[i][j])
                    single_sum_errors+=error
                sum_of_mean_absolute_errors=single_sum_errors/len(output)
    mean_mean_absolute_error = sum_of_mean_absolute_errors/len(outputs)
    print(f"Test Mean Mean Absolute Error: {mean_mean_absolute_error}")

if __name__ == "__main__":
    for _ in range(3):
        neural_net_train(epoch=1)#Change epoch to do more training between evals
        neural_net_eval()