import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# If missing libraries like sklearn, please install them first:
# !pip install scikit-learn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def convert_to_binary(labels, threshold=None):
    """
    Convert label data into binary categories (0/1).
    - If labels are continuous values, they are binarized based on a threshold (using the median as default).
    - If labels are categorical binary types (e.g., Yes/No), they are mapped to 0/1.
    """
    labels = np.array(labels)
    # If labels are non-numeric (object or string), try mapping to numeric
    if labels.dtype == object or labels.dtype == np.str_:
        unique_vals = np.unique(labels)
        if len(unique_vals) == 2:
            # Map binary labels
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            labels = np.array([mapping[x] for x in labels])
        else:
            raise ValueError("Labels provided have more than two categories, cannot convert to binary.")
    # Convert to float for threshold comparison
    labels = labels.astype(float)
    # Default to using median of labels as the threshold, assigning 1 to values >= median, 0 otherwise
    if threshold is None:
        threshold = np.median(labels)
    binary_labels = (labels >= threshold).astype(int)
    return binary_labels

def get_life_inputs(df, target_col):
    """
    Separate feature and target column from DataFrame.
    Returns feature matrix X and target vector y (as numpy arrays).
    """
    assert target_col in df.columns, f"Target column named {target_col} not found in DataFrame"
    X = df.drop(columns=[target_col]).values  # Feature data
    y = df[target_col].values                # Target data
    return X, y

def load_prep_data(file_path, target_col, test_size=0.2, random_state=42):
    """
    Load and preprocess data, split into train/test sets.
    Returns features and labels for training and testing in tensor format: (X_train, X_test, y_train, y_test).
    Parameters:
    - file_path: Path to data file (assumed CSV format).
    - target_col: Name of the target column in the DataFrame.
    - test_size: Proportion of the test set (default 0.2, i.e., 20%).
    - random_state: Random seed for reproducible splits.
    """
    # Read data using pandas
    df = pd.read_csv(file_path)
    # Clean data: drop rows with missing values (missing values could lead to errors during training)
    df = df.dropna().reset_index(drop=True)
    # Separate features and labels
    X, y = get_life_inputs(df, target_col)
    # Convert labels to binary if they are not already 0/1
    unique_y = np.unique(y)
    if not (len(unique_y) == 2 and sorted(unique_y) == [0, 1]):
        y = convert_to_binary(y)
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    # Normalize features (standardization: mean=0, variance=1, to help with convergence)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
    # Convert labels to Long tensor (for classification with CrossEntropyLoss)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor  = torch.tensor(y_test, dtype=torch.long)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

# Define the neural network model
class LifeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        """
        Simple feedforward neural network:
        input_dim -> hidden_dim -> hidden_dim -> output_dim
        """
        super(LifeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        # Forward pass: pass input through defined sequential layers
        return self.net(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    """
    Function to train the model. Evaluate loss and accuracy on validation set after each epoch.
    """
    for epoch in range(1, epochs+1):
        model.train()  # Switch to training mode
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()          # Clear previous gradients
            outputs = model(batch_X)       # Forward pass to get outputs
            loss = criterion(outputs, batch_y)  # Calculate loss
            loss.backward()                # Backpropagate to calculate gradients
            optimizer.step()               # Update network parameters
            running_loss += loss.item() * batch_X.size(0)  # Accumulate total loss (weighted by number of samples)
        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader.dataset)
        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    return model

def evaluate_model(model, data_loader, criterion):
    """
    Evaluate the model on a given dataset, return average loss and accuracy.
    Gradients are not calculated during evaluation.
    """
    model.eval()  # Switch to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():  # Disable gradient calculation
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            # For classification, predict the class with the highest logit
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_X.size(0)
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

def save_model(model, path):
    """Save model parameters to a specified path."""
    torch.save(model.state_dict(), path)

def load_model(model_class, path, *args, **kwargs):
    """Load model parameters and return model instance. Model class and initialization parameters are required."""
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()  # Switch model to evaluation mode
    return model
