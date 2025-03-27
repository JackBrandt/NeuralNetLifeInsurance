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
class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=2):
        """
        Simple feedforward neural network:
        input_dim -> hidden_dim -> hidden_dim -> output_dim
        """
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc2(x)
        return self.softmax(x)  # Use softmax if classification

    def save_model(self):
        """Saves the neural network model."""
        age=25+96-self.fc2.out_features
        filepath='models/'+str(age)+'.pth'
        torch.save(self, filepath)  # Save model parameters
        print(f"Model saved to {filepath}")

    # Training loop
    def neural_net_train(self,train_loader, epoch=1, print_statement=True):
        '''Trains the nn
        Args:
            epoch (int): number of times to run through the data'''
        self.train()
        num_epochs = 1  # Set number of epochs
        for epoch in range(num_epochs):
            running_loss = 0.0
            batch_counter=1
            for inputs, labels in train_loader:
                #print(batch_counter)
                self.optimizer.zero_grad()  # Zero the gradients
                outputs = self(inputs)  # Forward pass
                loss = self.criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                running_loss += loss.item()
                batch_counter+=1
            if print_statement:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
        print("Training complete!")

    # Evaluation
    def neural_net_eval(self,test_loader):
        self.eval()  # Set to evaluation mode
        sum_of_mean_absolute_errors=0
        with torch.no_grad():
            for inputs, labels in test_loader:
                #print(labels.shape)
                outputs=self(inputs)
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

    def train_eval_save(self, reps, epoch, eval_always=True):
        '''Trains, evaluates, and saves model:
        Args:
            reps: Number loops of training/eval
            epoch: Number of trainings between evals
            eval_always: Defaults to true, if false doesn't evaluate until final train'''
        X_train, X_test, y_train, y_test, self.scaler, self.cols = load_prep_data('data.csv',25+(96-self.fc2.out_features))
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        for i in range(reps):
            self.neural_net_train(train_loader,epoch)#Change epoch to do more training between evals
            if eval_always or i== reps:
                self.neural_net_eval(test_loader)
            self.save_model()
        return test_loader # For testing loading

    def train_save(self, reps, epoch):
        '''Trains, evaluates, and saves model:
        Args:
            reps: Number loops of training/eval
            epoch: Number of trainings between evals
        '''
        X_train, y_train, self.scaler, self.cols = load_prep_data(25+(96-self.fc2.out_features),test=False)
        print(f'Len X_train: {len(X_train)}')
        print(0.001*((10000/len(X_train))**(.1)))
        self.optimizer = optim.Adam(self.parameters(), lr=0.001*((10000/len(X_train))**(.5)))# Adjust learning speed using training data size
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        for i in range(reps):
            self.neural_net_train(train_loader,epoch)#Change epoch to do more training between evals
            self.save_model()

    def get_life_data(self, inputs=None, is_tensor=False,smooth_percentage=0,sigma=5):
        if inputs is None:
            # Get inputs
            inputs=get_life_inputs()
        if is_tensor==False:
            # Prep Inputs
            inputs=pd.DataFrame(inputs,columns=self.cols)
            for i,col in enumerate(inputs.select_dtypes(include=['object']).columns):
                inputs[col] = inputs[col].apply(lambda x: convert_to_binary(x))
            inputs=self.scaler.transform(inputs)
            tensor_input=torch.tensor(inputs, dtype=torch.float32)
        else:
            tensor_input=inputs
        # Get model predictions
        self.eval()
        with torch.no_grad():
            output = self(tensor_input)
        output = pd.DataFrame(output.numpy())
        output = output.transpose()
        output.index=[str(i) for i in range(25+(96-self.fc2.out_features),121)]
        #output.to_csv()
        if smooth_percentage>0:
            smooth_percentage=smooth_percentage/100
            return smooth_percentage*gaussian_smooth(output,sigma)+(1-smooth_percentage)*output
        else:
            return output

def make_all_models(age_cap: int):
    """
    Generate and train NeuralNet models for ages 25-age_cap with fixed training parameters.

    Creates a series of models for each age from 25 to age_cap-1 (inclusive), trains them using
    hardcoded training parameters, and saves them via train_eval_save().

    Args:
        age_cap (int): non-included age cap

    Returns:
        None: Models are saved but not returned directly

    Example:
        >>> make_all_models(80)

    Note:
        - Uses fixed training params: 2 epochs, 1 batch_size, save=True
        - Sequential execution may take substantial time for 55 models
    """
    for age in range(25,age_cap):
        model=NeuralNet(age)
        model.train_save(2,1)

if __name__ == "__main__":
    
    from utils import plot_mort
    make_all_models(80)
    model=load_model('models/25.pth')
    mort_df=model.get_life_data([[180,'m',72,130,'n','n',3,1,1,'n','n','n',4,'n',0,'n','n',200,'n','n','n','n','n']])
    plot_mort(mort_df)
    print(mort_df)
    smoothed_df = mort_df=model.get_life_data([[180,'m',72,130,'n','n',3,1,1,'n','n','n',4,'n',0,'n','n',200,'n','n','n','n','n']],smooth_percentage=100,)
    #print(smoothed_df.sum())
    plot_mort(smoothed_df)
    #print(smoothed_df)

