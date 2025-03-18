import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from utils import gaussian_smooth,load_prep_data

from sklearn.preprocessing import StandardScaler
from utils import get_life_inputs, convert_to_binary

def load_model(filepath: str, age: int):
    """
    Load a saved PyTorch model from disk and set it to evaluation mode.

    Args:
        filepath (str): Path to the saved model file (.pt or .pth format)
        age (int): Age of the model to initialize the NeuralNet class

    Returns:
        torch.nn.Module: Loaded PyTorch model in evaluation mode

    Example:
        >>> model = load_model('models/25.pth', 25)
    """
    model = NeuralNet(age)
    
    # Allowlist the pandas Index constructor
    from pandas.core.indexes.base import _new_Index
    torch.serialization.add_safe_globals([_new_Index])
    
    # Load the saved data
    saved_data = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(saved_data['state_dict'])  
    model.cols = pd.Index(saved_data['cols'])
    model.eval() 
    print(f"Model loaded from {filepath}")
    return model

class NeuralNet(nn.Module):
    def __init__(self,age):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(23, 10)  # First hidden layer with 64 neurons
        #self.fc2 = nn.Linear(64, 128) # Second hidden layer with 128 neurons
        self.fc2 = nn.Linear(10, int(96-(age-25))) # Output layer with 95 neurons
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax for multi-class classification
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()  # Use for classification
        self.batch_size = 32
        self.scaler = StandardScaler()
        self.cols = []

    def forward(self, x):
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc2(x)
        return self.softmax(x)  # Use softmax if classification

    def save_model(self):
        """Saves the neural network model."""
        age = 25 + 96 - self.fc2.out_features
        filepath = 'models/' + str(age) + '.pth'
        torch.save({
            'state_dict': self.state_dict(),
            'cols': list(self.cols)  
        }, filepath)
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
        X_train, X_test, y_train, y_test, self.scaler, self.cols = load_prep_data('data.csv', 25 + (96 - self.fc2.out_features))
        
        self.scaler.fit(X_train)  # Fit the scaler on the training data
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        for i in range(reps):
            self.neural_net_train(train_loader, epoch)  
            if eval_always or i == reps:
                self.neural_net_eval(test_loader)
            self.save_model()
        return test_loader  
    def get_life_data(self, inputs=None, is_tensor=False, smooth=False, sigma=5):
        if inputs is None:
            # Get inputs
            inputs = get_life_inputs()
        if is_tensor == False:
            # Prep Inputs
            inputs = pd.DataFrame(inputs, columns=self.cols)  # Use self.cols
            for i, col in enumerate(inputs.select_dtypes(include=['object']).columns):
                inputs[col] = inputs[col].apply(lambda x: convert_to_binary(x))
            self.scaler.fit(inputs)  # Fit the scaler on the inputs
            inputs = self.scaler.transform(inputs)
            tensor_input = torch.tensor(inputs, dtype=torch.float32)
        else:
            tensor_input = inputs
        # Get model predictions
        self.eval()
        with torch.no_grad():
            output = self(tensor_input)
        output = pd.DataFrame(output.numpy())
        output = output.transpose()
        output.index = [str(i) for i in range(25 + (96 - self.fc2.out_features), 121)]
        if smooth:
            return gaussian_smooth(output, sigma)
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
    for age in range(25, age_cap):
        model = NeuralNet(age)
        model.train_eval_save(2, 1, True)

if __name__ == "__main__":
    # from utils import plot_mort
    # make_all_models(26)
    # model=load_model('models/25.pth')
    # mort_df=model.get_life_data([[180,'m',72,130,'n','n',3,1,1,'n','n','n',4,'n',0,'n','n',200,'n','n','n','n','n']])
    # plot_mort(mort_df)
    # print(mort_df)
    # smoothed_df = gaussian_smooth(mort_df, sigma=10)
    # #print(smoothed_df.sum())
    # plot_mort(smoothed_df)
    # #print(smoothed_df)
    make_all_models(80)

