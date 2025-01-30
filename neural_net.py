import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import get_life_inputs, convert_to_binary

def load_model(model_class, filepath="model.pth"):
    """Loads the saved model and returns an instance of it."""
    model = model_class()  # Create a new instance of the model
    model.load_state_dict(torch.load(filepath))  # Load saved parameters
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(23, 10)  # First hidden layer with 64 neurons
        #self.fc2 = nn.Linear(64, 128) # Second hidden layer with 128 neurons
        self.fc2 = nn.Linear(10, 145) # Output layer with 95 neurons
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax for multi-class classification
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()  # Use for classification

    def forward(self, x):
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc2(x)
        return self.softmax(x)  # Use softmax if classification

    def save_model(self, filepath="model.pth"):
        """Saves the neural network model."""
        torch.save(self.state_dict(), filepath)  # Save model parameters
        print(f"Model saved to {filepath}")

    # Training loop
    def neural_net_train(self, epoch=1):
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
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
        print("Training complete!")

    # Evaluation
    def neural_net_eval(self):
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
        for i in range(reps):
            model.neural_net_train(epoch=epoch)#Change epoch to do more training between evals
            if eval_always or i== reps:
                model.neural_net_eval()
            model.save_model()

def print_life_data(cols):
    # Get inputs
    inputs=pd.DataFrame([get_life_inputs()],columns=cols)
    # Prep Inputs
    # Convert categorical columns to numerical values
    print(inputs)
    for i,col in enumerate(inputs.select_dtypes(include=['object']).columns):
        inputs[col] = inputs[col].apply(lambda x: convert_to_binary(x))
    print(inputs)
    inputs=scaler.transform(inputs)
    tensor_input=torch.tensor(inputs, dtype=torch.float32)
    model.eval()
    # Get model predictions
    with torch.no_grad():  # No need for gradient tracking
        outputs = model(tensor_input)
    print(outputs)

if __name__ == "__main__":
    from utils import load_prep_data
    # Example usage
    model = NeuralNet()

    # Split into train and test sets
    X_train, X_test, y_train, y_test, scaler, cols = load_prep_data('data.csv')
    #print(scaler.get_params)
    #print(label_encoders)
    # Create PyTorch DataLoader
    batch_size = 32
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #model.train_eval_save(1,1,True)
    model=load_model(NeuralNet)
    #model.neural_net_eval()
    print_life_data(cols)