import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pandas as pd
from utils import gaussian_smooth

from sklearn.preprocessing import StandardScaler
from utils import get_life_inputs, convert_to_binary
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_recall_curve
from custom_loss_function import AgeMortalityLoss
from utils import load_prep_data, plot_mort, gaussian_smooth


def load_model(filepath="model.pth"):
    """Loads the saved model and returns an instance of it."""
    model = torch.load('model.pth', weights_only=False)
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

class NeuralNet(nn.Module):
    def __init__(self, num_inputs=23, num_hidden=10, num_outputs=96):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)  # First hidden layer with 64 neurons
        self.fc2 = nn.Linear(num_hidden, num_outputs) # Output layer with 95 neurons
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)  # Apply softmax for multi-class classification
        # self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.criterion = nn.CrossEntropyLoss()  # Use for classification
        #     # write a custom loss, crosses 3 losses and something else that go through the prob and forces the 95 and 96 to be similar, kind of like smoothing
        self.batch_size = 32
        # self.scaler = StandardScaler()
        self.cols = []
        self.criterion = AgeMortalityLoss(
            num_ages=96,
            alpha=0.1,
            label_smoothing=0.05,
            heavy_smoothing_region=(70, 75),   # example: treat "older ages" region more carefully
            heavy_smoothing_factor=5.0
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
        x = self.fc2(x)
        # return self.softmax(x)
        return x

    def save_model(self, filepath="model.pth"):
        """Saves the neural network model."""
        torch.save(self, filepath)  # Save model parameters
        print(f"Model saved to {filepath}")

    # Training loop
    def neural_net_train(self, train_loader, epochs=5,print_statement=True):
        '''Trains the nn
        Args:
            epoch (int): number of times to run through the data
            train_loader (DataLoader): the data to train on
        '''
        self.train()
        num_epochs = 1  # Set number of epochs
        for epoch in range(num_epochs):
            running_loss = 0.0
            batch_counter=1
            for inputs, labels in train_loader:
                #print(batch_counter)
                self.optimizer.zero_grad()  # Zero the gradients
                logits = self(inputs)  # Forward pass
                loss = self.criterion(logits, labels)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                running_loss += loss.item()
                batch_counter+=1
            if print_statement:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
        print("Training complete!")

    # Evaluation
    def neural_net_eval(self, test_loader):
        self.eval()  # Set to evaluation mode
        sum_of_mean_absolute_errors=0
        # with torch.no_grad():
        #     for inputs, labels in test_loader:
        #         #print(labels.shape)
        #         outputs=self(inputs)
        #         #print(outputs.shape)
        #         #print(outputs)
        #         #_, predicted = torch.max(outputs, 1)  # Get class with highest probability
        #         for i, output in enumerate(outputs):
        #             single_sum_errors=0
        #             #print(output.shape)
        #             #try: print(labels[i].shape)
        #             #except IndexError: print(i, labels)
        #             #print(output)
        #             for j, value in enumerate(output):
        #                 error=abs(value-labels[i][j])
        #                 single_sum_errors+=error
        #             sum_of_mean_absolute_errors=single_sum_errors/len(output)
        # mean_mean_absolute_error = sum_of_mean_absolute_errors/len(outputs)
        # print(f"Test Mean Mean Absolute Error: {mean_mean_absolute_error}")
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                logits = model(inputs)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}")

    def train_eval_save(self, reps, epoch, eval_always=True):
        '''Trains, evaluates, and saves model:
        Args:
            reps: Number loops of training/eval
            epoch: Number of trainings between evals
            eval_always: Defaults to true, if false doesn't evaluate until final train'''
        X_train, X_test, y_train, y_test, self.scaler, self.cols = load_prep_data('data.csv')
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        for i in range(reps):
            self.neural_net_train(train_loader, epoch)#Change epoch to do more training between evals
            if eval_always or i == reps:
                self.neural_net_eval(test_loader)
            self.save_model()
        return test_loader # For testing loading

    def get_life_data(self, inputs=None, is_tensor=False,smooth=False,sigma=5):
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
            logits = self(tensor_input)
            probs = F.softmax(logits, dim=-1)
        output = pd.DataFrame(probs.numpy())

        # label rows
        output = output.transpose()
        output.index=[str(i) for i in range(25,121)]
        #output.to_csv()
        if smooth:
            return gaussian_smooth(output,sigma)
        else:
            return output

if __name__ == "__main__":
    model = NeuralNet()

    train_loader = model.train_eval_save(reps=1, epoch=500)

    model=load_model(NeuralNet)

    mort_df = model.get_life_data([[180,'m',72,130,'n','n',3,1,1,'n','n','n',4,'n',0,'n','n',200,'n','n','n','n','n']])
    plot_mort(mort_df)
    print(mort_df)
    smoothed_df = gaussian_smooth(mort_df, sigma=10)
    #print(smoothed_df.sum())
    plot_mort(smoothed_df)
    print(smoothed_df)
    print(smoothed_df.sum())
    smoothed_df.to_csv('mortality.csv')
    #print(model.get_life_data())
    # TODO: Create neural nets for people over 25,
    #  that is train a neural net on the mortality rates, given the person doesn't die when 25
    # then repeat for the next year up and so on, this means a lot of neural nets to train/store/use
    # but we probably don't need to do every possible neural net, i.e., we can stop once a certain age is reached

