import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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
                optimizer.zero_grad()  # Zero the gradients
                outputs = self(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

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


def print_life_data():
    weight = input("Weight(lbs): ")
    sex = input("Sex(m/f): ")
    height = input("Height(in): ")
    sys_bp = input("Sys_BP: ")
    smoker = input("Smoker (y/n): ")
    nic_other = input("Nicotine (other than smoking) use (y/n): ")
    num_meds = input("Number of medications: ")
    occup_danger = input("Occupational danger (1/2/3): ")
    ls_danger = input("Lifestyle danger (1/2/3): ")
    cannabis = input("Cannabis use (y/n): ")
    opioids = input("Opioid use (y/n): ")
    other_drugs = input("Other drug use (y/n): ")
    drinks_aweek = input("Drinks per week: ")
    addiction = input("Addiction history (y/n): ")
    major_surgery_num = input("Number of major surgeries: ")
    diabetes = input("Diabetes (y/n): ")
    hds = input("Heart disease history (y/n): ")
    cholesterol = input("Cholesterol: ")
    asthma = input("Asthma (y/n): ")
    immune_defic = input("Immune deficiency (y/n): ")
    family_cancer = input("Family history of cancer (y/n): ")
    family_heart_disease = input("Family history of heart disease (y/n): ")
    family_cholesterol = input("Family history of high cholesterol (y/n): ")

    # Store all inputs in an array then prep
    inputs = [
        weight, sex, height, sys_bp, smoker, nic_other, num_meds, occup_danger,
        ls_danger, cannabis, opioids, other_drugs, drinks_aweek, addiction,
        major_surgery_num, diabetes, hds, cholesterol, asthma, immune_defic,
        family_cancer, family_heart_disease, family_cholesterol
    ]
    #inputs=prep_inputs([inputs])

    print(model(inputs))

if __name__ == "__main__":
    from utils import load_prep_data
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
    for _ in range(1):
        model.neural_net_train(epoch=1)#Change epoch to do more training between evals
        model.neural_net_eval()
        model.save_model()
        model=load_model(NeuralNet)
        model.neural_net_eval()