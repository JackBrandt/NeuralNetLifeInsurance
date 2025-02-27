import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import gaussian_smooth, load_prep_data, get_life_inputs, convert_to_binary, plot_mort

class NeuralNet(nn.Module):
    def __init__(self, age):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(23, 10)  # First hidden layer
        self.fc2 = nn.Linear(10, 96 - (age - 25))  # Adaptive output layer based on age
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 32
        self.scaler = StandardScaler()
        self.cols = []

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

    def save_model(self):
        age = 25 + 96 - self.fc2.out_features
        filepath = f'models/{age}.pth'
        torch.save(self, filepath)
        print(f"Model saved to {filepath}")

    def train_eval_save(self, data_file, reps, epoch, eval_always=True):
        X_train, X_test, y_train, y_test, self.scaler, self.cols = load_prep_data(data_file, 25 + (96 - self.fc2.out_features))
        print(self.cols)
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        for i in range(reps):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            if eval_always or i == reps - 1:
                self.evaluate(test_loader)

            self.save_model()

    def evaluate(self, test_loader):
        self.eval()
        total_error = 0
        count = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self(inputs)
                errors = (outputs - labels).abs().sum(dim=1).mean()
                total_error += errors
                count += 1
        print(f"Test Mean Absolute Error: {total_error / count}")

    def get_life_data(self, inputs=None, is_tensor=False, smooth=False, sigma=5):
        if inputs is None:
            inputs = get_life_inputs()
        if not is_tensor:
            inputs = pd.DataFrame(inputs, columns=self.cols)
            inputs = inputs.applymap(lambda x: convert_to_binary(x) if isinstance(x, str) else x)
            inputs = self.scaler.transform(inputs)
        tensor_input = torch.tensor(inputs, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            output = self(tensor_input)
        output = pd.DataFrame(output.numpy()).transpose()
        output.index = [str(i) for i in range(25 + (96 - self.fc2.out_features), 121)]
        if smooth:
            output = gaussian_smooth(output, sigma)
        return output

def make_all_models(age_cap):
    for age in range(25, age_cap):
        model = NeuralNet(age)
        model.train_eval_save('data.csv', 2, 1)

if __name__ == "__main__":
    make_all_models(26)
    model = NeuralNet(25)
    mort_df = model.get_life_data([[180, 'm', 72, 130, 'n', 'n', 3, 1, 1, 'n', 'n', 'n', 4, 'n', 0, 'n', 'n', 200, 'n', 'n', 'n', 'n', 'n']])
    plot_mort(mort_df)
    smoothed_df = gaussian_smooth(mort_df, 10)
    plot_mort(smoothed_df)
