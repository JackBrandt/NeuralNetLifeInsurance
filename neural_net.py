import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from utils import gaussian_smooth,load_prep_data

from sklearn.preprocessing import StandardScaler
from utils import get_life_inputs, convert_to_binary

def load_model(filepath: str):
    """
    Load a saved PyTorch model from disk and set it to evaluation mode.

    Args:
        filepath (str): Path to the saved model file (.pt or .pth format)

    Returns:
        torch.nn.Module: Loaded PyTorch model in evaluation mode

    Example:
        >>> model=load_model('models/25.pth')
    """
    model = torch.load(filepath, weights_only=False)
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {filepath}")
    return model

class NeuralNet(nn.Module):
    def __init__(self,age):
        '''
        Initializes an instance of the NeuralNet class, setting up the architecture of the 
        neural network for multi-class classification. 
        This network includes two linear layers with ReLU activation for the first layer and 
        Softmax for the output layer. The output layer's
        number of neurons dynamically adjusts based on the `age` parameter. The constructor 
        also sets up the optimizer, loss function, batch size, and feature scaler.

        Parameters:
            age (int): Age parameter influences the number of neurons in the output layer. 
            Specifically, the number of output neurons is calculated as `96 - (age - 25)`. 
            This design assumes that the age input directly correlates with the desired 
            complexity or capacity of the model's output layer.

        Attributes:
            fc1 (nn.Linear): First hidden layer with 23 input features and 10 output neurons.
            fc2 (nn.Linear): Dynamically adjusted second/output layer with 10 input neurons 
            and a variable number of output neurons based on the `age` parameter.
            relu (nn.ReLU): ReLU activation function applied after the first hidden layer.
            softmax (nn.Softmax): Softmax activation function applied to the output of 
            the second layer, facilitating multi-class classification.
            optimizer (optim.Adam): Adam optimizer with a learning rate of 0.001, used for
              updating the weights and biases of the network during training.
            criterion (nn.CrossEntropyLoss): Loss function used for multi-class classification.
            batch_size (int): Specifies the number of samples in each batch to be fed to 
            the network during training (32 by default).
            scaler (StandardScaler): Feature scaler for normalizing/standardizing 
            the inputs to the network.
            cols (list): List to store column names or other relevant metadata, 
            initialized as an empty list.

        Returns:
            None: This constructor method does not return any value but 
            initializes the neural network's components.
        '''
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
        '''
        Propagates the input through the neural network model. The input data passes through
        the first linear layer followed by a ReLU activation function. 
        The second linear layer then processes the result, 
        and a Softmax function is applied to the output of the second layer 
        to generate probability distributions for the classes.

        Parameters:
            x (Tensor): The input data tensor that needs to be processed 
            by the neural network. 
            This should have the appropriate shape expected by the
              first linear layer (nn.Linear(23, 10)).

        Returns:
            Tensor: A tensor containing the softmax output of the network,
              representing probability distributions over the classes. 
            The shape of the output tensor depends on the input and 
            the dynamic configuration of the second layer 
            (determined by the `age` parameter set during initialization).
        '''
        x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc2(x)
        return self.softmax(x)  # Use softmax if classification

    def save_model(self):
        '''
        Saves the current state of the neural network model to a file. 
        The filename is determined by the configuration of the 
        neural network's output layer. 
        Specifically, the filename encodes the age parameter that 
        influences the output layer's neuron count, allowing for
          the identification and reuse of model configurations tailored to specific age values.

        Parameters:
            None: This method does not take any parameters.

        Returns:
            None: This method does not return any value.
              It outputs a confirmation message indicating the file path where the model is saved.

        Side Effects:
            - Saves the model to the filesystem under the 'models/' directory.
            - Prints the file path where the model is saved,
              providing feedback on the save operation.

        Usage:
            Call this method to save the model's state for later use or for deployment:
            >>> neural_network_instance.save_model()
        '''
        age=25+96-self.fc2.out_features
        filepath='models/'+str(age)+'.pth'
        torch.save(self, filepath)  # Save model parameters
        print(f"Model saved to {filepath}")

    # Training loop
    def neural_net_train(self,train_loader, epoch=1, print_statement=True):
        '''
        Trains the neural network using the provided training data loader.
          This method runs through the dataset for a specified number of epochs, 
        performing forward passes, loss computation, and backpropagation 
        for weights updates. Additionally, it prints the loss after each epoch if requested.

        Parameters:
            train_loader (DataLoader): The DataLoader instance that provides 
            batches of training data tuples (inputs, labels).
            epoch (int, optional): The number of epochs to train the network for.
              Defaults to 1.
            print_statement (bool, optional): Flag to control the
              printing of loss after each epoch. Defaults to True.

        Returns:
            None: This function does not return any value but prints the 
            training loss and completion status if `print_statement` is True.

        Usage:
            Call this method with a DataLoader containing the training data,
              and optionally adjust the number of training epochs and verbosity:
            >>> neural_net.neural_net_train(train_loader=my_train_loader, epoch=5, print_statement=True)
        '''
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
        '''
        Evaluates the neural network model on a given dataset using
          mean absolute error as the metric. 
        This method sets the model to evaluation mode, iterates through 
        the provided DataLoader, computes errors for each batch, 
        and finally calculates the average of these errors across
          all batches as the mean mean absolute error.

        Parameters:
            test_loader (DataLoader): The DataLoader containing the test dataset. 
            It should provide batches of data in the form of (inputs, labels) tuples.

        Returns:
            None: This method does not return any value. Instead, 
            it prints the calculated mean mean absolute error 
            after evaluating the entire test dataset.

        Side Effects:
            - Sets the model to evaluation mode to disable dropout 
            and batch normalization during inference.
            - Prints the mean mean absolute error, which is an average of 
            the mean absolute errors across all samples in each batch.

        Usage:
            To evaluate the model and understand its performance on unseen data,
              pass a DataLoader containing the test data:
            >>> neural_network_instance.neural_net_eval(test_loader)

        Note:
            - The function assumes the outputs and labels are
              in compatible formats for direct subtraction to calculate the absolute error. 
            Adjustments may be needed depending on the 
            specific output format of your model.
            - This method directly prints the results, which might not be ideal for all use cases. 
            Consider modifying the function to return the error 
            if it needs to be used programmatically.
        '''
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
        '''
        This method orchestrates the complete process of training, evaluating,
          and saving the neural network model. 
        It first loads and prepares the data, then repeatedly trains the network,
          evaluates it, and saves the state, according to the specified parameters. 
        It is designed to facilitate multiple iterations of 
        training and evaluation to refine the model's performance.

        Parameters:
            reps (int): The number of repetitions of the training and evaluation cycles.
            epoch (int): The number of epochs for which 
            the network should be trained in each repetition.
            eval_always (bool, optional): Determines whether to 
            evaluate the model after every training repetition. 
            If set to False, the model is only evaluated after
              the final repetition. Defaults to True.

        Returns:
            DataLoader: Returns the test DataLoader used in the evaluations. 
            This can be useful for further testing or verification 
            after the function has completed.

        Usage:
            To perform training, evaluation, and saving for 5 repetitions with 10 epochs each,
              and to evaluate after every training cycle, use:
            >>> neural_network_instance.train_eval_save(reps=5, epoch=10, eval_always=True)

        Notes:
            - The data is loaded and preprocessed from a specified CSV file.
              The number of features and the target setup in this CSV should
                match the expected input size 
            and output configuration of the neural network.
            - Each repetition consists of training for the specified number of epochs, 
            an optional evaluation, and a model save operation.
            - The function uses 'data.csv' to load the data, 
            and it adjusts preprocessing parameters based 
            on the neural network's current configuration.
        '''
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
        '''
        This method handles the training of the neural network 
        for a specified number of repetitions and epochs, 
        saving the model after each complete training cycle.
          It dynamically adjusts the learning rate based on the size of 
          the training data to optimize training efficiency.

        Parameters:
            reps (int): The number of complete training cycles to perform. 
            Each cycle includes training the network for the specified number 
            of epochs and then saving the model.
            epoch (int): The number of epochs to train the network in each repetition.

        Returns:
            None: This method does not return any value. It performs 
            training and saves the model state to disk after each repetition.

        Side Effects:
            - Prints the length of the training dataset and the 
            calculated learning rate adjustment factor.
            - Saves the model to disk after each training cycle.
            - Adjusts the learning rate of the optimizer dynamically 
            based on the size of the training dataset.

        Usage:
            To train the model for 3 repetitions with each consisting of 10 epochs, use:
            >>> neural_network_instance.train_save(reps=3, epoch=10)

        Notes:
            - The data is loaded and preprocessed by `load_prep_data` 
            which is configured to omit the test split when `test=False` is specified.
            - The learning rate is calculated as 
            `0.001 * ((10000 / len(X_train)) ** 0.5)`, 
            adjusting it based on the training set size to
              maintain effective learning speeds.
            - Ensure that the batch size and other training parameters 
            are properly set before calling this function.
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
        '''
        Retrieves and processes life data inputs to generate model predictions. 
        This function allows for optional smoothing of the output data
          based on specified parameters.

        Parameters:
            inputs (array-like or Tensor, optional): The input data to be used for predictions. 
            If not provided, the function fetches the inputs using the `get_life_inputs` method.
            is_tensor (bool, optional): Specifies whether the provided `inputs` are
              already in tensor format. Defaults to False.
            smooth_percentage (int, optional): The percentage of the output
              to be smoothed using a Gaussian function. Defaults to 0, indicating no smoothing.
            sigma (int, optional): The standard deviation for the Gaussian smoothing function. 
            Only relevant if `smooth_percentage` is greater than 0. Defaults to 5.

        Returns:
            DataFrame or Tensor: Returns a DataFrame or Tensor containing the model predictions. 
            If `smooth_percentage` is specified, the output 
            will be a smoothed version of the predictions.

        Side Effects:
            - Transforms input data into a DataFrame (if not already a tensor), applies conversions, 
            and standardizes using the pre-fitted scaler.
            - Evaluates the model in non-training mode and with no gradient calculations.
            - Optionally applies Gaussian smoothing to the predictions based on the `smooth_percentage`.

        Usage:
            # To get predictions without any smoothing:
            >>> predictions = neural_network_instance.get_life_data(inputs=my_data)

            # To get predictions with 20% Gaussian smoothing:
            >>> smoothed_predictions = 
            neural_network_instance.get_life_data(inputs=my_data, smooth_percentage=20, sigma=3)

        Note:
            - The function automatically handles the conversion of non-tensor 
            inputs into the appropriate tensor format needed for model prediction.
            - The range for the index of the output DataFrame is 
            dynamically calculated based on the configuration of the model's output layer.
        '''
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
    '''
    Creates and trains a series of neural network models for different ages
      ranging from 25 up to a specified age cap. Each model is saved after training.

    Parameters:
        age_cap (int): The maximum age for which to create and train models. 
        Models will be created for every year starting from 25 up to (but not including) this age.

    Returns:
        None: This function does not return any value but saves trained models to disk.

    Usage:
        >>> make_all_models(80)
    '''
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

