'''Goal: Does this work? Can we test to see if this model forecasts the cost of life insurance?'''
import torch
import numpy as np
from neural_net import NeuralNet
from utils import load_fold_data
from torch.utils.data import DataLoader, TensorDataset
from actu import life_liability_pv_mu, payout_pv

def actuarial_model_eval(I,training_reps,fold_num=5,smooth=False,sigma=5):
    '''Evaluates the performance of our model with regard to actuarial performance
    Args:
        I: The interest rate as a percentage, e.g. 1% => 1
        training_reps: The number of times the training set is ran through the model,
            high numbers cause overfitting and poor performance
        fold_num: The number of folds, must be a factor of 10000
    '''
    # Step 1: Load data and split into folds
    X_tensor,y_tensor,scalar,input_cols = load_fold_data('data.csv')
    X_folds=X_tensor.split(int(10000/fold_num)) # Creates 5 parallel folds in X and y
    y_folds=y_tensor.split(int(10000/fold_num))

    # Step 2: Loop through folds
    liability_difference=0
    for i,fold in enumerate(X_folds):
        print(f'Fold: {i}')

        # First we create our training set
        X_train=None
        y_train=None
        for j,fold in enumerate(X_folds):
            if j!=i:
                if X_train is None:
                    X_train=fold
                    y_train=y_folds[j]
                else:
                    X_train=torch.cat((X_train,fold))
                    y_train=torch.cat((y_train,y_folds[j]))
        #print(X_train)
        #print(y_train)
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Now we train the model
        NN_model = NeuralNet()
        NN_model.neural_net_train(train_loader,epochs=training_reps,print_statement=False)

        # Now we test: Compare expected liability with actual

        # calculate actual
        y_test=y_folds[i].numpy()
        actual_liability=0
        for indiv in y_test:
            indiv=np.array(indiv).tolist()
            years_til_death=indiv.index(1)+1
            actual_liability+=payout_pv(1,years_til_death,I)
        print(f'Actual liability is: ${actual_liability}')

        # calculate expected
        individuals = fold.split(1)
        expect_liability=0
        for indiv in individuals:
            mort_df=NN_model.get_life_data(indiv,True,smooth,sigma)
            mort_tab=mort_df[0].to_numpy()
            #print(mort_tab)
            expect_liability+=life_liability_pv_mu(1,I,mort_tab)
        print(f'Expected liability is: ${expect_liability:.2f}')

        # Difference
        diff=actual_liability-expect_liability
        print(f'Difference between actual liability and expected liability is: {diff}')
        liability_difference+=diff
    average_diff_percent=liability_difference/100
    print(f'The average difference between actual and expected liability is {average_diff_percent:.4f}%')

if __name__ == "__main__":
    #actuarial_model_eval(1,2,10)
    actuarial_model_eval(1,2,10,True,10)
    # Conclusion: Smoothing improves accuracy and reduces bias
    # Not sure on ideal sigma though...
    # Also need to test number of training reps...
    # And layers...