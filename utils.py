import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def convert_to_binary(value):
    """Converts 'm' or 'y' to 1, otherwise returns 0."""
    return 1 if str(value).lower() in ["m", "y"] else 0

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
    input_cols=X.columns
    # Convert categorical columns to numerical values
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].apply(lambda x: convert_to_binary(x))

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Standardize input features

    # Convert to PyTorch tensors
    #print(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)  # Use long for classification

    # Split into train and test sets
    X_train, X_test, y_train, y_test =  train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler, input_cols

def get_life_inputs():
    weight = float(input("Weight(lbs): "))
    sex = input("Sex(m/f): ")
    height = float(input("Height(in): "))
    sys_bp = float(input("Sys_BP: "))
    smoker = input("Smoker (y/n): ")
    nic_other = input("Nicotine (other than smoking) use (y/n): ")
    num_meds = float(input("Number of medications: "))
    occup_danger = float(input("Occupational danger (1/2/3): "))
    ls_danger = float(input("Lifestyle danger (1/2/3): "))
    cannabis = input("Cannabis use (y/n): ")
    opioids = input("Opioid use (y/n): ")
    other_drugs = input("Other drug use (y/n): ")
    drinks_aweek = float(input("Drinks per week: "))
    addiction = input("Addiction history (y/n): ")
    major_surgery_num = float(input("Number of major surgeries: "))
    diabetes = input("Diabetes (y/n): ")
    hds = input("Heart disease history (y/n): ")
    cholesterol = float(input("Cholesterol: "))
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
    return inputs