import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import streamlit as st

#TODO: Fix function headers

def convert_to_binary(value):
    """Converts 'm' or 'y' to 1, otherwise returns 0."""
    return 1 if str(value).lower() in ["m", "y"] else 0

def load_prep_data(file_path,age):
    '''Loads and preps data...
    Args:
        file_path (str):...
    Returns:
        X (array of arrays)
        y (array of arrays)
    '''
    # Load CSV file
    df = pd.read_csv(file_path, header=0)
    df = df[df['age'] >= age]

    # Extract target (y) and features (X)
    empty=[0]*(int(96-(age-25)))
    y_vals = df.iloc[:, 0].values  # First column is target
    y_vals=[value-age for value in y_vals]
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

def load_fold_data(file_path):
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
    empty=[0]*96
    y_vals = df.iloc[:, 0].values  # First column is target
    y_vals=[value-25 for value in y_vals]
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
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2), scaler, X.columns.tolist()

def convert_to_binary(x: str) -> int:
    """
    Converts a categorical variable to binary (0/1).
    """
    if pd.isnull(x):
        return 0
    x = x.strip().lower()
    return 1 if x in {'yes', 'y', 'true', '1'} else 0

def collect_life_expectancy_inputs():
    """
    Collects user inputs for life expectancy predictions using interactive prompts.
    """
    fields = ['Weight (lbs)', 'Sex (m/f)', 'Height (in)', 'Systolic BP', 'Smoker (y/n)', 'Nicotine use (y/n)', 
              'Number of medications', 'Occupational danger (1-3)', 'Lifestyle danger (1-3)', 'Cannabis use (y/n)', 
              'Opioid use (y/n)', 'Other drug use (y/n)', 'Drinks per week', 'Addiction history (y/n)', 
              'Number of major surgeries', 'Diabetes (y/n)', 'Heart disease history (y/n)', 'Cholesterol', 
              'Asthma (y/n)', 'Immune deficiency (y/n)', 'Family history of cancer (y/n)', 'Family history of heart disease (y/n)', 
              'Family history of high cholesterol (y/n)']
    inputs = [input(f"{field}: ") for field in fields]
    return [inputs]

def plot_mortality_rate(mort_df: pd.DataFrame):
    """
    Plots the mortality rate from a DataFrame.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(mort_df.index, mort_df[0], marker='o', linestyle='-')
    plt.xticks(rotation=90)
    plt.xlabel("Year")
    plt.ylabel("Mortality Rate")
    plt.title("Mortality Rate Line Plot")
    plt.grid(True)
    plt.show()

def apply_gaussian_smoothing(df: pd.DataFrame, sigma: int = 15) -> pd.DataFrame:
    """
    Applies Gaussian smoothing to the first column of a DataFrame.
    """
    smoothed_values = gaussian_filter1d(df.iloc[:, 0], sigma=sigma, mode='nearest')
    return pd.DataFrame(smoothed_values, index=df.index)

# Streamlit-specific utility functions
def store_streamlit_value(perm_key: str):
    """
    Stores a value in Streamlit's session state.
    """
    st.session_state[perm_key] = st.session_state["_" + perm_key]

def retrieve_streamlit_value(perm_key: str):
    """
    Retrieves a value from Streamlit's session state.
    """
    st.session_state["_" + perm_key] = st.session_state[perm_key]

# Additional utility functions for formatting
def format_sex(sex_option: str) -> str:
    return "Male" if sex_option.lower() == 'm' else "Female"

def format_yes_no(option: str) -> str:
    return "Yes" if option.lower() == 'y' else "No"

def format_risk_level(num: int) -> str:
    return {1: "Low", 2: "Medium", 3: "High"}.get(num, "Unknown")

def format_policy_type(policy_type: str, duration: int = None) -> str:
    if policy_type == 'fl':
        return 'Fixed-Rate for Life'
    elif policy_type == 'fd':
        return f'Fixed-Rate for {duration} years' if duration else 'Fixed-Rate for Duration'
    return 'Variable Rate'

def store_value(perm_key):
    # Copy the value to the permanent key
    st.session_state[perm_key] = st.session_state["_"+perm_key]

def load_value(perm_key):
    # Copy the value to the permanent key
    st.session_state["_"+perm_key] = st.session_state[perm_key]

def get_storage_function(perm_key):
    return lambda : store_value(perm_key)

def get_loading_function(perm_key):
    return lambda : load_value(perm_key)