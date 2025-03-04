import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import streamlit as st

def load_and_preprocess_data(filepath: str, target_age: int):
    """
    Loads data from a CSV file, cleans, and preprocesses it.
    """
    # Load data
    data = pd.read_csv(filepath)
    print("Dataset columns:", data.columns.tolist())
    
    # Clean data
    # Allow two missing values per row
    data.dropna(axis=0, thresh=len(data.columns)-2, inplace=True)
    # Drop columns with more than 20% missing values
    data.dropna(axis=1, thresh=0.8*len(data), inplace=True)
    
    # Filter data by age and blood pressure
    if 'age_column' in data.columns:
        data = data[(data['age_column'] >= 18) & (data['age_column'] <= 100)]
    if 'bp_column' in data.columns:
        data = data[data['bp_column'] > 0]
    
    # Remove duplicates
    if 'id_column' in data.columns:
        data.drop_duplicates(subset=['id_column'], inplace=True)
    
    # Extract features and target
    X = data.drop('target_column', axis=1)  # Placeholder: Replace 'target_column' with actual column name
    y = data['target_column']
    
    # Scale features
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
