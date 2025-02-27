import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import streamlit as st

def convert_to_binary(value):
    """Converts 'm' or 'y' to 1, otherwise returns 0."""
    return 1 if str(value).lower() in ['m', 'y'] else 0

def load_prep_data(file_path, age):
    """Loads and prepares data for training."""
    df = pd.read_csv(file_path)
    df = df[df['age'] >= age]
    y = df.iloc[:, 0] - age
    y = pd.get_dummies(y.apply(lambda x: max(min(x, 95), 0))).reindex(columns=range(96), fill_value=0)
    X = df.iloc[:, 1:]
    X[X.select_dtypes(include=['object']).columns] = X.select_dtypes(include=['object']).applymap(convert_to_binary)
    scaler = StandardScaler()
    print(X.columns)
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32), test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler, X.columns

def get_life_inputs():
    """Collects user inputs for life expectancy predictions."""
    fields = ['Weight (lbs)', 'Sex (m/f)', 'Height (in)', 'Systolic BP', 'Smoker (y/n)', 'Nicotine use (y/n)', 'Number of medications',
              'Occupational danger (1-3)', 'Lifestyle danger (1-3)', 'Cannabis use (y/n)', 'Opioid use (y/n)', 'Other drug use (y/n)',
              'Drinks per week', 'Addiction history (y/n)', 'Number of major surgeries', 'Diabetes (y/n)', 'Heart disease history (y/n)',
              'Cholesterol', 'Asthma (y/n)', 'Immune deficiency (y/n)', 'Family history of cancer (y/n)', 'Family history of heart disease (y/n)',
              'Family history of high cholesterol (y/n)']
    inputs = [input(f"{field}: ") for field in fields]
    return [inputs]

def plot_mort(mort_df):
    """Plots mortality rate from a DataFrame."""
    plt.figure(figsize=(10, 5))
    plt.plot(mort_df.index, mort_df[0], marker='o', linestyle='-')
    plt.xticks(rotation=90)
    plt.xlabel("Year")
    plt.ylabel("Mortality Rate")
    plt.title("Line Plot of Mortality Table")
    plt.grid(True)
    plt.show()

def gaussian_smooth(df, sigma=15):
    """Applies Gaussian smoothing to a DataFrame."""
    smoothed_values = gaussian_filter1d(df.iloc[:, 0], sigma=sigma, mode='nearest')
    return pd.DataFrame(smoothed_values, index=df.index)

# Utility functions for formatting
def format_sex(sex_option):
    return "Male" if sex_option.lower() == 'm' else "Female"

def format_yes_no(option):
    return "Yes" if option.lower() == 'y' else "No"

def format_risk_level(num):
    return {1: "Low", 2: "Medium", 3: "High"}.get(num, "Unknown")

def format_policy_type(pol_type, duration=None):
    if pol_type == 'fl':
        return 'Fixed-Rate for Life'
    elif pol_type == 'fd':
        return f'Fixed-Rate for {duration} years' if duration else 'Fixed-Rate for Duration'
    return 'Variable Rate'

# Streamlit-specific functions
def store_value(perm_key):
    st.session_state[perm_key] = st.session_state["_" + perm_key]

def load_value(perm_key):
    st.session_state["_" + perm_key] = st.session_state[perm_key]
