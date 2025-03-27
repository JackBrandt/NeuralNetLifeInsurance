import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file and return a DataFrame.
    
    Parameters:
    - file_path (str): The path to the CSV file to be loaded.
    
    Returns:
    - DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def display_basic_info(df: pd.DataFrame):
    """
    Print basic information and the first few rows of the DataFrame.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame whose info to print.
    """
    print("Dataset Information:")
    print(df.info())
    print("\nFirst five rows of the dataset:")
    print(df.head())

def display_summary_statistics(df: pd.DataFrame):
    """
    Display summary statistics for the DataFrame.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame to summarize.
    """
    print("\nSummary Statistics:")
    print(df.describe())

def check_and_report_missing_values(df: pd.DataFrame):
    """
    Check and print a report of missing values for each column in the DataFrame.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame to check for missing values.
    """
    missing_values = df.isnull().sum()
    print("\nMissing Values in Each Column:")
    print(missing_values[missing_values > 0])

def plot_feature_distributions(df: pd.DataFrame):
    """
    Generate and display histograms for each numerical feature in the DataFrame.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame to visualize.
    """
    df.hist(figsize=(12, 8), bins=30, edgecolor='black')
    plt.suptitle("Histograms of Feature Distributions", fontsize=16)
    plt.show()

def convert_binary_columns_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert binary categorical attributes to numerical {0, 1} format.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame to process.
    
    Returns:
    - DataFrame: The DataFrame with binary columns converted.
    """
    binary_mappings = {'m': 1, 'f': 0, 'y': 1, 'n': 0}
    for column in df.select_dtypes(include=['object']).columns:
        if df[column].nunique() == 2:
            df[column] = df[column].map(binary_mappings)
    return df

def display_correlation_matrix(df: pd.DataFrame):
    """
    Display a heatmap of the correlation matrix for the DataFrame.
    
    Parameters:
    - df (DataFrame): The pandas DataFrame to analyze.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.show()

def run_eda(file_path: str):
    """
    Execute a complete Exploratory Data Analysis (EDA) process on a dataset.
    
    Parameters:
    - file_path (str): The path to the dataset file.
    """
    df = load_data(file_path)
    display_basic_info(df)
    display_summary_statistics(df)
    check_and_report_missing_values(df)
    plot_feature_distributions(df)
    df = convert_binary_columns_to_numeric(df)
    display_correlation_matrix(df)

if __name__ == "__main__":
    file_path = input("Enter the path to your dataset: ")
    run_eda(file_path)
