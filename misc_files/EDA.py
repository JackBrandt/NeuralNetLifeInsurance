import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)

def basic_info(df):
    """Print basic information about the dataset."""
    print("Dataset Info:")
    print(df.info())
    print("\nFirst five rows:")
    print(df.head())

def summary_statistics(df):
    """Display summary statistics of the dataset."""
    print("\nSummary Statistics:")
    print(df.describe())

def check_missing_values(df):
    """Check for missing values in the dataset."""
    print("\nMissing Values:")
    print(df.isnull().sum())

def visualize_distributions(df):
    """Generate histograms for numerical features."""
    df.hist(figsize=(12, 8), bins=30, edgecolor='black')
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.show()

def encode_binary_columns(df):
    """Convert binary categorical attributes ('m/f', 'y/n') to numerical values."""
    binary_mappings = {'m': 1, 'f': 0, 'y': 1, 'n': 0}
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() == 2:
            df[col] = df[col].map(binary_mappings)
    return df

def correlation_matrix(df):
    """Display the correlation matrix as a heatmap."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.show()

def run_eda(file_path):
    """Run the complete EDA process on a given dataset."""
    df = load_data(file_path)
    basic_info(df)
    summary_statistics(df)
    check_missing_values(df)
    visualize_distributions(df)
    correlation_matrix(encode_binary_columns(df))

if __name__ == "__main__":
    file_path = input("Enter the path to your dataset: ")
    run_eda(file_path)
