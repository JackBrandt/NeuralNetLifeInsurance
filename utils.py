import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import streamlit as st

#TODO: Fix function headers

def convert_to_binary(value):
    """Converts 'm' or 'y' to 1, otherwise returns 0."""
    return 1 if str(value).lower() in ["m", "y"] else 0

def load_data():
    df = pd.read_csv('data.csv', header=0)
    return df

def prep_data(df, age,test=True,sigma=10):
    df = df[df['age'] >= age]

    # Extract target (y) and features (X)
    empty=[0]*(int(96-(age-25)))
    y_vals = df.iloc[:, 0].values  # First column is target
    y_vals=[value-age for value in y_vals]
    y=[empty.copy() for _ in y_vals]
    for i,y_val in enumerate(y_vals):
        y[i][y_val]=1.0
        y[i]=gaussian_filter1d(y[i],sigma)
        #print(y[i])
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

    if test:
        # Split into train and test sets
        X_train, X_test, y_train, y_test =  train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, scaler, input_cols
    else:
        return X_tensor,y_tensor,scaler,input_cols

def load_prep_data(age,test=True):
    '''Loads and preps data...
    Args:
        file_path (str):...
    Returns:
        X (array of arrays)
        y (array of arrays)
    '''
    # Load CSV file
    df=load_data()
    return prep_data(df,age,test)

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
    X = scaler.fit_transform(X)  # Standardize input features

    # Convert to PyTorch tensors
    #print(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)  # Use long for classification

    return X_tensor,y_tensor, scaler, input_cols

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
    return [inputs]

def plot_mort(mort_df):
        plt.figure(figsize=(10, 5))
        plt.plot(mort_df.index, mort_df[0], marker='o', linestyle='-')
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.xlabel("Year")
        plt.ylabel("Mort Rate")
        plt.title("Line Plot of Mort Table")
        plt.grid()

        plt.show()

def gaussian_smooth(df, sigma=15):
    """
    Applies Gaussian smoothing to a numerical DataFrame column.

    Parameters:
        df (pd.DataFrame): DataFrame containing numerical data.
        sigma (float): Standard deviation of the Gaussian kernel. Higher values = more smoothing.

    Returns:
        pd.DataFrame: A new DataFrame with smoothed values.
    """
    smoothed_values = gaussian_filter1d(df[0], sigma=sigma, mode='nearest')
    return pd.DataFrame({0: smoothed_values}, index=df.index)

def sex_format(sex_option):
    if sex_option=='m': return "Male"
    return "Female"

def yn_format(yn):
    if yn=='y': return "Yes"
    return "No"

def risk_num_format(num):
    match num:
        case 1:
            return "Low"
        case 2:
            return "Medium"
        case _:
            return "High"

def policy_type_format(pol_type,duration=None):
    match pol_type:
        case 'fl':
            return 'Fixed-Rate for Life'
        case 'fd':
            if duration==None:
                return 'Fixed-Rate for Duration'
            else:
                return f'fixed-rate for duration of {duration} years'
        case _:
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

def st_get_inputs(previous_user_inputs):
    st.write("Please enter your personal info and risk factors below to get started")
    dob = st.date_input("Date of birth",value=previous_user_inputs[0],format="MM/DD/YYYY")
    weight = st.text_input("Weight(lbs): ",value=previous_user_inputs[1])
    sex = st.pills("Sex:", ['m','f'],key='sex', selection_mode="single", format_func=sex_format, label_visibility="visible",default=previous_user_inputs[2])
    height = st.text_input("Height(in): ",value=previous_user_inputs[3])
    sys_bp = st.text_input("Sys_BP: ",value=previous_user_inputs[4])
    smoker = st.pills("Do you smoke:", ['y','n'],key='smoke', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[5])
    nic_other = st.pills("Do you use other forms of nicotine? (e.g., vape or chewing tobacco):", ['y','n'],key='nic', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[6])
    num_meds = st.text_input("Number of medications: ",value=previous_user_inputs[7])
    occup_danger = st.pills("How would you describe your occupational danger? (Example: Underwater welding -> High, Office work -> Low)", [1,2,3],key='occupy', selection_mode="single", format_func=risk_num_format, label_visibility="visible",default=previous_user_inputs[8])
    ls_danger = st.pills("How would you describe your lifestyle danger? (Example: Frequent skydiving -> High, )", [1,2,3],key='ls', selection_mode="single", format_func=risk_num_format, label_visibility="visible",default=previous_user_inputs[9])
    cannabis = st.pills("Do you use cannabis, weed, or pot?", ['y','n'],key='weed', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[10])
    opioids = st.pills("Do you use opioids?", ['y','n'],key='opioids', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[11])
    other_drugs = st.pills("Do you use any other drugs:", ['y','n'],key='drugs', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[12])
    drinks_aweek = st.text_input("Drinks per week: ",value=previous_user_inputs[13])
    addiction = st.pills("Do you have a history of addiction?", ['y','n'],key='addict', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[14])
    major_surgery_num = st.text_input("Number of major surgeries: ",value=previous_user_inputs[15])
    diabetes = st.pills("Do you have diabetes?", ['y','n'],key='diab', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[16])
    hds = st.pills("Do you have a history of heart disease or stroke?", ['y','n'],key='hds', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[17])
    cholesterol = st.text_input("Cholesterol: ",value=previous_user_inputs[18])
    asthma = st.pills("Do you have asthma?", ['y','n'], selection_mode="single",key='asthma', format_func=yn_format, label_visibility="visible",default=previous_user_inputs[19])
    immune_defic = st.pills("Do you have an immune deficiency?", ['y','n'],key='immune', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[20])
    family_cancer = st.pills("Do you have a family history of cancer?", ['y','n'],key='cancer', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[21])
    family_heart_disease = st.pills("Do you have a family history of heart disease or stroke?", ['y','n'], key='familyhds',selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[22])
    family_cholesterol = st.pills("Do you have a family history of high cholesterol?", ['y','n'],key='chol', selection_mode="single", format_func=yn_format, label_visibility="visible",default=previous_user_inputs[23])
    inputs = [
        dob, weight, sex, height, sys_bp, smoker, nic_other, num_meds, occup_danger,
        ls_danger, cannabis, opioids, other_drugs, drinks_aweek, addiction,
        major_surgery_num, diabetes, hds, cholesterol, asthma, immune_defic,
        family_cancer, family_heart_disease, family_cholesterol
    ]
    for i,input in enumerate(inputs):
        try:
            inputs[i]=float(input)
        except:
            pass
    return inputs

if __name__ == '__main__':
    load_prep_data(25,False)