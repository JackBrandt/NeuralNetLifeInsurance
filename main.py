import streamlit as st

# Title
st.title("Death Predictors: Neural Network Life Insurance Calculator")

# Sidebar
st.sidebar.header("Personal Information and Risk Factors")
weight = float(st.sidebar.text_input("Weight(lbs): "))
sex = st.sidebar.text_input("Sex(m/f): ")
height = float(st.sidebar.text_input("Height(in): "))
sys_bp = float(st.sidebar.text_input("Sys_BP: "))
smoker = st.sidebar.text_input("Smoker (y/n): ")
nic_other = st.sidebar.text_input("Nicotine (other than smoking) use (y/n): ")
num_meds = float(st.sidebar.text_input("Number of medications: "))
occup_danger = float(st.sidebar.text_input("Occupational danger (1/2/3): "))
ls_danger = float(st.sidebar.text_input("Lifestyle danger (1/2/3): "))
cannabis = st.sidebar.text_input("Cannabis use (y/n): ")
opioids = st.sidebar.text_input("Opioid use (y/n): ")
other_drugs = st.sidebar.text_input("Other drug use (y/n): ")
drinks_aweek = float(st.sidebar.text_input("Drinks per week: "))
addiction = st.sidebar.text_input("Addiction history (y/n): ")
major_surgery_num = float(st.sidebar.text_input("Number of major surgeries: "))
diabetes = st.sidebar.text_input("Diabetes (y/n): ")
hds = st.sidebar.text_input("Heart disease history (y/n): ")
cholesterol = float(st.sidebar.text_input("Cholesterol: "))
asthma = st.sidebar.text_input("Asthma (y/n): ")
immune_defic = st.sidebar.text_input("Immune deficiency (y/n): ")
family_cancer = st.sidebar.text_input("Family history of cancer (y/n): ")
family_heart_disease = st.sidebar.text_input("Family history of heart disease (y/n): ")
family_cholesterol = st.sidebar.text_input("Family history of high cholesterol (y/n): ")
inputs = [
    weight, sex, height, sys_bp, smoker, nic_other, num_meds, occup_danger,
    ls_danger, cannabis, opioids, other_drugs, drinks_aweek, addiction,
    major_surgery_num, diabetes, hds, cholesterol, asthma, immune_defic,
    family_cancer, family_heart_disease, family_cholesterol
]
for input in inputs:
    try:
        input=float(input)
    except:
        pass

# Main Content
st.write("Please enter your personal info and risk factors in the side bar on the left to get started")
st.write(f"You entered: {inputs}")

# Interactive Components
if st.button("Click me"):
    st.success("Button clicked!")
