import streamlit as st

# Title
st.title("Death Predictors: Neural Network Life Insurance Calculator")

# Sidebar
st.sidebar.header("IDK sidebar stuff")

# Main stuff
st.write("Please enter your personal info and risk factors below to get started")
weight = st.text_input("Weight(lbs): ")
sex = st.text_input("Sex(m/f): ")
height = st.text_input("Height(in): ")
sys_bp = st.text_input("Sys_BP: ")
smoker = st.text_input("Smoker (y/n): ")
nic_other = st.text_input("Nicotine (other than smoking) use (y/n): ")
num_meds = st.text_input("Number of medications: ")
occup_danger = st.text_input("Occupational danger (1/2/3): ")
ls_danger = st.text_input("Lifestyle danger (1/2/3): ")
cannabis = st.text_input("Cannabis use (y/n): ")
opioids = st.text_input("Opioid use (y/n): ")
other_drugs = st.text_input("Other drug use (y/n): ")
drinks_aweek = st.text_input("Drinks per week: ")
addiction = st.text_input("Addiction history (y/n): ")
major_surgery_num = st.text_input("Number of major surgeries: ")
diabetes = st.text_input("Diabetes (y/n): ")
hds = st.text_input("Heart disease history (y/n): ")
cholesterol = st.text_input("Cholesterol: ")
asthma = st.text_input("Asthma (y/n): ")
immune_defic = st.text_input("Immune deficiency (y/n): ")
family_cancer = st.text_input("Family history of cancer (y/n): ")
family_heart_disease = st.text_input("Family history of heart disease (y/n): ")
family_cholesterol = st.text_input("Family history of high cholesterol (y/n): ")
inputs = [
    weight, sex, height, sys_bp, smoker, nic_other, num_meds, occup_danger,
    ls_danger, cannabis, opioids, other_drugs, drinks_aweek, addiction,
    major_surgery_num, diabetes, hds, cholesterol, asthma, immune_defic,
    family_cancer, family_heart_disease, family_cholesterol
]
for i,input in enumerate(inputs):
    try:
        inputs[i]=float(input)
    except:
        pass

# Main Content
st.write(f"You entered: {inputs}")

# Interactive Components
st.write('After you enter your personal information, click the button to calculate your expected insurance cost')
if st.button("Click me"):
    st.success("Button clicked!")
