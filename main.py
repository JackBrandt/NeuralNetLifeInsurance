import streamlit as st
from actu import actu_str
from utils import sex_format,yn_format,risk_num_format,policy_type_format
# Title
st.title("Death Predictors: Neural Network Life Insurance Calculator")

# Main stuff
# TODO: Replace text_input with number_input with sensible parameters (e.g., height should be positive)

policy_type=st.pills("Enter Desired Policy Type", ['fl','fd','v'],key='pol_type', selection_mode="single", format_func=policy_type_format, label_visibility="visible")
age=st.number_input('What\'s your current age?',max_value=79,value=25,min_value= 25 if policy_type=='v' else 0)
if policy_type=='fd':
    duration=st.number_input("Policy Duration (years): ", 1 if age>24 else 26-age,int(120-age),20,1)
else:
    duration=None
fv=st.number_input("Policy Amount",125000)

st.write("Please enter your personal info and risk factors below to get started")
weight = st.text_input("Weight(lbs): ")
sex = st.pills("Sex:", ['m','f'],key='sex', selection_mode="single", format_func=sex_format, label_visibility="visible")
height = st.text_input("Height(in): ")
sys_bp = st.text_input("Sys_BP: ")
smoker = st.pills("Do you smoke:", ['y','n'],key='smoke', selection_mode="single", format_func=yn_format, label_visibility="visible")
nic_other = st.pills("Do you use other forms of nicotine? (e.g., vape or chewing tobacco):", ['y','n'],key='nic', selection_mode="single", format_func=yn_format, label_visibility="visible")
num_meds = st.text_input("Number of medications: ")
occup_danger = st.pills("How would you describe your occupational danger? (Example: Underwater welding -> High, Office work -> Low)", [1,2,3],key='occupy', selection_mode="single", format_func=risk_num_format, label_visibility="visible")
ls_danger = st.pills("How would you describe your lifestyle danger? (Example: Frequent skydiving -> High, )", [1,2,3],key='ls', selection_mode="single", format_func=risk_num_format, label_visibility="visible")
cannabis = st.pills("Do you use cannabis, weed, or pot?", ['y','n'],key='weed', selection_mode="single", format_func=yn_format, label_visibility="visible")
opioids = st.pills("Do you use opioids?", ['y','n'],key='opioids', selection_mode="single", format_func=yn_format, label_visibility="visible")
other_drugs = st.pills("Do you use any other drugs:", ['y','n'],key='drugs', selection_mode="single", format_func=yn_format, label_visibility="visible")
drinks_aweek = st.text_input("Drinks per week: ")
addiction = st.pills("Do you have a history of addiction?", ['y','n'],key='addict', selection_mode="single", format_func=yn_format, label_visibility="visible")
major_surgery_num = st.text_input("Number of major surgeries: ")
diabetes = st.pills("Do you have diabetes?", ['y','n'],key='diab', selection_mode="single", format_func=yn_format, label_visibility="visible")
hds = st.pills("Do you have a history of heart disease or stroke?", ['y','n'],key='hds', selection_mode="single", format_func=yn_format, label_visibility="visible")
cholesterol = st.text_input("Cholesterol: ")
asthma = st.pills("Do you have asthma?", ['y','n'], selection_mode="single",key='asthma', format_func=yn_format, label_visibility="visible")
immune_defic = st.pills("Do you have an immune deficiency?", ['y','n'],key='immune', selection_mode="single", format_func=yn_format, label_visibility="visible")
family_cancer = st.pills("Do you have a family history of cancer?", ['y','n'],key='cancer', selection_mode="single", format_func=yn_format, label_visibility="visible")
family_heart_disease = st.pills("Do you have a family history of heart disease or stroke?", ['y','n'], key='familyhds',selection_mode="single", format_func=yn_format, label_visibility="visible")
family_cholesterol = st.pills("Do you have a family history of high cholesterol?", ['y','n'],key='chol', selection_mode="single", format_func=yn_format, label_visibility="visible")
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

#st.write(f"You entered: {inputs}")

# Interactive Components
st.write('After you enter your personal information, enter payment type then click the button to calculate your expected insurance cost')
if policy_type=='v':
    payment_type=st.pills("Payment Type", ['Annual','Monthly'], selection_mode="single", label_visibility="visible")
elif policy_type in ['fl','fd']:
    payment_type=st.pills("Payment Type", ['Lump','Annual','Monthly','Compare Options'], selection_mode="single", label_visibility="visible")

if st.button("Click me"):
    st.write(actu_str(inputs,fv,age,policy_type,duration,payment_type))
