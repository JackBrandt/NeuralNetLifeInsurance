import streamlit as st
from actu import actu_str
from utils import policy_type_format, st_get_inputs
from cloud_storage import update_user_data_item
# Title
st.title("Neural Net Life Cost Predictor")

# Main stuff
# TODO: Replace text_input with number_input with sensible parameters (e.g., height should be positive)

policy_type=st.pills("Enter Desired Policy Type", ['fl','fd','v'],key='pol_type', selection_mode="single", format_func=policy_type_format, label_visibility="visible")
age=st.number_input('What\'s your current age?',max_value=79,value=25,min_value= 25 if policy_type=='v' else 0)
if policy_type=='fd':
    duration=st.number_input("Policy Duration (years): ", 1 if age>24 else 26-age,int(120-age),20,1)
else:
    duration=None
fv=st.number_input("Policy Amount",125000)

inputs=st_get_inputs(st.session_state['prev_user_inputs'])
st.write(f"You entered: {inputs}")

# Interactive Components
st.write('After you enter your personal information, enter payment type then click the button to calculate your expected insurance cost')
if policy_type=='v':
    payment_type=st.pills("Payment Type", ['Annual','Monthly'], selection_mode="single", label_visibility="visible")
elif policy_type in ['fl','fd']:
    payment_type=st.pills("Payment Type", ['Lump','Annual','Monthly','Compare Options'], selection_mode="single", label_visibility="visible")

make_policy=st.checkbox('Check this box to specify you want to create a model policy, leave it unchecked to only calculate the expected cost')

if st.button("Click me"):
    st.session_state['prev_user_inputs']=inputs
    try:
        st.write(update_user_data_item(st.experimental_user.email,1,inputs))
    except:
        pass
    st.write(actu_str(inputs,fv,age,policy_type,duration,payment_type,st.session_state['interest_rate']))
    if make_policy:
        pass
