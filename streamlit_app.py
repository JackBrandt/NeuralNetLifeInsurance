import streamlit as st
from neural_net import NeuralNet
from cloud_storage import load_user_data
import json

login = st.Page("login.py",title='Neural Net Life Login Page', icon='🏠')
life_insurance = st.Page("life_insurance_calculator.py", title="Neural Net Life Cost Predictor", icon='🧮')
life_predictor = st.Page('life_predictor.py',title='Life Predictor', icon='🧮')
death_predictor = st.Page("death_predictor.py",title="Death Predictor Game",icon='🎮')
settings = st.Page("settings.py",title="Settings",icon='⚙️')
friends = st.Page("friends.py",title='Friends',icon='😎')

pg = st.navigation([login,life_insurance,life_predictor,death_predictor,friends,settings])
if 'interest_rate' not in st.session_state:
    st.session_state["interest_rate"]=1
if 'people/prices/mu' not in st.session_state:
    st.session_state["people/prices/mu"]=None
if 'score' not in st.session_state:
    st.session_state['score']=0
if 'guessed' not in st.session_state:
    st.session_state['guessed']=False
if 'check_options' not in st.session_state:
    st.session_state['check_options']=False
if st.experimental_user.is_logged_in:
    if 'prev_user_inputs' not in st.session_state:
        try:
            inputs=load_user_data(st.experimental_user.email)[0][1]
            inputs = inputs.replace("None", "null")
            inputs = inputs.replace("'", '"')
            print(inputs)
            st.session_state['prev_user_inputs']=json.loads(inputs)
        except:
            st.session_state['prev_user_inputs']=[None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]
else:
    if 'prev_user_inputs' not in st.session_state:
        st.session_state['prev_user_inputs']=[None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,]
if 'current_friends' not in st.session_state:
    st.session_state.current_friends = ['Alice', 'Bob', 'Charlie']
if 'potential_friends' not in st.session_state:
    st.session_state.potential_friends = ['David', 'Eva', 'Frank']


pg.run()