import streamlit as st
from neural_net import NeuralNet
from cloud_storage import load_user_data
import json
from google.cloud import storage
import google.auth
import os
from auto_emailer import string_to_list

#credentials, project = google.auth.default()
#st.write("Credentials:", credentials)

login = st.Page("login.py",title='Neural Net Life Login Page', icon='🏠')
life_insurance = st.Page("life_insurance_calculator.py", title="Neural Net Life Cost Predictor", icon='🧮')
life_predictor = st.Page('life_predictor.py',title='Life Predictor', icon='🧮')
death_predictor = st.Page("death_predictor.py",title="Death Predictor Game",icon='🎮')
settings = st.Page("settings.py",title="Settings",icon='⚙️')


pg = st.navigation([login,life_insurance,life_predictor,death_predictor,settings])
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
if 'prev_user_inputs' not in st.session_state or 'high_score' not in st.session_state:
    try:
        user_data=load_user_data(st.experimental_user.get('email'))
        try:
            st.session_state['prev_user_inputs']=string_to_list(user_data[1],convert_to_age=False)
        except:
            st.session_state['prev_user_inputs']=[None,None,None,None,None,None,None,None,
                                                None,None,None,None,None,None,None,None,
                                                None,None,None,None,None,None,None,None]
        try:
            st.session_state['high_score']=float(user_data[2])
        except:
            st.session_state['high_score']=0
    except:
        print("Error with loading user data at startup")
        st.session_state['prev_user_inputs']=[None,None,None,None,None,None,None,None,
                                                None,None,None,None,None,None,None,None,
                                                None,None,None,None,None,None,None,None]
        st.session_state['high_score']=0
pg.run()