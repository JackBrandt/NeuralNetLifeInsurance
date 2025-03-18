import streamlit as st
from neural_net import NeuralNet

if not st.experimental_user.is_logged_in:
    if st.button("Log in"):
        st.login()
else:
    if st.button("Log out"):
        st.logout()
    st.write(f"Hello, {st.experimental_user.name}!")

life_insurance = st.Page("life_insurance_calculator.py", title="Neural Net Life Cost Predictor", icon='🧮')
life_predictor = st.Page('life_predictor.py',title='Life Predictor', icon='🧮')
death_predictor = st.Page("death_predictor.py",title="Death Predictor Game",icon='🎮')
settings = st.Page("settings.py",title="Settings",icon='⚙️')

pg = st.navigation([life_insurance,life_predictor,death_predictor,settings])
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

pg.run()