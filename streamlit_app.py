import streamlit as st
from neural_net import NeuralNet

pg = st.navigation([st.Page("main.py", title="Neural Net Life Cost Predictor"),st.Page("settings.py",title="Settings",icon='⚙️')])
if 'interest_rate' not in st.session_state:
    st.session_state["interest_rate"]=1
pg.run()