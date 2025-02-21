import streamlit as st
from neural_net import NeuralNet

main_page = st.Page("main.py", title="Neural Net Life Cost Predictor")
death_predictor = st.Page("death_predictor.py",title="Death Predictor Game",icon='🎮')
settings = st.Page("settings.py",title="Settings",icon='⚙️')

pg = st.navigation([main_page,death_predictor,settings])
if 'interest_rate' not in st.session_state:
    st.session_state["interest_rate"]=1
pg.run()