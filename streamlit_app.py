import streamlit as st
from neural_net import NeuralNet

pg = st.navigation([st.Page("main.py", title="Neural Net Life Cost Predictor",)])
pg.run()