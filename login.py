import streamlit as st
from cloud_storage import load_user_data, unload_user_data
import os
import subprocess as sp

st.header('Welcome to Neural Net Life')
st.subheader("Please login below, than navigate to any of the following pages on the left")

import streamlit as st


st.title("Login/Logout")
try:
    st.session_state.get("authentication_status")
    st.switch_page("life_insurance_calculator.py")
except:
    if st.button("Log in"):
        st.login()
