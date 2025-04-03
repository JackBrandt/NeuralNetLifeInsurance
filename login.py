import streamlit as st
from cloud_storage import load_user_data, unload_user_data
import os
import subprocess as sp

st.header('Welcome to Neural Net Life')
st.subheader("Please login below, than navigate to any of the following pages on the left")

st.title("Login/Logout")
st.write(st.experimental_user.get('email'))
#st.switch_page("life_insurance_calculator.py")
if st.experimental_user.get('email') is None:
    if st.button("Log in"):
        st.login()
else:
    if st.button("Log out"):
        st.logout()



