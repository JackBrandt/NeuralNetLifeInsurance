import streamlit as st
from cloud_storage import load_user_data, unload_user_data
import os
import subprocess as sp

#st.write("GOOGLE_APPLICATION_CREDENTIALS:", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))

st.header('Welcome to Neural Net Life')
st.subheader("Please login below, than navigate to any of the following pages on the left")

try:
    if not st.experimental_user.is_logged_in:
        if st.button("Log in"):
            st.login()
            #user_data=load_user_data(st.experimental_user.email)
            #st.write(user_data)
    else:
        user_data=(st.session_state['prev_user_inputs'],st.session_state['high_score'])
        st.write(user_data)
        st.write(f"Hello, {st.experimental_user.name}!")
        #st.write(st.experimental_user.items())
        if st.button("Log out"):
            st.logout()
            #user_data=unload_user_data(user_data)
            #st.write(user_data)
except:
    if st.button("Log in"):
            st.login()
