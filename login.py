import streamlit as st
from cloud_storage import load_user_data, unload_user_data,get_username,set_username
import os
import subprocess as sp

st.header('Welcome to Neural Net Life')

st.title("Login/Logout")

#st.switch_page("life_insurance_calculator.py")
if st.experimental_user.get('email') is None:
    st.subheader("Please login below, than navigate to any of the following pages on the left")
    if st.button("Log in"):
        st.login()
else:
    username = get_username(st.experimental_user.get('email'))
    #st.write(username)
    #st.write(st.experimental_user.get('email'))
    if username is not None and username != '':
        st.write(f'Hello {username}')
    else:
        st.write(f'Hello {st.experimental_user.get("email")}')
        new_username = st.text_input('Create an username')
        if st.button("Set username"):
            set_username(st.experimental_user.get('email'),new_username)
            st.rerun()
    if st.button("Log out"):
        st.logout()

try:
    load_user_data(st.experimental_user.get('email'))
    st.write('Cloud storage access validated')
except:
    st.write('Cloud storage access failed...')