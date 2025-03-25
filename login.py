import streamlit as st

st.header('Welcome to Neural Net Life')
st.subheader("Please login below, than navigate to any of the following pages on the left")

if not st.experimental_user.is_logged_in:
    if st.button("Log in"):
        st.login()
else:
    st.write(f"Hello, {st.experimental_user.name}!")
    st.write(st.experimental_user.items())
    if st.button("Log out"):
        st.logout()