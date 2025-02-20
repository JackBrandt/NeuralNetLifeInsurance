import streamlit as st
st.title("Settings")
def store_value():
    # Copy the value to the permanent key
    st.session_state["interest_rate"] = st.session_state["_interest_rate"]

st.session_state["_interest_rate"] = st.session_state["interest_rate"]
st.number_input("Enter custom interest rate",key="_interest_rate",on_change=store_value)
