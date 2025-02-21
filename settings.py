import streamlit as st
from utils import get_storage_function,get_loading_function
st.title("Settings")

store_i = get_storage_function("interest_rate")
load_i = get_loading_function("interest_rate")

load_i()
st.number_input("Enter custom interest rate",key="_interest_rate",on_change=store_i)
