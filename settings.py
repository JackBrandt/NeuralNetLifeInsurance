import streamlit as st
from utils import get_storage_function,get_loading_function
from cloud_storage import update_user_data_item
st.title("Settings")

store_i = get_storage_function("interest_rate")
load_i = get_loading_function("interest_rate")
def reset_highscore():
    try:
        update_user_data_item(st.experimental_user.email,2,0)
    except:
        pass
    st.session_state['high_score']=0
    st.session_state['score']=0

load_i()
st.number_input("Enter custom interest rate",key="_interest_rate",on_change=store_i)
st.button('Reset highscore',on_click=reset_highscore)
