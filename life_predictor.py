import streamlit as st
from utils import st_get_inputs
from game_utils import get_yrs_left

st.header('Life Predictor')
st.subheader('Enter your info to get predictions about your life expectancy')
age=st.number_input('What\'s your current age?',max_value=79,value=21)
inputs=st_get_inputs()
person=['',age]
person.extend(inputs)
print(person)

if st.button('Click me after you enter your information'):
    st.write(f'Remaining life expectancy of {get_yrs_left(person):.1f} years')