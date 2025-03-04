import streamlit as st
from utils import st_get_inputs
from game_utils import get_yrs_left
from life_predict_utils import possible_options,bool_to_together_separat,life_predictor
from actu import get_mort_tab
import pandas as pd

st.header('Life Predictor')
st.subheader('Enter your info to get predictions about your life expectancy')
age=st.number_input('What\'s your current age?',max_value=79,value=21)
inputs=st_get_inputs()
life_p=life_predictor(inputs,age)

def check_options():
    st.session_state['check_options']=True

possible_funcs=life_p.get_possible_funcs()

st.button('Click me after you enter your information',on_click=check_options)
if st.session_state['check_options']:
    st.subheader(f'Remaining life expectancy of {get_yrs_left(life_p.person):.1f} years')
    chart_data = pd.DataFrame(get_mort_tab(life_p.person[1],inputs))
    st.bar_chart(chart_data,x_label='Years from today',y_label='Probability said year is when you will die')
    st.subheader('Select options to see how they would affect your remaining lifespace')
    options = st.multiselect(label='',options=possible_options)
    combine = st.radio(label='Consider options together or separate?',options=[True,False],format_func=bool_to_together_separat)
    if combine:
        life_p.multi_what_if(options)
    else:
        for i,option in enumerate(possible_options):
            if option in options:
                possible_funcs[i]()
