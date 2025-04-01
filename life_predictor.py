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
    '''
    Sets a Streamlit session state variable 'check_options' to True. 
    This function is typically used as a callback linked to a 
    Streamlit button to trigger updates or new calculations in 
    the user interface after user inputs have been registered.

    Parameters:
        None: This function does not take any arguments.

    Returns:
        None: This function does not return any values.
        It modifies the Streamlit session state directly.

    Usage:
        # This function is triggered by a Streamlit button.
        #  When the button is clicked, it sets the session state variable:
        st.button('Click me after you enter your information', 
        on_click=check_options)
        # This setup is used to enable further interactive
        #  elements in the UI based on the user's inputs.

    Note:
        - The function simply sets the 'check_options' state to True, 
        and it is assumed to be used within a logic block that 
        checks this state to activate further UI elements or calculations.
        - Ensure that this session state variable is properly handled 
        elsewhere in your Streamlit application to perform the 
        intended actions when its value is True.
        - This function is part of a user interaction flow 
        where conditional UI elements depend on the user confirming their input.
    '''
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
