import streamlit as st
from utils import st_get_inputs
from game_utils import get_yrs_left,yn_to_bool,yn_to_does_not
from actu import get_mort_tab
import pandas as pd

st.header('Life Predictor')
st.subheader('Enter your info to get predictions about your life expectancy')
age=st.number_input('What\'s your current age?',max_value=79,value=21)
inputs=st_get_inputs()
person=['',age]
person.extend(inputs)
print(person)

def what_if(input_index, what_if_str,new_value):
    old_value=inputs[input_index]
    inputs[input_index]=new_value
    person=['',age]
    person.extend(inputs)
    st.subheader(f'Remaining life expectancy if {what_if_str}: {get_yrs_left(person):.1f} years')
    chart_data = pd.DataFrame(get_mort_tab(person[1],inputs))
    st.bar_chart(chart_data,x_label='Years from today',y_label='Probability said year is when you will die')
    inputs[input_index]=old_value # Return weight to normal

def yn_flip(index):
    if inputs[index]=='y':
        return 'n'
    else:
        return 'y'

def value_flip(index):
    if inputs[index]>1:
        return 1
    else:
        return 3

def multi_what_if(options,person):
    if len(options)==0:
        return
    option_indexes=[]
    old_values=[]
    for option in options:
        option_indexes.append(option_to_index[option])
    print(option_indexes)
    for index in option_indexes:
        old_values.append(inputs[index])
        print(index)
        match index:
            case 0: new_value=inputs[0]-20
            case 1: new_value=inputs[0]+20
            case 4|5|9|10|11|15|16: new_value=yn_flip(index)
            case 7|8: new_value=value_flip(index)
        inputs[index]=new_value
    chart_data = pd.DataFrame(get_mort_tab(person[1],inputs))
    what_if_str=options[0]
    try:
        for option in options[1:]:
            what_if_str+=', '+option
    except:
        pass
    person=['',age]
    person.extend(inputs)
    st.subheader(f'Remaining life expectancy if {what_if_str} all change: {get_yrs_left(person):.1f} years')
    st.bar_chart(chart_data,x_label='Years from today',y_label='Probability said year is when you will die')
    for i,index in enumerate(option_indexes):
        inputs[index]=old_values[i]

def yn_what_if(input_index,thing_str, positive_verb='starts', negative_verb='stops'):
    if yn_to_bool(inputs[input_index]):
        what_if(input_index,negative_verb+' '+thing_str,'n')
    else:
        what_if(input_index,positive_verb+' '+thing_str,'y')

def value_what_if(input_index,thing_str,check_value, other_value, positive_verb='increases', negative_verb='reduces'):
    if inputs[input_index]>check_value:
        what_if(input_index,negative_verb+' '+thing_str,check_value)
    else:
        what_if(input_index,positive_verb+' '+thing_str,other_value)

def check_options():
    st.session_state['check_options']=True

possible_options=['loses 20 pounds','gains 20 pounds','smoking','using other nic. products'\
            ,'work hazard','lifestyle hazard','using weed','using opiods','using recreational drugs','diabetes','heart disease']

option_to_index={'loses 20 pounds':0,
                 'gains 20 pounds':0,
                 'smoking':4,
                 'using other nic. products':5,
                 'work hazard':7,
                 'lifestyle hazard':8,
                 'using weed':9,
                 'using opiods':10,
                 'using recreational drugs':11,
                 'diabetes':15,
                 'heart disease':16
                 }

possible_funcs=[lambda:what_if(0,'loses 20 pounds', inputs[0]-20),
    lambda:what_if(0,'gains 20 pounds', inputs[0]+20),
    lambda:yn_what_if(4,'smoking'),
    lambda:yn_what_if(5,'using other nic. products'),
    lambda:value_what_if(7,'work hazard',1,3),
    lambda:value_what_if(8,'lifestyle hazard',1,3),
    lambda:yn_what_if(9,'using weed'),
    lambda:yn_what_if(10,'using opiods'),
    lambda:yn_what_if(11,'using recreational drugs'),
    lambda:yn_what_if(15,'diabetes','aquires','reverses'),
    lambda:yn_what_if(16,'heart disease','aquires','reverses')]

def bool_to_together_separat(pic):
    if pic:
        return 'together'
    return 'separate'

st.button('Click me after you enter your information',on_click=check_options)
if st.session_state['check_options']:
    st.subheader(f'Remaining life expectancy of {get_yrs_left(person):.1f} years')
    chart_data = pd.DataFrame(get_mort_tab(person[1],inputs))
    st.bar_chart(chart_data,x_label='Years from today',y_label='Probability said year is when you will die')
    st.subheader('Select options to see how they would affect your remaining lifespace')
    options = st.multiselect(label='',options=possible_options)
    combine = st.radio(label='Consider options together or separate?',options=[True,False],format_func=bool_to_together_separat)
    if combine:
        multi_what_if(options,person)
    else:
        for i,option in enumerate(possible_options):
            if option in options:
                possible_funcs[i]()
