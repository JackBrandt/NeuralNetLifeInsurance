import streamlit as st
import pandas as pd
from actu import get_mort_tab
from game_utils import get_yrs_left, yn_to_bool


class life_predictor():
    def __init__(self,inputs,age):
        self.inputs=inputs
        self.age=age
        person=['',age]
        person.extend(inputs)
        self.person=person

    def what_if(self,input_index, what_if_str,new_value):
        old_value=self.inputs[input_index]
        self.inputs[input_index]=new_value
        person=['',self.age]
        person.extend(self.inputs)
        st.subheader(f'Remaining life expectancy if {what_if_str}: {get_yrs_left(person):.1f} years')
        chart_data = pd.DataFrame(get_mort_tab(person[1],self.inputs))
        st.bar_chart(chart_data,x_label='Years from today',y_label='Probability said year is when you will die')
        self.inputs[input_index]=old_value # Return weight to normal

    def yn_flip(self,index):
        if self.inputs[index]=='y':
            return 'n'
        else:
            return 'y'

    def value_flip(self,index):
        if self.inputs[index]>1:
            return 1
        else:
            return 3

    def multi_what_if(self,options):
        if len(options)==0:
            return
        option_indexes=[]
        old_values=[]
        for option in options:
            option_indexes.append(option_to_index[option])
        print(option_indexes)
        for index in option_indexes:
            old_values.append(self.inputs[index])
            print(index)
            match index:
                case 0: new_value=self.inputs[0]-20
                case 1: new_value=self.inputs[0]+20
                case 4|5|9|10|11|15|16: new_value=self.yn_flip(index)
                case 7|8: new_value=self.value_flip(index)
            self.inputs[index]=new_value
        chart_data = pd.DataFrame(get_mort_tab(self.person[1],self.inputs))
        what_if_str=options[0]
        try:
            for option in options[1:]:
                what_if_str+=', '+option
        except:
            pass
        person=['',self.age]
        person.extend(self.inputs)
        st.subheader(f'Remaining life expectancy if {what_if_str} all change: {get_yrs_left(person):.1f} years')
        st.bar_chart(chart_data,x_label='Years from today',y_label='Probability said year is when you will die')
        for i,index in enumerate(option_indexes):
            self.inputs[index]=old_values[i]

    def yn_what_if(self,input_index,thing_str, positive_verb='starts', negative_verb='stops'):
        if yn_to_bool(self.inputs[input_index]):
            self.what_if(input_index,negative_verb+' '+thing_str,'n')
        else:
            self.what_if(input_index,positive_verb+' '+thing_str,'y')

    def value_what_if(self,input_index,thing_str,check_value, other_value, positive_verb='increases', negative_verb='reduces'):
        if self.inputs[input_index]>check_value:
            self.what_if(input_index,negative_verb+' '+thing_str,check_value)
        else:
            self.what_if(input_index,positive_verb+' '+thing_str,other_value)

    def get_possible_funcs(self):
        return [lambda:self.what_if(0,'loses 20 pounds', self.inputs[0]-20),
            lambda:self.what_if(0,'gains 20 pounds', self.inputs[0]+20),
            lambda:self.yn_what_if(4,'smoking'),
            lambda:self.yn_what_if(5,'using other nic. products'),
            lambda:self.value_what_if(7,'work hazard',1,3),
            lambda:self.value_what_if(8,'lifestyle hazard',1,3),
            lambda:self.yn_what_if(9,'using weed'),
            lambda:self.yn_what_if(10,'using opiods'),
            lambda:self.yn_what_if(11,'using recreational drugs'),
            lambda:self.yn_what_if(15,'diabetes','aquires','reverses'),
            lambda:self.yn_what_if(16,'heart disease','aquires','reverses')]

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

def bool_to_together_separat(pic):
    if pic:
        return 'together'
    return 'separate'
