import pandas as pd
import numpy as np
import streamlit as st
from actu import life_liability_pv_mu,get_mort_tab
from neural_net import NeuralNet
from faker import Faker
from utils import sex_format,risk_num_format
file_path='data.csv'

def np_to_int(mixed_array):
    for i,value in enumerate(mixed_array):
        try:
            mixed_array[i]=int(value)
        except:
            pass
    return mixed_array

def generate_person(avoid_names=[]):
    fake = Faker()
    df = pd.read_csv(file_path, header=0)
    person=np_to_int(df.iloc[np.random.randint(0,df.shape[0])].to_numpy())
    while person[0]>79:
        person=np_to_int(df.iloc[np.random.randint(0,df.shape[0])].to_numpy())
    if person[2]=='m':
        name = fake.name_male()
        while name in avoid_names:
            name = fake.name_female()
    else:
        name = fake.name_female()
        while name in avoid_names:
            name = fake.name_female()
    person = np.append([name],[person])
    return person

def generate3people():
    # Probably easiest just to sample 3 random people from data.csv
    person1=generate_person()
    #print(person1)
    person2=generate_person([person1[0]])
    #print(person2)
    person3=generate_person([person1[0],person2[0]])
    #print(person3)
    return person1,person2,person3
#generate3people()

def generate2people():
    # Probably easiest just to sample 3 random people from data.csv
    person1=generate_person()
    #print(person1)
    person2=generate_person([person1[0]])
    #print(person2)
    return person1,person2

def price_person(person,I):
    fv=125000
    person=person[1:]
    age=person[0]
    inputs=person[1:]
    mort_tab=get_mort_tab(age,inputs)
    return life_liability_pv_mu(fv,I,mort_tab,0)

def price3people(person1,person2,person3,I=1):
    price1=price_person(person1,I)
    price2=price_person(person2,I)
    price3=price_person(person3,I)
    return price1,price2,price3

def price2people(person1,person2,I=1):
    price1=price_person(person1,I)
    price2=price_person(person2,I)
    return price1,price2

def yn_to_does_not(yn):
    if yn=='y':
        return 'do'
    else:
        return 'do not'

def print_person(person):
    st.header(person[0] + ' is...')
    st.markdown('**'+ str(person[1])+'** years old')
    st.markdown('Weighs **' + str(person[2]) + '** pounds')
    st.markdown('**' + sex_format(person[3]).lower()+'**')
    st.markdown('**' + str(person[4]) + "** inches tall")
    st.markdown('Their blood pressure is **' + str(person[5])+'**')
    st.markdown('They **' + yn_to_does_not(person[6])+ '** smoke')
    st.markdown('They **' + yn_to_does_not(person[7])+ '** use other forms of nicotine (like vaping or chewing tobacco)')
    st.markdown('They are on **' + str(person[8])+ '** medications')
    st.markdown('Their occupation harzard is **' + risk_num_format(person[9]).lower()+'**')
    st.markdown('Their lifestyle harzard is **' + risk_num_format(person[10]).lower()+'**')
    st.markdown('They **' + yn_to_does_not(person[11])+ '** use weed')
    print(person[11])
    st.markdown('They **' + yn_to_does_not(person[12])+ '** use opiods')
    st.markdown('They **' + yn_to_does_not(person[13])+ '** use other drugs')
    st.markdown('They have **' + str(person[14])+ '** drinks per week')
    st.markdown('They **' + yn_to_does_not(person[15])+ '** have a history of addiction')
    st.markdown('They have had **' + str(person[16])+ '** major surgeries')
    st.markdown('They **' + yn_to_does_not(person[17])+ '** have diabetes')
    st.markdown('They **' + yn_to_does_not(person[18])+ '** have a history of heart disease or stroke')
    st.markdown('Their cholesterol is **' + str(person[19])+'**')
    st.markdown('They **' + yn_to_does_not(person[20])+ '** have asthma')
    st.markdown('They **' + yn_to_does_not(person[21])+ '** have an immune deficiency')
    st.markdown('They **' + yn_to_does_not(person[22])+ '** have a family history of cancer')
    st.markdown('They **' + yn_to_does_not(person[23])+ '** have a family history of heart disease or stroke')
    st.markdown('They **' + yn_to_does_not(person[24])+ '** have a family history of high cholesterol')

def print3people(person1,person2,person3):
    col1, col2, col3 = st.columns(3,border=True,gap='medium')

    with col1:
        print_person(person1)
    with col2:
        print_person(person2)
    with col3:
        print_person(person3)

def print2people(person1,person2):
    col1, col2 = st.columns(2,border=True,gap='medium')

    with col1:
        print_person(person1)
    with col2:
        print_person(person2)

if __name__ == "__main__":
    person1,person2,person3=generate3people()
    print(price3people(person1,person2,person3))