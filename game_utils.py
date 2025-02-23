import pandas as pd
import numpy as np
import streamlit as st
from actu import life_liability_pv_mu,get_mort_tab,\
      years_left_mu
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


def generate_people(num_people):
    people=[]
    avoid_names=[]
    for i in range(num_people):
        people.append(generate_person(avoid_names))
        avoid_names.append(people[i][0])
    return people

def price_person(person,I):
    fv=1250
    person=person[1:]
    age=person[0]
    inputs=person[1:]
    mort_tab=get_mort_tab(age,inputs)
    return life_liability_pv_mu(fv,I,mort_tab,0)

def get_yrs_left(person):
    person=person[1:]
    age=person[0]
    if age<25:
        def_yrs=25-age
    else:
        def_yrs=0
    inputs=person[1:]
    mort_tab=get_mort_tab(age,inputs)
    return years_left_mu(mort_tab,def_yrs)

def get_mus(people):
    return [get_yrs_left(person) for person in people]

def price3people(person1,person2,person3,I=1):
    price1=price_person(person1,I)
    price2=price_person(person2,I)
    price3=price_person(person3,I)
    return price1,price2,price3

def price2people(person1,person2,I=1):
    price1=price_person(person1,I)
    price2=price_person(person2,I)
    return price1,price2

def price_people(people,I=1):
    return [price_person(person,I) for person in people]

def yn_to_does_not(yn):
    if yn=='y':
        return 'do'
    else:
        return 'do not'

def yn_to_bool(yn):
    if yn=='y':
        return True
    else:
        return False

def print_person(person):
    st.subheader(person[0] + ' is...')
    st.markdown('**'+ str(person[1])+'** years old')
    st.markdown('**' + str(person[2]) + '** pounds')
    st.markdown('**' + sex_format(person[3])+'**')
    st.markdown('**' + str(person[4]) + "** inches tall")
    st.markdown('Their blood pressure is **' + str(person[5])+'**')
    st.markdown('Their cholesterol is **' + str(person[19])+'**')
    st.markdown('They are on **' + str(person[8])+ '** medications')
    st.markdown('Their occupation harzard is **' + risk_num_format(person[9]).lower()+'**')
    st.markdown('Their lifestyle harzard is **' + risk_num_format(person[10]).lower()+'**')
    st.markdown('They have had **' + str(person[16])+ '** major surgeries')
    st.markdown('They have **' + str(person[14])+ '** drinks per week')
    if yn_to_bool(person[6]):
        st.markdown('They **smoke**')
    if yn_to_bool(person[7]):
        st.markdown('They use alternative forms of nicotine (like vaping or chewing tobacco)')
    if yn_to_bool(person[11]):
        st.markdown('They use **weed**')
    if yn_to_bool(person[12]):
        st.markdown('They use **opioids**')
    if yn_to_bool(person[13]):
        st.markdown('They use recreational drugs (besides alcohol, nicotine, weed, or opioids)')
    if yn_to_bool(person[15]):
        st.markdown('They have a history of **addiction**')
    if yn_to_bool(person[17]):
        st.markdown('They **' + yn_to_does_not(person[17])+ '** have diabetes')
    if yn_to_bool(person[18]):
        st.markdown('They have a **history of heart disease or stroke**')
    if yn_to_bool(person[20]):
        st.markdown('They have **asthma**')
    if yn_to_bool(person[21]):
        st.markdown('They have an **immune deficiency**')
    if yn_to_bool(person[22]):
        st.markdown('They have a **family history of cancer**')
    if yn_to_bool(person[23]):
        st.markdown('They have a *family history of heart disease or stroke*')
    if yn_to_bool(person[24]):
        st.markdown('They have a *family history of high cholesterol*')

def print_people(people):
    columns = st.columns(len(people),border=True,gap='medium')
    for i,col in enumerate(columns):
        with col:
            print_person(people[i])

def dp_print_header():
    score=st.session_state['score']
    st.title('Death Predictor Game')
    st.subheader('Who (statistically) has the longest left to live?')
    st.markdown('Guess correctly to gain points, guess wrongly to lose points')
    st.subheader(f'*Current Score:\t{round(score)}*')
    return score

def update_score(mu1, mu2,amount,age):
    st.session_state['guessed']=True
    score = st.session_state['score']
    if age==max(mu1,mu2):
        st.session_state['score']=score+amount
    else:
        st.session_state['score']=score-amount



if __name__ == "__main__":
    #person1,person2,person3=generate3people()
    #print(price3people(person1,person2,person3))
    pass