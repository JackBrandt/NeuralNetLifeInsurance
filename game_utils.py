import pandas as pd
import numpy as np
import streamlit as st
from actu import life_liability_pv_mu,get_mort_tab
from neural_net import NeuralNet
from faker import Faker
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
    name = fake.first_name()
    while name in avoid_names:
        name = fake.first_name()
    df = pd.read_csv(file_path, header=0)
    person=np_to_int(df.iloc[np.random.randint(0,df.shape[0])].to_numpy())
    while person[0]>79:
        person=np_to_int(df.iloc[np.random.randint(0,df.shape[0])].to_numpy())
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

def print_person(person):
    st.header(person[0])
    st.text(person[1:])

def print3people(person1,person2,person3):
    print_person(person1)
    print_person(person2)
    print_person(person3)

if __name__ == "__main__":
    person1,person2,person3=generate3people()
    print(price3people(person1,person2,person3))