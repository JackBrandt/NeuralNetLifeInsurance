import pandas as pd
import numpy as np
import streamlit as st
from actu import calculate_life_insurance_liability, get_mortality_table, actuarial_string
from neural_net import LifeNet
from faker import Faker
from utils import format_sex, format_risk_level
file_path = 'data.csv'

def np_to_int(mixed_array):
    """Convert elements in an array to integers where possible."""
    for i, value in enumerate(mixed_array):
        try:
            mixed_array[i] = int(value)
        except:
            pass
    return mixed_array

def generate_person(avoid_names=[]):
    """Generate a synthetic person with unique characteristics avoiding duplicate names."""
    fake = Faker()
    df = pd.read_csv(file_path, header=0)
    person = np_to_int(df.iloc[np.random.randint(0, df.shape[0])].to_numpy())
    # Ensure the age is below 80
    while person[0] > 79:
        person = np_to_int(df.iloc[np.random.randint(0, df.shape[0])].to_numpy())
    # Assign a gender-appropriate name
    if person[2] == 'm':
        name = fake.name_male()
        while name in avoid_names:
            name = fake.name_female()
    else:
        name = fake.name_female()
        while name in avoid_names:
            name = fake.name_female()
    person = np.append([name], person)
    return person

def generate_people(num_people):
    """Generate a list of unique people."""
    people = []
    avoid_names = []
    for i in range(num_people):
        people.append(generate_person(avoid_names))
        avoid_names.append(people[i][0])
    return people

def price_person(person, I):
    """Calculate insurance pricing for a person."""
    fv = 1250
    person = person[1:]
    age = person[0]
    inputs = person[1:]
    mort_tab = get_mortality_table(age, inputs)
    return calculate_life_insurance_liability(fv, I, mort_tab, 0)

def get_yrs_left(person):
    """Calculate expected years left for a person based on mortality data."""
    person = person[1:]
    age = person[0]
    def_yrs = 25 - age if age < 25 else 0
    inputs = person[1:]
    mort_tab = get_mortality_table(age, inputs)
    return actuarial_string(mort_tab, def_yrs)

def get_mus(people):
    """Get expected years left for a list of people."""
    return [get_yrs_left(person) for person in people]

def price_people(people, I=1):
    """Price insurance for a list of people."""
    return [price_person(person, I) for person in people]

def yn_to_does_not(yn):
    """Convert yes/no to affirmative/negative phrases."""
    return 'do' if yn == 'y' else 'do not'

def yn_to_bool(yn):
    """Convert yes/no string to boolean."""
    return yn == 'y'

def print_person(person):
    """Display detailed information about a person in Streamlit."""
    st.subheader(person[0] + ' is...')
    st.markdown('**'+ str(person[1])+'** years old')
    st.markdown('**' + str(person[2]) + '** pounds')
    st.markdown('**' + format_sex(person[3])+'**')
    st.markdown('**' + str(person[4]) + "** inches tall")
    st.markdown('Their blood pressure is **' + str(person[5])+'**')
    st.markdown('Their cholesterol is **' + str(person[19])+'**')
    st.markdown('They are on **' + str(person[8])+ '** medications')
    st.markdown('Their occupation hazard is **' + format_risk_level(person[9]).lower()+'**')
    st.markdown('Their lifestyle hazard is **' + format_risk_level(person[10]).lower()+'**')
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
        st.markdown('They **' + yn_to_does_not(person[17]) + '** have diabetes')
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
    """Display all people's information in Streamlit columns."""
    columns = st.columns(len(people), border=True, gap='medium')
    for i, col in enumerate(columns):
        with col:
            print_person(people[i])
