import pandas as pd
import numpy as np
from faker import Faker
import streamlit as st

from actu import calculate_life_insurance_liability, get_mortality_table, actuarial_string
from neural_net import LifeNet
from utils import format_sex, format_risk_level

FILE_PATH = 'data.csv'

def convert_np_to_integers(array):
    """Converts elements of a numpy array to integers if possible."""
    for i, value in enumerate(array):
        try:
            array[i] = int(value)
        except ValueError:
            pass
    return array

def generate_individual(avoid_names=None):
    """Generates a random person's details from the dataset ensuring unique names."""
    if avoid_names is None:
        avoid_names = []
        
    fake = Faker()
    df = pd.read_csv(FILE_PATH)
    individual = convert_np_to_integers(df.iloc[np.random.randint(len(df))].to_numpy())
    
    # Regenerate if age is over 79
    while individual[0] > 79:
        individual = convert_np_to_integers(df.iloc[np.random.randint(len(df))].to_numpy())
    
    name = fake.name_male() if individual[2] == 'm' else fake.name_female()
    while name in avoid_names:
        name = fake.name_female()
    
    return np.append([name], individual)

def generate_multiple_individuals(number):
    """Generates multiple unique individuals."""
    individuals = []
    avoid_names = []
    for _ in range(number):
        person = generate_individual(avoid_names)
        individuals.append(person)
        avoid_names.append(person[0])
    return individuals

def price_individual(individual, interest_rate):
    """Calculates the price for insuring an individual."""
    age = individual[1]
    inputs = individual[2:]
    mortality_table = get_mortality_table(age, inputs)
    face_value = 1250
    return calculate_life_insurance_liability(face_value, interest_rate, mortality_table, 0)

def years_remaining(individual):
    """Estimates years left for an individual based on actuarial data."""
    age = individual[1]
    default_years = 25 - age if age < 25 else 0
    inputs = individual[2:]
    mortality_table = get_mortality_table(age, inputs)
    return actuarial_string(mortality_table, default_years)

def calculate_years_remaining_for_all(individuals):
    """Calculates years remaining for all individuals in a list."""
    return [years_remaining(individual) for individual in individuals]

def price_all_individuals(individuals, interest_rate=1):
    """Prices insurance for all individuals in a list."""
    return [price_individual(individual, interest_rate) for individual in individuals]

def convert_yn_to_verb(yn):
    """Converts a yes/no value to appropriate verb."""
    return 'do' if yn == 'y' else 'do not'

def convert_yn_to_boolean(yn):
    """Converts a yes/no value to boolean."""
    return yn == 'y'

def display_individual_details(individual):
    """Displays detailed information about an individual using Streamlit."""
    st.subheader(f'{individual[0]} is...')
    attributes = {
        'Age': individual[1],
        'Weight': individual[2],
        'Sex': format_sex(individual[3]),
        'Height': individual[4],
        'Blood Pressure': individual[5],
        'Cholesterol': individual[19],
        'Medications': individual[8],
        'Occupational Hazard': format_risk_level(individual[9]).lower(),
        'Lifestyle Hazard': format_risk_level(individual[10]).lower(),
        'Major Surgeries': individual[16],
        'Drinks per Week': individual[14],
    }
    for key, value in attributes.items():
        st.markdown(f'**{key}:** {value}')

    # Additional health details
    health_details = [
        (6, 'They **smoke**'),
        (7, 'They use alternative forms of nicotine (like vaping or chewing tobacco)'),
        (11, 'They use **weed**'),
        (12, 'They use **opioids**'),
        (13, 'They use recreational drugs (besides alcohol, nicotine, weed, or opioids)'),
        (15, 'They have a history of **addiction**'),
        (17, f'They **{convert_yn_to_verb(individual[17])}** have diabetes'),
        (18, 'They have a **history of heart disease or stroke**'),
        (20, 'They have **asthma**'),
        (21, 'They have an **immune deficiency**'),
        (22, 'They have a **family history of cancer**'),
        (23, 'They have a **family history of heart disease or stroke**'),
        (24, 'They have a **family history of high cholesterol**'),
    ]

    for index, message in health_details:
        if convert_yn_to_boolean(individual[index]):
            st.markdown(message)

def display_individuals_in_columns(individuals):
    """Display individuals' details in Streamlit columns."""
    columns = st.columns(len(individuals))
    for col, individual in zip(columns, individuals):
        with col:
            display_individual_details(individual)

def setup_individuals(difficulty):
    """Initial setup for generating and displaying individuals based on difficulty settings."""
    num_people = np.random.randint(1, 4)
    st.set_page_config(layout='wide' if num_people > 2 else 'centered')
    
    individuals = generate_multiple_individuals(num_people)
    remaining_years = calculate_years_remaining_for_all(individuals)
    
    # Ensure significant age gaps
    while any(abs(ry1 - ry2) < difficulty or abs(ry1 - ry2) > (difficulty + 10) for ry1 in remaining_years for ry2 in remaining_years if ry1 != ry2):
        individuals = generate_multiple_individuals(num_people)
        remaining_years = calculate_years_remaining_for_all(individuals)
    
    prices = price_all_individuals(individuals)
    print(prices)  # Display prices for debug purposes

    return individuals, remaining_years, prices

def play_game():
    """Streamlit app to play a guessing game about individuals' longevity."""
    st.title('Death Predictor Game')
    difficulty = st.sidebar.selectbox('Select Difficulty:', [10, 20, 30], index=1)
    individuals, remaining_years, prices = setup_individuals(difficulty)
    guessed = st.session_state.get('guessed', False)
    
    for i, individual in enumerate(individuals):
        if st.button(individual[0], key=f'person{i}', on_click=lambda: play_game(remaining_years, prices[i], i), disabled=guessed):
            correct = remaining_years[i] == max(remaining_years)
            st.subheader('Correct!' if correct else 'Wrong!')
            for year, price in zip(remaining_years, prices):
                st.text(f'{year:.1f} years: {"+" if correct else "-"}{round(price)} points')
            st.session_state['guessed'] = True
