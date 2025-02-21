import streamlit as st
from game_utils import generate2people,price2people,print2people
from neural_net import NeuralNet

st.title('Death Predictor Game')
st.markdown('Checkout the people below and predict which of them is *most likely* to have the *greatest number of years left* in their life')
st.markdown('You lose points for every wrong attempt and gain points for every correct one (points are proportional to present value of liability for a 125,000 full life insurance policy for the chosen person)')
score=0
# Generate 3 random people
person1,person2 = generate2people()
# Calculate how much each of them would cost for a life insurance policy
price1,price2 = price2people(person1,person2)
print(price1,price2)
# Ensure they are separate enough
while abs(price1-price2)<15000:#Maybe this value can be a difficulty setting
    person1,person2 = generate2people()
    price1,price2 = price2people(person1,person2)
    print(price1,price2)
# Display them
print2people(person1,person2)
# Have user pick an option
# If they're right add the value of the policy
# If they're wrong subtract the value of the selected policy
# Let them keep picking until they get it right
# Then repeat
#st.button('Option 1', key='option1', on_click=check_answer1())
#st.button('Option 2', key='option1', on_click=check_answer2())
#st.button('Option 3', key='option1', on_click=check_answer3())