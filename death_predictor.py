import streamlit as st
from game_utils import generate3people,price3people,print3people
from neural_net import NeuralNet

st.title('Death Predictor Game')
st.text('Checkout the people below and predict which of them is most likely to have the greatest number of years left in their life')
st.text('You lose points for every wrong attempt and gain points for every correct one (points are proportional to present value of liability for a 125,000 full life insurance policy for the chosen person)')
score=0
# Generate 3 random people
person1,person2,person3 = generate3people()
# Calculate how much each of them would cost for a life insurance policy
price1,price2,price3 = price3people(person1,person2,person3)
print(price1,price2,price3)
# Ensure they are separate enough
while min(abs(price1-price2),abs(price1-price3),abs(price2-price3))<10000:#Maybe this value can be a difficulty setting
    person1,person2,person3 = generate3people()
    price1,price2,price3 = price3people(person1,person2,person3)
    print(price1,price2,price3)
# Display them
print3people(person1,person2,person3)
# Have user pick an option
# If they're right add the value of the policy
# If they're wrong subtract the value of the selected policy
# Let them keep picking until they get it right
# Then repeat
#st.button('Option 1', key='option1', on_click=check_answer1())
#st.button('Option 2', key='option1', on_click=check_answer2())
#st.button('Option 3', key='option1', on_click=check_answer3())