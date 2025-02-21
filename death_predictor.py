import streamlit as st
from game_utils import generate2people,price2people,print2people
from neural_net import NeuralNet

score=st.session_state['score']
st.title('Death Predictor Game')
st.header('Current Score:'+str(score))
st.markdown('Checkout the people below and predict which of them is *most likely* to have the *greatest number of years left* in their life')
st.markdown('You lose points for every wrong attempt and gain points for every correct one (points are proportional to present value of liability for a 125,000 full life insurance policy for the chosen person)')


if st.session_state["people/prices"]==None:# Generate 3 random people
    person1,person2 = generate2people()
    # Calculate how much each of them would cost for a life insurance policy
    price1,price2 = price2people(person1,person2)
    # Ensure they are separate enough
    while abs(price1-price2)<15000:#Maybe this value can be a difficulty setting
        person1,person2 = generate2people()
        price1,price2 = price2people(person1,person2)
    print(price1,price2)
    # Then save
    st.session_state["people/prices"]=[(person1,person2),(price1,price2)]
else:
    person1,person2=st.session_state["people/prices"][0]
    price1,price2=st.session_state["people/prices"][1]
print(price1,price2)


# Display them
st.header('Consider these 2 people: ')
print2people(person1,person2)
# Have user pick an option
# If they're right add the value of the policy
# If they're wrong subtract the value of the selected policy
# Let them keep picking until they get it right
# Then repeat
st.header('Which of them do you think statistically has longer left to live?')
col1,col2,col3,col4=st.columns((.13,.3,.2,.37))
with col2:
    if st.button(person1[0]):
        pass
with col4:
    if st.button(person2[0]):
        pass
