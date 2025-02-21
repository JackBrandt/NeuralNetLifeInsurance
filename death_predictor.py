import streamlit as st
from game_utils import generate2people,price2people,print2people
from neural_net import NeuralNet
from utils import get_loading_function,get_storage_function
import streamlit_extras.stylable_container as stesc
from streamlit.components.v1 import html
from streamlit_app import js

score=st.session_state['score']
st.title('Death Predictor Game')
st.subheader('Who (statistically) has the longest left to live?')
st.markdown('Guess correctly to gain points, guess wrongly to lose points')
st.subheader('*Current Score:\t'+str(score)+'*')


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
print2people(person1,person2)
# Have user pick an option
# If they're right add the value of the policy
# If they're wrong subtract the value of the selected policy
# Let them keep picking until they get it right
# Then repeat
col1,col2,col3,col4=st.columns((.13,.3,.2,.37))
def update_score(amount):
    st.session_state['guessed']=True
    if amount==min(price1,price2):
        st.session_state['score']=score+amount
    else:
        st.session_state['score']=score-amount
update_w_price1 = lambda : update_score(price1)
update_w_pricec2 = lambda : update_score(price2)

def next_round():
    st.session_state['guessed']=False
    st.session_state["people/prices"]=None

with col2:
    if st.button(person1[0],on_click=update_w_price1,disabled=st.session_state['guessed']):
        if price1<price2:
            st.subheader('Correct!')
            st.text(f'Plus {price1:.2f} points')
            #score+=price1
            #st.session_state['score']=score
        else:
            st.subheader('Wrong!')
            st.text(f'Minus {price1:.2f} points')
            #score-=price1
            #st.session_state['score']=score
with col4:
    if st.button(person2[0],key='person2',on_click=update_w_pricec2,disabled=st.session_state['guessed']):
        if price1>price2:
            st.subheader('Correct!')
            st.text(f'Plus {price2:.2f} points')
            #score+=price2
            #st.session_state['score']=score
        else:
            st.subheader('Wrong!')
            st.text(f'Minus {price2:.2f} points')
            #score-=price2
            #st.session_state['score']=score

if st.session_state['guessed']:
    if st.button('Next Round',on_click=next_round):
        pass