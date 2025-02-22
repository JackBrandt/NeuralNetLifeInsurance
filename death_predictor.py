import streamlit as st
from game_utils import generate2people,price2people,print2people,get2yrsleft
from neural_net import NeuralNet
from utils import get_loading_function,get_storage_function
import streamlit_extras.stylable_container as stesc
from streamlit.components.v1 import html

score=st.session_state['score']
st.title('Death Predictor Game')
st.subheader('Who (statistically) has the longest left to live?')
st.markdown('Guess correctly to gain points, guess wrongly to lose points')
st.subheader(f'*Current Score:\t{round(score)}*')

difficulty=1

if st.session_state["people/prices/mu"]==None:# Generate 3 random people
    person1,person2 = generate2people()
    # Calculate how much each of them would cost for a life insurance policy
    mu1,mu2=get2yrsleft(person1,person2)
    price1,price2 = price2people(person1,person2)
    # Ensure they are separate enough
    while abs(mu1-mu2)<difficulty or abs(mu1-mu2)>(difficulty+10):#Maybe this value can be a difficulty setting
        person1,person2 = generate2people()
        mu1,mu2=get2yrsleft(person1,person2)
        price1,price2 = price2people(person1,person2)
    print(price1,price2)
    # Then save
    st.session_state["people/prices/mu"]=[(person1,person2),(price1,price2),(mu1,mu2)]
else:
    person1,person2=st.session_state["people/prices/mu"][0]
    price1,price2=st.session_state["people/prices/mu"][1]
    mu1,mu2=st.session_state["people/prices/mu"][2]
#print(price1,price2)


# Display them
print2people(person1,person2)
# Have user pick an option
# If they're right add the value of the policy
# If they're wrong subtract the value of the selected policy
# Let them keep picking until they get it right
# Then repeat
col1,col2,col3,col4=st.columns((.13,.3,.2,.37))
def update_score(amount,age):
    st.session_state['guessed']=True
    if age==max(mu1,mu2):
        st.session_state['score']=score+amount
    else:
        st.session_state['score']=score-amount
update_w_price1 = lambda : update_score(price1,mu1)
update_w_pricec2 = lambda : update_score(price2,mu2)

def next_round():
    st.session_state['guessed']=False
    st.session_state["people/prices/mu"]=None

with col2:
    if st.button(person1[0],on_click=update_w_price1,disabled=st.session_state['guessed']):
        if mu1>mu2:
            st.subheader('Correct!')
            st.text('Remaining Life Expectancies of:')
            st.text(f'{mu1:.1f} vs {mu2:.1f}')
            st.text(f'Plus {round(price1)} points')
            #score+=price1
            #st.session_state['score']=score
        else:
            st.subheader('Wrong!')
            st.text('Remaining Life Expectancies of:')
            st.text(f'{mu1:.1f} vs {mu2:.1f}')
            st.text(f'Minus {round(price1)} points')
            #score-=price1
            #st.session_state['score']=score
with col4:
    if st.button(person2[0],key='person2',on_click=update_w_pricec2,disabled=st.session_state['guessed']):
        if mu1<mu2:
            st.subheader('Correct!')
            st.text('Remaining Life Expectancies of:')
            st.text(f'{mu1:.1f} vs {mu2:.1f}')
            st.text(f'Plus {round(price2)} points')
            #score+=price2
            #st.session_state['score']=score
        else:
            st.subheader('Wrong!')
            st.text('Remaining Life Expectancies of:')
            st.text(f'{mu1:.1f} vs {mu2:.1f}')
            st.text(f'Minus {round(price2)} points')
            #score-=price2
            #st.session_state['score']=score

if st.session_state['guessed']:
    if st.button('Next Round',on_click=next_round):
        pass
