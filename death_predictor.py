import streamlit as st
from game_utils import dp_print_header, update_score,\
    print_people,people_setup,guess_button
from neural_net import NeuralNet

difficulty=1

score = dp_print_header()
people,mus,prices=people_setup(2,1)
print_people(people)

update_w_price1 = lambda : update_score(mus[0],mus[1],prices[0],mus[0])
update_w_pricec2 = lambda : update_score(mus[0],mus[1],prices[1],mus[1])

col1,col2,col3,col4=st.columns((.13,.3,.2,.37))
with col2:
    guess_button(0,update_w_price1,people,mus,prices)
with col4:
    guess_button(1,update_w_pricec2,people,mus,prices)

def next_round():
    st.session_state['guessed']=False
    st.session_state["people/prices/mu"]=None

if st.session_state['guessed']:
    if st.button('Next Round',on_click=next_round):
        pass
