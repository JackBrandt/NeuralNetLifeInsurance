import streamlit as st
from game_utils import dp_print_header, update_score,\
    print_people,people_setup,guess_button
from cloud_storage import update_user_data_item
from neural_net import NeuralNet

difficulty=1
people,mus,prices=people_setup(difficulty)
highscore, score = dp_print_header()
print_people(people)

update_w_price1 = lambda : update_score(mus,prices[0],mus[0])
update_w_price2 = lambda : update_score(mus,prices[1],mus[1])
update_w_price3 = lambda : update_score(mus,prices[2],mus[2])

def check_guess():
    points = prices[0]*(1-abs(mu_guess-mus[0])/7.5)
    st.session_state['score']=st.session_state['score']+points
    #print(f'Current score {st.session_state['score']}')
    #print(f'Current highscore: {st.session_state['']}')
    if st.session_state['score']>highscore:
        #print('High score')
        st.session_state['high_score']=st.session_state['score']
        try:
            update_user_data_item(st.experimental_user.get('email'), 2, st.session_state['high_score'])
        except AttributeError:
            pass
    st.session_state['guessed']=True

def print_guess_results():
    points = prices[0]*(1-abs(mu_guess-mus[0])/7.5)
    #print(f'points={points}')
    if points>=0:
        st.subheader('Good job!')
        if st.session_state['score']==st.session_state['high_score']:
            st.subheader('New highscore!')
        st.text(f'Correct answer was: {mus[0]:.1f}')
        st.text(f'Plus {points:.1f} points')
    else:
        st.subheader('Wrong!')
        st.text(f'Correct answer was: {mus[0]:.1f}')
        st.text(f'Minus {points:.1f} points')

if len(people)==1:
    mu_guess=st.slider('Expected Years Left',5.0,35.0,20.0,.1,)
    if st.button('Guess',on_click=check_guess,disabled=st.session_state['guessed']):
        print_guess_results()
        print(prices[0])
elif len(people)==2:
    col1,col2,col3,col4=st.columns((.17,.33,.2,.3))
    with col2:
        guess_button(0,update_w_price1,people,mus,prices)
    with col4:
        guess_button(1,update_w_price2,people,mus,prices)
elif len(people)==3:
    col1,col2,col3,col4,col5,col6=st.columns((.1,.2,.2,.2,.2,.2))
    with col2:
        guess_button(0,update_w_price1,people,mus,prices)
    with col4:
        guess_button(1,update_w_price2,people,mus,prices)
    with col6:
        guess_button(2,update_w_price3,people,mus,prices)

def next_round():
    st.session_state['guessed']=False
    st.session_state["people/prices/mu"]=None

if st.session_state['guessed']:
    if st.button('Next Round',on_click=next_round):
        pass
