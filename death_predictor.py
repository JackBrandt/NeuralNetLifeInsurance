import streamlit as st
from game_utils import dp_print_header, update_score,\
    print_people,generate_people,get_mus,price_people
from neural_net import NeuralNet

difficulty=1

score = dp_print_header()

def people_setup(num_people):
    if st.session_state['people/prices/mu'] is not None:
        people=st.session_state["people/prices/mu"][0]
        prices=st.session_state["people/prices/mu"][1]
        mus=st.session_state["people/prices/mu"][2]
        return people,mus,prices
    match num_people:
        case 1:
            pass
        case 2:
            people=generate_people(2)
            # Calculate how much each of them would cost for a life insurance policy
            mus=get_mus(people)
            prices=price_people(people,1)
            # Ensure they are separate enough
            while abs(mus[0]-mus[1])<difficulty or abs(mus[0]-mus[1])>(difficulty+10):#Maybe this value can be a difficulty setting
                people=generate_people(2)
                mus=get_mus(people)
                prices=price_people(people,1)
            print(prices)
        case 3:
            pass
        case _:
            raise KeyError
        # Then save
    st.session_state["people/prices/mu"]=[people,prices,mus]
    return people,mus,prices

people,mus,prices=people_setup(2)

# Display them
print_people(people)

# Have user pick an option
# If they're right add the value of the policy
# If they're wrong subtract the value of the selected policy
# Let them keep picking until they get it right
# Then repeat

update_w_price1 = lambda : update_score(mus[0],mus[1],prices[0],mus[0])
update_w_pricec2 = lambda : update_score(mus[0],mus[1],prices[1],mus[1])


col1,col2,col3,col4=st.columns((.13,.3,.2,.37))
with col2:
    if st.button(people[0][0],on_click=update_w_price1,disabled=st.session_state['guessed']):
        if mus[0]>mus[1]:
            st.subheader('Correct!')
            st.text('Remaining Life Expectancies of:')
            st.text(f'{mus[0]:.1f} vs {mus[1]:.1f}')
            st.text(f'Plus {round(prices[0])} points')
            #score+=price1
            #st.session_state['score']=score
        else:
            st.subheader('Wrong!')
            st.text('Remaining Life Expectancies of:')
            st.text(f'{mus[0]:.1f} vs {mus[1]:.1f}')
            print(prices[0])
            st.text(f'Minus {round(prices[0])} points')
            #score-=price1
            #st.session_state['score']=score
with col4:
    if st.button(people[1][0],key='person2',on_click=update_w_pricec2,disabled=st.session_state['guessed']):
        if mus[0]<mus[1]:
            st.subheader('Correct!')
            st.text('Remaining Life Expectancies of:')
            st.text(f'{mus[0]:.1f} vs {mus[1]:.1f}')
            st.text(f'Plus {round(prices[1])} points')
            #score+=price2
            #st.session_state['score']=score
        else:
            st.subheader('Wrong!')
            st.text('Remaining Life Expectancies of:')
            st.text(f'{mus[0]:.1f} vs {mus[1]:.1f}')
            st.text(f'Minus {round(prices[1])} points')
            #score-=price2
            #st.session_state['score']=score

def next_round():
    st.session_state['guessed']=False
    st.session_state["people/prices/mu"]=None

if st.session_state['guessed']:
    if st.button('Next Round',on_click=next_round):
        pass
