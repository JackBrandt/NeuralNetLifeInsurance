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
    '''
    Calculates and updates the user's score based on their guess
      compared to a predefined value (mu). The score
        is calculated based on how close the guess (mu_guess) 
        is to the actual value (mus[0]), with the maximum score 
        reduced proportionally by the absolute error, scaled by a factor.
          The function updates the session state to record 
          the new score and marks that a guess has been made.

    Parameters:
        None: This function does not take any arguments.

    Returns:
        None: This function does not return any values. 
        It updates the session state within the application context.

    Usage:
        # Assuming that the necessary session state variables 
        # ('score' and 'guessed') and
        #  global variables ('prices', 'mu_guess', 'mus') are set:
        >>> st.session_state['score'] = 0  # Initialize score
        >>> st.session_state['guessed'] = False  # Initialize guessed status
        >>> check_guess()
        # This will update the 'score' and 'guessed' state 
        # based on the user's guess.

    Note:
        - It is assumed that 'prices', 'mu_guess', and 'mus'
          are defined in the global scope before this function is called.
        - 'prices[0]' is used to determine the maximum points 
        available for the guess.
        - 'mu_guess' should be the user's current guess, 
        and 'mus[0]' should be the correct value against
          which the guess is evaluated.
        - The error scaling factor of 7.5 adjusts the 
        sensitivity of scoring to the magnitude of the guess error.
        - The function modifies the Streamlit session state,
          which is used to maintain state across interactions 
          in a Streamlit application. It specifically updates
            the 'score' and 'guessed' variables to reflect the outcome of the latest guess.
        - Proper error handling should be in place to manage
          cases where necessary variables are not initialized or incorrectly set up.
    '''
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
    '''
    Calculates the points based on the user's guess against a
      predefined correct value and displays the results 
      using Streamlit's UI components. The function prints 
      whether the guess was close enough (good job) or
        not (wrong) and displays the exact number of points awarded or deducted.

    Parameters:
        None: This function does not take any arguments.

    Returns:
        None: This function does not return any values. 
        It updates the UI components of the Streamlit 
        application to reflect the results of the user's guess.

    Usage:
        # To be called after a user makes a guess to
        #  provide immediate feedback:
        >>> print_guess_results()
        # This will update the Streamlit interface with 
        # the results of the guess, including 
        # whether the guess was accurate and the points awarded.

    Note:
        - The function uses global variables 'prices', 'mu_guess', 
        and 'mus' assumed to be defined in the environment 
        where this function is executed.
        - 'prices[0]' is used to calculate the maximum points
          available for the guess.
        - 'mu_guess' is the user's guess, and 'mus[0]' 
        is the correct answer against which the guess is evaluated.
        - The scoring mechanism adjusts the points based 
        on the absolute error of the guess, scaled by 7.5, 
        and could result in positive or negative points.
        - Streamlit's UI components (`st.subheader`, `st.text`) 
        are used to render the results directly in the web interface.
        - This function should only be triggered afte
        r a guess has been made to ensure that all required variables are initialized.
    '''
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
    '''
    Resets specific Streamlit session state variables to 
    prepare the application for the next round of interaction. 
    This function is typically used in multi-round applications 
    or games to clear previous states and ensure that 
    the application is ready for new user inputs.

    Parameters:
        None: This function does not take any arguments.

    Returns:
        None: This function does not return any values.
          It modifies the Streamlit session state directly.

    Usage:
        # This function is usually tied to a button 
        # in the Streamlit interface:
        if st.session_state['guessed']:
            if st.button('Next Round', on_click=next_round):
                pass
        # When the 'Next Round' button is clicked, 
        # this function will be invoked, resetting the session state as required.

    Note:
        - This function sets the 'guessed' state to False and 
        clears the "people/prices/mu" state to None, which are
          assumed to be used elsewhere in the application to 
          track the state of user interactions and game logic.
        - The function is typically invoked through a callback 
        linked to a Streamlit button as shown in the usage example.
          This setup requires that 'guessed' is a boolean flag
            in the session state indicating whether the user 
            has made a guess in the current round.
        - Ensure that all session state variables manipulated 
        by this function are properly initialized at the 
        start of the application to avoid errors.
        - It is important to design the UI and state management 
        in a way that does not allow this function to be called 
        out of sequence, which could disrupt the application 
        flow or user experience.
    '''
    st.session_state['guessed']=False
    st.session_state["people/prices/mu"]=None

if st.session_state['guessed']:
    if st.button('Next Round',on_click=next_round):
        pass
