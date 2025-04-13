import streamlit as st
import pandas as pd
from cloud_storage import send_friend_request, set_friend

# Function to add a friend from potential to current
def add_friend(name):
    if name in st.session_state.potential_friends:
        try:
            st.session_state.current_friends.append(name)
        except AttributeError:
            st.session_state.current_friends=[name]
        st.session_state.potential_friends.remove(name)
        st.success(f'{name} has been added to your friends list!')
        set_friend(st.experimental_user.get('email'),name)
        send_friend_request(st.experimental_user.get('email'),name)
        st.rerun()
    else:
        st.error('This name is not in the potential friends list.')

def get_potential_friends():
    return [friend for friend in st.session_state.potential_friends if friend != '' and friend != st.experimental_user.get('email')]

# Streamlit interface
st.title('Friends Manager')

st.header('Current Friends')
if st.session_state.current_friends:
    for friend in st.session_state.current_friends:
        st.markdown(f"* {friend}")
else:
    st.write('No friends added yet.')

st.header('Potential Friends')
if st.session_state.potential_friends:
    potentials = get_potential_friends()
    for friend in potentials:
        st.markdown(f"* {friend}")
else:
    st.write('No potential friends available.')

st.header('Add a New Friend')
friend_to_add = st.selectbox('Select a friend to add:', get_potential_friends())
if st.button('Add Friend'):
    add_friend(friend_to_add)
