# import streamlit as st

# # Title of the page
# st.title('Friends')

# # List to store the names of friends
# if 'friends_list' not in st.session_state:
#     st.session_state.friends_list = []

# # Displaying current friends
# st.header('Current Friends')
# if st.session_state.friends_list:
#     for friend in st.session_state.friends_list:
#         st.write(friend)
# else:
#     st.write('No friends added yet.')

# # Section to add a new friend
# st.header('Add a New Friend')
# new_friend = st.text_input('Enter the name of a friend')
# if st.button('Add Friend'):
#     if new_friend:
#         st.session_state.friends_list.append(new_friend)
#         st.success(f'{new_friend} has been added to your friends list!')
#     else:
#         st.error('Please enter a name.')

import streamlit as st
import pandas as pd
from cloud_storage import send_friend_request,set_friend

# Initialize session state variables if they don't exist


# Function to add a friend from potential to current
def add_friend(name):
    if name in st.session_state.potential_friends:
        st.session_state.current_friends.append(name)
        st.session_state.potential_friends.remove(name)
        st.success(f'{name} has been added to your friends list!')
         # Save updated lists to CSV
        set_friend(st.experimental_user.get('email'),name)
    else:
        st.error('This name is not in the potential friends list.')

# Streamlit interface
st.title('Friends Manager')

st.header('Current Friends')
if st.session_state.current_friends:
    st.write(", ".join(st.session_state.current_friends))
else:
    st.write('No friends added yet.')

st.header('Pending Friends Request')
if st.session_state.pending_friends_request:
    st.write(", ".join(st.session_state.pending_friends_request))
else:
    st.write('No friends added yet.')

st.header('Potential Friends')
if st.session_state.potential_friends:
    st.write(", ".join(st.session_state.potential_friends))
else:
    st.write('No potential friends available.')

st.header('Add a New Friend')
friend_to_add = st.selectbox('Select a friend to add:', st.session_state.potential_friends)
if st.button('Add Friend'):
    add_friend(friend_to_add)
    send_friend_request(st.experimental_user.get('email'),friend_to_add) 
