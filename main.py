import streamlit as st

# Title
st.title("Death Predictors")

# Sidebar
st.sidebar.header("Sidebar Example")
user_input = st.sidebar.text_input("Enter some text:")

# Main Content
st.write("Hello, Streamlit!")
st.write(f"You entered: {user_input}")

# Interactive Components
if st.button("Click me"):
    st.success("Button clicked!")
