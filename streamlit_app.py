import streamlit as st
from neural_net import NeuralNet

main_page = st.Page("main.py", title="Neural Net Life Cost Predictor", icon='🧮')
death_predictor = st.Page("death_predictor.py",title="Death Predictor Game",icon='🎮')
settings = st.Page("settings.py",title="Settings",icon='⚙️')

pg = st.navigation([main_page,death_predictor,settings])
if 'interest_rate' not in st.session_state:
    st.session_state["interest_rate"]=1
if 'people/prices' not in st.session_state:
    st.session_state["people/prices"]=None
if 'score' not in st.session_state:
    st.session_state['score']=0
if 'guessed' not in st.session_state:
    st.session_state['guessed']=False

js = '''
<script>
    var body = window.parent.document.querySelector(".main");
    console.log(body);
    body.scrollTop = 0;
</script>
'''
pg.run()