import streamlit as st

import numpy as np
import pandas as pd
import requests

url = "http://localhost:8000/"

st.set_page_config(
    page_title="Hungry Birds",
    page_icon="🦤")

st.markdown("""# Hungry birds Tool
## Which bird specy is singing in your garden?
Import your .wav file""")

with st.form(key='params_for_api'):
    your_path_to_wav_file = st.number_input('Your path to wav file', value=45)

    st.form_submit_button('Make prediction')

params = your_path_to_wav_file

response = requests.get(url, params=params)
