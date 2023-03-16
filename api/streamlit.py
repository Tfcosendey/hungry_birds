import streamlit as st
import os

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

upload_file = st.file_uploader("Choose an audio file", type=[".wav"], accept_multiple_files=False)

if upload_file is not None:
    audio_bytes = upload_file.read()
    st.audio(audio_bytes,format="audio/wav")

with st.form(key='params_for_api'):
    your_path_to_wav_file = st.number_input('Your path to wav file', value=45)

    st.form_submit_button('Make prediction')

params = your_path_to_wav_file

response = requests.get(url, params=params)
