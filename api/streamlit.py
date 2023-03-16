import streamlit as st
import os

import numpy as np
import pandas as pd
import requests

url = "http://127.0.0.1:8000/"

st.set_page_config(
    page_title="Hungry Birds",
    page_icon="🦤")

st.markdown("""# Hungry birds Tool
## Which bird specy is singing in your garden?""")

upload_file = st.file_uploader("Choose an audio file", type=[".wav"], accept_multiple_files=False)

if upload_file is not None:
    audio_bytes = upload_file.read()
    st.audio(audio_bytes,format="audio/wav")
    st.form_submit_button('Make prediction')


params = upload_file

response = requests.post(url, params=params)

prediction = response.json()

st.header(f'It is very likely that the bird specy is:{prediction}')
