import streamlit as st
import os

import numpy as np
import pandas as pd
import requests


url = "http://127.0.0.1:8000/"
url_image = "/Users/rodolpheterlinden/Desktop/Rodolphe/Projects/Wagon/hb.jpg"


st.set_page_config(
    page_title="Hungry Birds",
    page_icon="🦤",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""# Hungry birds Tool
## Which bird specy is singing in your garden?""")

upload_file = st.file_uploader("Choose an audio file", type=[".wav"], accept_multiple_files=False)

if upload_file is not None:
    audio_bytes = upload_file.read()
    st.audio(audio_bytes,format="audio/wav")
    if st.button("Make prediction"):
        # Send the file to the API for prediction
        files = {"file": upload_file}
        response = requests.post(url, files=files)

        # Get the prediction from the response
        prediction = response.json()["prediction"]

        st.success(f"It is very likely that the bird species is: {prediction}")
