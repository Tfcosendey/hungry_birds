import streamlit as st
import os

import numpy as np
import pandas as pd
import requests

url = "http://127.0.0.1:8000/"
url_image = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.justdial.com%2FVadodara%2FHungry-Birds-Restaurant-Opposite-Sterling-Hospital-Opposite-Indraprashtha-Complex-Beside-Inox-Wadi-Wadi-Opposite-Indrapr-Ellora-Park%2F0265PX265-X265-150107161932-H4Q9_BZDET&psig=AOvVaw0kdzLf44DDx47NN_uXfm1_&ust=1679061558108000&source=images&cd=vfe&ved=0CBAQjRxqFwoTCLiU-PjN4P0CFQAAAAAdAAAAABAD"

st.sidebar.image(url_image, width=100)


st.set_page_config(
    page_title="Hungry Birds",
    page_icon="🦤")

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
