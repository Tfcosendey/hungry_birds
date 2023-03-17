import streamlit as st
import os
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import seaborn as sns
import os
import urllib.request
import plotly.express as px
from geopy.distance import distance
import plotly.graph_objects as pgo
import librosa
import librosa.display
import IPython.display as ipd
from IPython.display import Audio
from pydub import AudioSegment
from scipy.io import wavfile
from tempfile import mktemp
import os
import io
from PIL import Image

import numpy as np
import pandas as pd
import requests

# Layout
st.set_page_config(
    page_title="Audio file",
    page_icon="🦤",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""# Hungry Birds Project
## What is an audio file?""")

#####
#####

# Upload file
audio_path = st.file_uploader("Choose an audio file", type=[".wav"], accept_multiple_files=False)


#audiogram
if audio_path is not None:

    if st.button("Plot an audiogram"):
        audio_bytes = audio_path.read()
        st.audio(audio_bytes, format="audio/wav")
        def audiogram(audio_data):
            with open('temp_wav_file.wav', 'wb') as wav_file:
                wav_file.write(audio_data)
            wav_file.close()
            y, sr = librosa.load('temp_wav_file.wav')

            time = librosa.times_like(y)
            fig, ax = plt.subplots(figsize=(14, 5))

            ax.plot(time, y)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Audio waveform')
            st.pyplot(fig)

        audiogram(audio_bytes)
#spectogram
if audio_path is not None:
    if st.button("Plot a spectrogram"):
        audio_bytes_ = audio_path.read()
        def plot_spectrogram(audio_data):
            with open('temp_wav_file.wav', 'wb') as wav_file_:
                wav_file_.write(audio_data)
            wav_file_.close()
            y_, sr_ = librosa.load('temp_wav_file.wav')

            S = librosa.stft(y_)
            S_db = librosa.amplitude_to_db(abs(S))
            fig_, ax_ = plt.subplots(figsize=(14, 5))

            img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr_, ax=ax_)
            fig_.colorbar(img, ax=ax_, format='%+2.0f dB')
            ax_.set_title('Spectrogram')
            st.pyplot(fig_)

        plot_spectrogram(audio_bytes_)

# mel-spectogram


# different
if audio_path is not None:
    audio_bytes = audio_path.read()
    if st.button("Plot differences between species"):
        image = Image.open("../hungry_birds/Images/specs.jpeg")
        # Display image
        st.image(image, use_column_width=True)
