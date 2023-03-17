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
    audio_bytes = audio_path.read()
    st.audio(audio_bytes, format="audio/wav")
    if st.button("Plot an audiogram"):
        def audiogram(audio_data):
            with open('temp_wav_file.wav', 'wb') as wav_file:
                wav_file.write(audio_data)
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
    audio_bytes = audio_path.read()
    if st.button("Plot a spectrogram"):
        def plot_spectrogram(audio_bytes):
            with open('temp_wav_file.wav', 'wb') as wav_file:
                wav_file.write(audio_bytes)
            audio_data, sample_rate = librosa.load('temp_wav_file.wav')

            S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128, fmax=8000)
            S_DB = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(figsize=(14, 5))
            img = librosa.display.specshow(S_DB, x_axis='time', y_axis='mel', sr=sample_rate, ax=ax, fmax=8000)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set(title='Mel-frequency spectrogram')
            st.pyplot(fig)

        plot_spectrogram(audio_bytes)


def plot_spectrogram(audio_bytes):
    with io.BytesIO(audio_bytes) as audio_file:
        # Load audio data
        audio_data, sample_rate = librosa.load(audio_file)

        # Save audio data to disk
        with open('temp_wav_file.wav', 'wb') as wav_file:
            librosa.output.write_wav(wav_file, audio_data, sample_rate)

        # Load saved audio data
        audio_data, sample_rate = librosa.load('temp_wav_file.wav')

        # Compute spectrogram
        S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=128, fmax=8000)
        S_DB = librosa.power_to_db(S, ref=np.max)

        # Plot spectrogram
        fig, ax = plt.subplots(figsize=(14, 5))
        img = librosa.display.specshow(S_DB, x_axis='time', y_axis='mel', sr=sample_rate, ax=ax, fmax=8000)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        st.pyplot(fig)

if audio_path is not None:
    audio_bytes = audio_path.read()
    st.audio(audio_bytes, format="audio/wav")
    if st.button("Plot a spectrogram"):
        plot_spectrogram(audio_bytes)
