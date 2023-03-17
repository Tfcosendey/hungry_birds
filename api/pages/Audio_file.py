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
        def plot_spectrogram(audio_data):
            with open('temp_wav_file.wav', 'wb') as wav_file:
                wav_file.write(audio_data)
            y, sr = librosa.load('temp_wav_file.wav')

            S = librosa.stft(y)
            S_db = librosa.amplitude_to_db(abs(S))
            fig, ax = plt.subplots(figsize=(14, 5))

            img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title('Spectrogram')
            st.pyplot(fig)

        plot_spectrogram(audio_bytes)


# mel-spectogram


# different

aramides_1='../raw_data/songs/Aramides cajaneus/18517.mp3'
aramides_2 = '../raw_data/songs/Aramides cajaneus/7187.mp3'
aramides_3 = '../raw_data/songs/Aramides cajaneus/7258.mp3'

atilla_1 ='../raw_data/songs/Attila rufus/1509.mp3'
atilla_2 = '../raw_data/songs/Attila rufus/483.mp3'
atilla_3 = '../raw_data/songs/Attila rufus/1318.mp3'

automolus_1 ='../raw_data/songs/Automolus leucophthalmus/463.mp3'
automolus_2 = '../raw_data/songs/Automolus leucophthalmus/34539.mp3'
automolus_3 ='../raw_data/songs/Automolus leucophthalmus/80769.mp3'


filenames = [[aramides_1,aramides_2,aramides_3],[atilla_1,atilla_2,atilla_3],[automolus_1,automolus_2,automolus_3]]

titles = [["aramides_1","aramides_2","aramides_3"],["atilla_1","atilla_2","atilla_3"],["automolus_1","automolus_2","automolus_3"]]

# Create 3x3 grid of subplots
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10,10))

# Loop through each audio file and plot spectrogram on corresponding subplot
for i in range(3):
    for j in range(3):
        audio_bird = st.file_uploader(f"Upload {filenames[i][j]}", type=['mp3', 'wav'])
        if audio_bird:
            y, sr = librosa.load(audio_path)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            ax = axs[i][j]
            img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                            y_axis='mel', fmax=8000, x_axis='time', ax=ax)
            axs[i][j].set_title(titles[i][j])
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            axs[i][j].set_xlim([0, 15])
plt.tight_layout()
st.pyplot()
