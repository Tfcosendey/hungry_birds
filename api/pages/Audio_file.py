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
y, sr = librosa.load(audio_path)
time = librosa.times_like(y)

# Draft the audiogram etc
if audio_path is not None:
    audio_bytes = audio_path.read()
    st.audio(audio_bytes,format="audio/wav")
    if st.button("Convert your audio"):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(time, y)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Audio waveform')
        st.pyplot(fig)
