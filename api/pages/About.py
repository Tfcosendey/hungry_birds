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

st.set_page_config(
    page_title="About",
    page_icon="🦤",
    layout="wide",
    initial_sidebar_state="expanded")

def get_data():
    birds_df = pd.DataFrame()
    for page in range(130):
        html = f'https://xeno-canto.org/api/2/recordings?query=cnt:brazil&page={page+1}'
        print(f'requesting page {page+1}')
        response = requests.get(html).json()
        new_df = pd.DataFrame(response['recordings'])
        birds_df = pd.concat([birds_df, new_df], ignore_index=True)
    return birds_df
def save_csv(df,name):
    path = os.path.expanduser(f'~/{name}.csv')
    df.to_csv(path, index = False)

birds_df = get_data()
birds_df.shape
