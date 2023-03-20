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

# Layout
st.set_page_config(
    page_title="About",
    page_icon="🦤",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""# Hungry Birds Project""")

#####
#####

filtered_df = pd.read_csv("../hungry_birds/notebooks/filtered_df.csv")

############
############

# Key figures
st.markdown("""## Data exploration""")
st.markdown("""1. Number of audio recordings = 16.322""")
st.markdown("""2. Number of different species = 121 """)
st.markdown("""3. Number of different features = 39 (ex: rating, length, autor, location, type of sound,...) """)

# table per year
fig, ax = plt.subplots(figsize=(16, 6))

filtered_df['date'] = pd.to_datetime(filtered_df['date'], format='%Y-%m-%d', errors='coerce')
filtered_df['year'] = filtered_df['date'].dt.year
year_counts = filtered_df['year'].value_counts()
ax.bar(year_counts.index, year_counts.values, color = "darkgreen")

ax.set_xlabel('Year')
ax.set_ylabel('Number of Recordings')
ax.set_title('Number of Recordings per Year', fontsize=20, fontweight='bold')

st.pyplot(fig)

# table with rates

fig, ax = plt.subplots(figsize=(16, 6))


ratings = filtered_df['q'].values.tolist()
occurrences = [ratings.count('A'), ratings.count('B'), ratings.count('C'), ratings.count('D'), ratings.count('E'),ratings.count('no score')]
colors = ['darkgreen', 'forestgreen', 'gold', 'orange', 'red', 'gray']

plt.bar(['A', 'B', 'C', 'D','E','no score'], occurrences, color=colors)
plt.grid(axis='y', alpha=0.1)
plt.title('Rating Occurrences', fontsize=20, fontweight='bold')
plt.xlabel('Rating')
plt.ylabel('Occurrences')

# Display plot in Streamlit
st.pyplot(fig)


# Map
st.markdown("""## Mapping our data""")
def display_filtered_df_on_map(filtered_df):
    fig = px.scatter_mapbox(filtered_df, lat='lat', lon='lng', zoom=4, height=800, color='gen_sp')
    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig)

display_filtered_df_on_map(filtered_df)
