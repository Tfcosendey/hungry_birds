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

# data processing
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

# Processing the latitudes and longitudes
birds_df['lat'] = pd.to_numeric(birds_df['lat'])
birds_df['lng'] = pd.to_numeric(birds_df['lng'])

# Cleaning null and "na" values
birds_df = birds_df[(birds_df['lat'] != 0) & (birds_df['lng'] != 0)]
birds_df = birds_df.dropna()

# Focus on a radius of 500km around Rio de Janeiro (Brazil)
point = (-22.91216,-43.17501)
distance_km = 500

rio_df = birds_df[birds_df.apply(lambda row: distance(point, (row['lat'], row['lng'])).km <= distance_km, axis=1)]

# Creating a column merging generic and specific name of the birds
rio_df['gen_sp'] = rio_df['gen'] + ' ' + rio_df['sp']
birds_df['gen_sp'] = birds_df['gen'] + ' ' + birds_df['sp']

# Taking all those species observed <500km from Rio and keeping those where we have > 100 recordings
rio_sp = rio_df['gen_sp'].unique()
birds_df_filt = birds_df[birds_df['gen_sp'].isin(rio_sp)]
counts = birds_df_filt['gen_sp'].value_counts()
filtered_df = birds_df_filt[birds_df_filt['gen_sp'].isin(counts[counts > 100].index)]

# Removing unknown birds
filtered_df = filtered_df[filtered_df['gen_sp'] != 'Mystery mystery']

# Processing the index
filtered_df.reset_index(drop=True,inplace=True)

############
############

# Key figures
st.markdown("""## Data exploration""")
st.text("**1. Number of audio recordings** = 16.322")
st.text("**2. Number of different species** = 121 ")
st.text("**3. Number of different features** = 39 (ex: rating, length, autor, location, type of sound,...) ")

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
