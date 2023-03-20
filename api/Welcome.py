import streamlit as st
import os

import numpy as np
import pandas as pd
import requests
from PIL import Image

### welcome page

st.set_page_config(
    page_title="Welcome!",
    page_icon="🦤",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""# Welcome!""")
st.markdown("""## To Hungry Birds classification tool""")

st.markdown("""- In the World, :red[1 out of 8] bird species are under threat of extinction.""")
st.markdown("""- Brazil is the :red[2nd country] in the world in number of globally threatened birds""")

image = Image.open("../Images/Image 20-03-23 à 15.08.jpg")

st.image(image, width=500)

st.markdown("""Researchers use population monitoring to understand how birds react to changes in the environment and conservation efforts. But many bird species across the world are isolated in difficult-to-access habitats. With physical monitoring difficult, sound recording is a solution to sicentists. Known as bioacoustic monitoring, this approach could provide a :green[passive, low labor, and cost-effective] strategy for studying endangered bird populations""")

st.markdown(""":green[Group members]: Valentina, Thales, Rodolphe""")
