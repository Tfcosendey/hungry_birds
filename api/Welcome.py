import streamlit as st
import os

import numpy as np
import pandas as pd
import requests

### welcome page

st.set_page_config(
    page_title="Welcome!",
    page_icon="🦤",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""# Welcome!""")
st.markdown("""## To Hungry Birds classification tool""")
