import streamlit as st
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import numpy as np
import io
import glob as glob
import os
import io
from sklearn.model_selection import train_test_split

from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pickle

import tensorflow as tf
import tensorflow_hub as hub
import librosa
from fastapi import FastAPI, File, UploadFile

import io
from urllib.request import urlopen
import librosa
import pydub
import soundfile as sf
import aiofiles
import numba

import numpy as np
import pandas as pd
import requests

# layout
st.set_page_config(
    page_title="The model",
    page_icon="🦤",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""# Hungry Birds Prediction Tool
## Which bird specy is singing in your garden?""")

upload_file = st.file_uploader("Choose an audio file", type=[".wav"], accept_multiple_files=False)

#upload the song
if upload_file is not None:
    audio_bytes = upload_file.read()
    st.audio(audio_bytes,format="audio/wav")
    if st.button("Make prediction"):
        #importing the model
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print(os.getcwd())
        my_model = tf.keras.models.load_model("yamnet_full", compile=False)
        # my_model=tf.saved_model.load(
        #            "/home/tcosendey/code/Tfcosendey/hungry_birds/yamnet_full")
        assert my_model is not None

        # preprocessing
        def load_wav_16k_mono(filename):
            """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
            wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
            wav = tf.convert_to_tensor(wav, dtype=tf.float32)
            return wav

        # embedding

        def extract_embedding(wav_data, label):
            scores, embeddings, spectrogram = yamnet_model(wav_data)
            scores = tf.reshape(scores[:,106], [-1, 1])
            embeddings =  scores * embeddings
            num_embeddings = tf.shape(embeddings)[0]
            return embeddings, tf.repeat(label, num_embeddings)

        #the model

        filename = "/home/tcosendey/code/Tfcosendey/hungry_birds/Drymophila ochropyga.wav"

        def predict(file: UploadFile):
            with aiofiles.open(file, 'wb') as out_file:
                content = file.read()  # async read
                out_file.write(content)  # async write
            wav = load_wav_16k_mono(file)
            # yamnet model
            scores, embeddings, spectrogram = yamnet_model(wav)
            scores = scores[:,106]
            scores = tf.reshape(scores, [-1, 1])
            final_scores = scores * my_model(embeddings)
            final_scores = tf.reduce_sum(final_scores, axis=0)
            row_sum = tf.reduce_sum(final_scores)
            final_scores = tf.divide(final_scores, row_sum)
            final_score = pd.DataFrame(final_scores, columns = ['Probability'])
            with open('yamnet_full/label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
            final_score.index = le.inverse_transform(final_score.index)
            final_score = final_score.sort_values(by = 'Probability', ascending = False).applymap(lambda x: "{:.2%}".format(x))
            print(final_score.head(10).index)
            os.remove(file)
            return final_score.head(10)
    st.table(predict(audio_bytes))
