import streamlit as st
import os
import glob as glob
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import librosa
import numpy as np
import PIL.Image
from IPython.display import Image, HTML

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

        img_dict = pd.read_csv('Images/img_df.csv',index_col='gen_sp').to_dict()['links']

        def path_to_image_html(link):
            return f'<img src="{link}" width="120" height="120" >'
        def convert_df(input_df):
            return input_df.to_html(escape=False, formatters=dict(Img=path_to_image_html))

        def predict(file):
            with open('temp_wav_file.wav', 'wb') as wav_file:
                wav_file.write(file)
            wav = load_wav_16k_mono('temp_wav_file.wav')
            # yamnet model
            scores, embeddings, spectrogram = yamnet_model(wav)
            scores = scores[:,106]
            scores = tf.reshape(scores, [-1, 1])
            final_scores = scores * my_model(embeddings)
            final_scores = tf.reduce_sum(final_scores, axis=0)
            row_sum = tf.reduce_sum(final_scores)
            final_scores = tf.divide(final_scores, row_sum)
            final_score = pd.DataFrame(final_scores, columns = ['Probability'])
            with open('yamnet_full/label_encoder.pkl', 'rb') as model_file:
                le = pickle.load(model_file)
            final_score.index = le.inverse_transform(final_score.index)
            final_score = final_score.sort_values(by = 'Probability', ascending = False).applymap(lambda x: "{:.2%}".format(x)).head(10)
            final_score['Img'] = final_score.index.map(img_dict)
            html = convert_df(final_score)
            os.remove('temp_wav_file.wav')
            # Splitting the table into two side-by-side tables
            table_left = final_score.iloc[::2]
            table_right = final_score.iloc[1::2]

            # Centering the tables
            table_left_html = f"<div style='text-align:center; display:inline-block;'>{convert_df(table_left)}</div>"
            table_right_html = f"<div style='text-align:center; display:inline-block;'>{convert_df(table_right)}</div>"

            # Combining the tables
            html = f"<div style='text-align:center;'>{table_left_html}{table_right_html}</div>"

            return html

        html = predict(audio_bytes)

        st.markdown(
            html,
            unsafe_allow_html=True
)
        st.markdown("""### The final model archieved a top-10-accuracy of **55%**, although it can definitely be improved it is already **6.7x** better than a dummy model (8.2%).""")
