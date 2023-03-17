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

        def path_to_image_html(gen_sp):
            return f'<img src="home/tcosendey/code/Tfcosendey/hungry_birds/Images/Aramides cajaneus.jpeg" width="64">'
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
            final_score['Img'] = final_score.index.map(lambda x: x)
            #final_score['Img'] = final_score['Probability'].apply(st.image(path_to_image_html))
            print(final_score)
            html = convert_df(final_score)
            os.remove('temp_wav_file.wav')
            return html
        html = predict(audio_bytes)

        st.markdown(
            html,
            unsafe_allow_html=True)

        st.download_button(
            label="Download data as HTML",
            data=html,
            file_name='output.html',
            mime='text/html',
 )
#        st.image(path_to_image_html('Aramides cajaneus'))
        # st.image(PIL.image.open('https://www.countries-ofthe-world.com/flags-normal/flag-of-United-States-of-America.png'))
