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

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

app = FastAPI()

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# preprocessing
def load_wav_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# embedding

def extract_embedding(wav_data, label):
  ''' run YAMNet to extract embedding from the wav data '''
  scores, embeddings, spectrogram = yamnet_model(wav_data)
  scores = tf.reshape(scores[:,106], [-1, 1])
  embeddings =  scores * embeddings
#  top_scores, top_indices = tf.math.top_k(scores[:,106], k=3)
#  top_embeddings = tf.gather(embeddings, top_indices)
  num_embeddings = tf.shape(embeddings)[0]
  return embeddings, tf.repeat(label, num_embeddings)


@app.get("/predict")
def predict(filename):
  wav = load_wav_16k_mono(filename)
  # yamnet model
  scores, embeddings, spectrogram = yamnet_model(wav)
  scores = scores[:,106]
  scores = tf.reshape(scores, [-1, 1])
  final_scores = scores * my_model(embeddings)
  final_scores = tf.reduce_sum(final_scores, axis=0)
  row_sum = tf.reduce_sum(final_scores)
  final_scores = tf.divide(final_scores, row_sum)
  final_score = pd.DataFrame(final_scores, columns = ['Probability'])
  final_score.index = le.inverse_transform(final_score.index)
  final_score = final_score.sort_values(by = 'Probability', ascending = False).applymap(lambda x: "{:.2%}".format(x))
  return final_score.head(3).index


my_model=tf.saved_model.load(
            "yamnet_bird_1", tags=None, options=None
        )
    assert model is not None



    filename = "/Users/rodolpheterlinden/Docs data/wav_files/aramides_cajaneus_455602.wav"
    prediction = model.predict(path_file_to_predict)


@app.get("/")
def index():

    return {"status": "ok"}
