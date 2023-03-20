import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pickle
import librosa
import matplotlib.pyplot as plt
import csv
from PIL import Image
# layout

st.set_page_config(
    page_title="The Model",
    page_icon="🦤",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""# Hungry Birds Prediction Tool
## What goes on under the hood?""")
image = Image.open("Images/workflow.JPG")
# Display image
st.image(image, use_column_width=True)
upload_file = st.file_uploader("Choose an audio file", type=[".wav"], accept_multiple_files=False)

#upload the song
if upload_file is not None:
    audio_bytes = upload_file.read()
    st.audio(audio_bytes,format="audio/wav")
    if st.button("The Yamnet Model"):
        st.markdown("""## Yamnet Model""")
        st.markdown("""### Yamnet model is the heart of Hungry Birds, it is a 3.7M params model optimized for audio classification""")
        st.markdown("""### It is used by Hungry Birds for feature extraction and preprocessing, but how does it work?""")
        #importing the model
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        def load_wav_16k_mono(filename):
            """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
            wav, sample_rate = librosa.load(filename, sr=16000, mono=True)
            wav = tf.convert_to_tensor(wav, dtype=tf.float32)
            return wav
        def run_yamnet(file):
            with open('temp_wav_file.wav', 'wb') as wav_file:
                wav_file.write(file)
            wav = load_wav_16k_mono('temp_wav_file.wav')
            # yamnet model
            scores, embeddings, spectrogram = yamnet_model(wav)
            return scores, wav, spectrogram

        # Find the name of the class with the top score when mean-aggregated across frames.
        def class_names_from_csv(class_map_csv_text):
            """Returns list of class names corresponding to score vector."""
            class_names = []
            with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    class_names.append(row['display_name'])

            return class_names

        my_model = tf.keras.models.load_model("yamnet_full", compile=False)

        def predict_score(file):
            with open('temp_wav_file.wav', 'wb') as wav_file:
                wav_file.write(file)
            wav = load_wav_16k_mono('temp_wav_file.wav')
            # yamnet model
            scores, embeddings, spectrogram = yamnet_model(wav)
            scores = scores[:,106]
            scores = tf.reshape(scores, [-1, 1])
            final_scores = scores * my_model(embeddings)
            final_score = tf.reduce_sum(final_scores, axis=0)
            row_sum = tf.reduce_sum(final_score)
            final_score = tf.divide(final_score, row_sum)
            final_score = pd.DataFrame(final_score, columns = ['Probability'])
            with open('yamnet_full/label_encoder.pkl', 'rb') as model_file:
                le = pickle.load(model_file)
            final_score.index = le.inverse_transform(final_score.index)
            final_score = final_score.sort_values(by = 'Probability', ascending = False).applymap(lambda x: "{:.2%}".format(x)).head(10)
            return final_scores.numpy(), final_score.index


        class_map_path = yamnet_model.class_map_path().numpy()
        class_names = class_names_from_csv(class_map_path)

        scores, wav, spectrogram = run_yamnet(audio_bytes)
        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()

        # Define the figure object.
        fig = plt.figure(figsize=(8, 10))

        # Plot the waveform.
        plt.subplot(4, 1, 1)
        plt.plot(wav)
        plt.xlim([0, len(wav)])
        plt.title('Input Wave')
        # Plot the log-mel spectrogram (returned by the model).
        plt.subplot(4, 1, 2)
        plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')
        plt.title('Mel Spectogram')

        # Plot and label the model output scores for the top-scoring classes.
        mean_scores = np.mean(scores, axis=0)
        top_n = 10
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
        plt.subplot(4, 1, 3)
        plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
        plt.title('Yamnet Model Results')

        # patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
        # values from the model documentation
        patch_padding = (0.025 / 2) / 0.01
        plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
        # Label the top_N classes.
        yticks = range(0, top_n, 1)
        plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
        _ = plt.ylim(-0.5 + np.array([top_n, 0]))

        final_scores_np, final_scores_index = predict_score(audio_bytes)

        plt.subplot(4, 1, 4)
        # Plot and label the model output scores for the top-scoring classes.
        mean_scores = np.mean(final_scores_np, axis=0)
        top_n = 10
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
        plt.imshow(final_scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')
        plt.title('Hungry Birds Model Results')

        #patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
        #values from the model documentation
        patch_padding = (0.025 / 2) / 0.01
        plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
        #Label the top_N classes.
        yticks = range(0, top_n, 1)
        plt.yticks(yticks, [final_scores_index[x] for x in yticks])
        _ = plt.ylim(-0.5 + np.array([top_n, 0]))
            # Display the figure using Streamlit.
        plt.subplots_adjust(hspace=0.5)
        st.pyplot(fig)

        st.markdown("""### The final model archieved a top-10-accuracy of **55%**, although it can definitely be improved it is already **6.7x** better than a dummy model (8.2%).""")
        st.markdown("""### Here are some potential solutions for model improvement:""")
        st.markdown("""#### - Refine the data cleanning and noise reduction""")
        st.markdown("""#### - Improve Dataset inbalance (same lenght across species)""")
        st.markdown("""#### - Use source separation during predict (BirdMixit)""")
        st.markdown("""#### - Unfreeze the layers from Yamnet to specialize it with our Dataset **(time consuming)**""")
