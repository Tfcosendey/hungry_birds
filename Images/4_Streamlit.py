
import streamlit as st



#title = '<p style="font-family:Courier;  font-size: 40px;">Streamlit</p>'
#st.markdown(title, unsafe_allow_html=True)
st.title("Streamlit")

st.text('''

Streamlit is an open source python based framework for developing and deploying
interactive data science dashboards and machine learning models.
It is compatible with major Python libraries such as scikit-learn, Keras, PyTorch,
SymPy(latex),NumPy, pandas, Matplotlib etc.

So you do not have to rely on a team of front end developers or learn web design
languages. Because of that, there's huge excitement about it in the Data Science
community and there are data showing the extremely rapid adoption rate of
it compared to other similar tools.

''')


st.title("Simple and Agil")

st.text("Streamlit turns data scripts into shareable web apps in minutes, just using python.")
st.text("And it's also open source and very well documenteted.")



st.title("Embrace Scripting")
st.markdown('''

Build an app in a few lines of python code.

Just need to dowlond it:
    `pip install streamlit`

And to run it, symple type:
    `streamlit run name_file.py`

Making changes in the code:
    You can see it automatically update as you iteratively save the source file.

''')



st.title("Weave in interaction")
st.text('''

Adding a widget is the same as declaring a variable,
so the app can be very personalized by displaying the text in many formats:
    title, header, subheader, markdown, code and latex
and by making it interactive using buttons, sliders, chekbox

''')


st.title("Deploy instantly")

st.text('''

You can share, manage and deploy the app directly from Stremlit and without cost.
Streamlit watches for changes on updates of the linked Git repository and the
application will be deployed automatically in the shared link.

''')


st.title("Hungry Bird")

st.markdown('''

The application we built using Streamlit was written in a python file, test.py, and is organized in this way:


- Layout of the page:
Using `st.set_page_config()`

- Upload the file:
Ask the user to upldoad the audio file using `st.file_uploader()`
and transform it to .wav

- Import and load the model:
the yamnet model using `hub.load()` and then `tf.keras.models.load_model()`

- Preprocessing:
Load a .wav file, convert it to a float tensor, resample to 16 kHz single-channel audio

- Embedding:
Represent discrete variables as continuous vectors

- Predict:
Show the probability of the ten bird species with the highest likelihood
of beeing their songs the one of the audio with their respective images
''')
