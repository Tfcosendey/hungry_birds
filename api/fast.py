from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
import io

app = FastAPI()

#@app.post("/predict/")
#async def predict_audio(file: UploadFile = File(...)):
    # Process the audio file
    # ...
    # Return the predicted class label
    
    import tensorflow as tf

    model=tf.saved_model.load(
        "yamnet_bird_1", tags=None, options=None
    )

    #preprocess

    #file here
    model.predict()
    #return {"class": "music"}


@app.get("/")
def index():
    
    return {"status": "ok"}
