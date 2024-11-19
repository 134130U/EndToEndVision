from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import PIL
from PIL import Image
from io import BytesIO
import tensorflow as tf
# from tensorflow import keras
# from keras.layers import TFSMLayer

app = FastAPI()

MODEL = tf.keras.models.load_model("../saved_models/version_1.keras")
CLASS_NAME = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']
@app.get('/')
async def ping():
    return "Hello it's Babou"

def read_file_as_image(data) -> np.array:
    image = np.array(Image.open(BytesIO(data)))
    return image
@app.post('/predict')
async def predict(
        file : UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAME[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


if __name__ == '__main__':
    uvicorn.run(app,host='localhost',port=9090)