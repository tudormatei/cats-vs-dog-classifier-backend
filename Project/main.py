from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import tensorflow as tf

app = Flask(__name__)


def predict_image(model, img):
    image_shape = (200, 200, 3)

    my_image = image.load_img(img, target_size=image_shape)
    my_image = image.img_to_array(my_image)

    my_image = my_image / 255
    my_image = np.expand_dims(my_image, axis=0)

    return model.predict(my_image)[0][0]


@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'


model = load_model('cats_vs_dogs_model.h5')


@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files['image']

    img = Image.open(file.stream)

    img.save("image_to_predict.jpg")

    results = predict_image(model=model, img='image_to_predict.jpg')

    classes = ['cat', 'dog']

    return jsonify(classes[int(results.round())])


if __name__ == '__main__':
    app.run(debug=True)