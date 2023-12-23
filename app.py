from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import base64
import io
from io import BytesIO

app = Flask(__name__)

# Load the model
model = load_model('sev_mnist_model.h5')
@app.route('/', methods=['GET'])
def index():
    # Render the index.html page
    return render_template('index.html')

@app.route('/submit_drawing', methods=['POST'])
def handle_drawing():
    # Receive the image as base64
    data = request.json
    image_data = data['image']

    # Decode the image
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Convert to grayscale and resize
    img = img.convert('L').resize((28, 28), Image.LANCZOS)

    # Normalize the image
    img_array = np.array(img) / 255.0
    img_array = 1.0 - img_array
    plt.imshow(img_array, cmap=plt.cm.binary)
    plt.show()
    img_batch = img_array.reshape((1, 28, 28))

    # Make a prediction
    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions[0])

    # Return the prediction
    return jsonify({'prediction': int(predicted_class)})


if __name__ == '__main__':
    app.run(debug=True)
