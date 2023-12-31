from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load the models
model1 = tf.keras.models.load_model('sev_mnist_model.h5')
model2 = tf.keras.models.load_model('sev_mnist_model2.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit_drawing', methods=['POST'])
def handle_drawing():
    data = request.json
    image_data = data['image']
    selected_model = data.get('model', 'model1')  # Default to 'model1'

    # Decode the image
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))

    # Convert to grayscale and resize
    img = img.convert('L').resize((28, 28), Image.LANCZOS)

    # Normalize the image
    img_array = np.array(img) / 255.0
    img_array = 1.0 - img_array
    img_batch = img_array.reshape((1, 28, 28))

    # Make a prediction with the selected model
    model = model1 if selected_model == 'model1' else model2
    predictions = model.predict(img_batch)
    predicted_class = np.argmax(predictions[0])

    return jsonify({'prediction': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
