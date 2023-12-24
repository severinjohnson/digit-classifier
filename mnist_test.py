import numpy as np
import requests
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageOps
import base64
from io import BytesIO

# Load MNIST data
(train_images, train_labels), _ = mnist.load_data()

# Select an image
index = 9  # Change index to test different images
img = train_images[index]
true_label = train_labels[index]

# Convert to PIL Image
img_pil = Image.fromarray(img)

# Resize to match the input expected by the Flask app
img_resized = img_pil.resize((28, 28), Image.LANCZOS)
img_resized = ImageOps.invert(img_resized)
# Convert to base64
buffer = BytesIO()
img_resized.save(buffer, format="PNG")
img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

# URL of your Flask app
url = 'http://127.0.0.1:5000/submit_drawing'

# Send POST request
response = requests.post(url, json={'image': img_str})

# Print the response
print('True Label:', true_label)
print('Response:', response.json())
