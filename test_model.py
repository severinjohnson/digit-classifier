import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Load the model
model = load_model('sev_mnist_model.h5')

# Now you can use the model to make predictions or evaluate it


def predict_and_show_mnist_digit(index, model, test_images, test_labels):
    # Select the image and label from the test set
    img = test_images[index]
    true_label = test_labels[index]

    # Normalize the image
    img_normalized = img / 255

    # Reshape the image to add a batch dimension
    img_batch = img_normalized.reshape((1, 28, 28))

    # Make a prediction
    predictions = model.predict(img_batch)

    # Convert logits to probabilities
    probabilities = tf.nn.softmax(predictions[0])

    # Get the predicted class
    predicted_class = np.argmax(probabilities)

    # Plot the image
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f"Predicted: {predicted_class}, True Label: {true_label}")
    plt.colorbar()
    plt.grid(False)
    plt.show()

    return predicted_class, true_label

index = 9 # Replace with the index of the image you want to test
for i in range(15):
    predict_and_show_mnist_digit(i, model, test_images, test_labels)