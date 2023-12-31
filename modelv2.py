import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the data to include a channel dimension
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Data augmentation with zoom range for both zooming in and out
datagen = ImageDataGenerator(
    rotation_range=10,  # Rotate images by up to 10 degrees
    width_shift_range=0.1,  # Shift images horizontally by up to 10%
    height_shift_range=0.1,  # Shift images vertically by up to 10%
    zoom_range=[0.9, 1.1],  # Zoom in or out by up to 10%
    fill_mode='constant',  # Fill in new pixels
    cval=0  # Value for the constant fill mode
)

# Define the model with dropout
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Add dropout layer
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fit the model using the augmented data generator
model.fit(datagen.flow(train_images, train_labels, batch_size=32),
          epochs=15,
          validation_data=(test_images, test_labels))

# After training the model
model.save('sev_mnist_model2.h5')  # Save the model as an HDF5 file
