# Matplotlib
import matplotlib.pyplot as plt
# Tensorflow
import tensorflow as tf
# Numpy and Pandas
import numpy as np
import pandas as pd
# Ohter import
import sys


from sklearn.preprocessing import StandardScaler # z = (x-u)/s, s standard deviation

assert hasattr(tf, "function") # Be sure to use tensorflow 2.0

# Load the dataset
# MNIST
(images, targets), (_, _) = tf.keras.datasets.mnist.load_data(path="mnist.npz")


# Get only a subpart of the dataset
images = images[:10000]
targets = targets [:10000]

# flatten
images = images.reshape(-1, 784)
images = images.astype(float)

scaler = StandardScaler()

images = scaler.fit_transform(images) # Normalization

from sklearn.model_selection import train_test_split # Test Set

images_train, images_test, targets_train, targets_test = train_test_split(images, targets, test_size=0.2, random_state=1)

print(images_train.shape, targets_train.shape)
print(images_test.shape, targets_test.shape)

# Plot one of the data
targets_names = ["0", "1", "2", "3", "4", "5", 
                 "6", "7", "8", "9"
]
# Plot one image
plt.imshow(images[10].reshape(28, 28), cmap="binary")
plt.title(targets_names[targets[10]])
plt.show()

print("First line of one image", images[10])
print("Associated target", targets[10])

# Create the model
# Flatten
model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))

# Add the layers
model.add(tf.keras.layers.Dense(256, activation="sigmoid"))
model.add(tf.keras.layers.Dense(128, activation="sigmoid"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model_output = model.predict(images[0:1])
print(model_output, targets[0:1]) # Values are smaller as it is normalized, the model is uncertain on all of the classes as it should be

model.summary()

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy", # "sparse" keyword allows to represent labels more easily. We can replace "softmax" by "sigmoid"
    optimizer="sgd",
    metrics=["accuracy"]
)

# Train the model
history = model.fit(images_train, targets_train, epochs=10, validation_split=0.2) # Epoch is the number of iteration; validation_split enough to split the training set, for example if 0.2, 20% of the testing set will be validation set (the training set will then be only 80% of its original value)

# Training Set
loss_curve = history.history["loss"]
acc_curve = history.history["accuracy"]

# Validation Set
loss_val_curve = history.history["val_loss"]
acc_val_curve = history.history["val_accuracy"]

plt.plot(loss_curve, label="Train")
plt.plot(loss_val_curve, label="Val")
plt.legend(loc='upper left')
plt.title("Loss")
plt.show()

plt.plot(acc_curve, label="Train")
plt.plot(acc_val_curve, label="Val")
plt.legend(loc='upper left')
plt.title("Accuracy")
plt.show()