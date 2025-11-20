#imports
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#for confusion matrix
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

#Load fashion MINST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print("Original x_train shape:", x_train.shape)
print("Original x_test shape:", x_test.shape)

#pre-processing for MLP model
x_train_mlp = x_train.astype("float32") / 255.0
x_test_mlp = x_test.astype("float32") / 255.0

x_train_mlp = x_train_mlp.reshape(-1, 28 * 28)
x_test_mlp = x_test_mlp.reshape(-1, 28 * 28)

print("x_train_mlp shape (flattened):", x_train_mlp.shape)
print("x_test_mlp shape (flattened):", x_test_mlp.shape)

#building the model
mlp_model = keras.Sequential([
    layers.Input(shape=(28 * 28,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")   # 10 classes
])

#compiling
mlp_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)

mlp_model.summary()