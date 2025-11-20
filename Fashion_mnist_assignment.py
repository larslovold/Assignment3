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

#building the model - can change hyperparamaters for questiuon 3
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

#training the model - for question 1
BATCH_SIZE = 64

history_mlp = mlp_model.fit(
    x_train_mlp,
    y_train,
    validation_split=0.2,   #20% of training data used for validation
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2
)

#plot training and validation - for question 1
def plot_history(history, title_prefix="MLP"):
    #Loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} - Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_history(history_mlp, title_prefix="MLP")

#Print final training and validation metrics (last epoch)
final_train_loss = history_mlp.history["loss"][-1]
final_train_acc = history_mlp.history["accuracy"][-1]
final_val_loss = history_mlp.history["val_loss"][-1]
final_val_acc = history_mlp.history["val_accuracy"][-1]

print("\n=== Final MLP Training/Validation Metrics (last epoch) ===")
print(f"Train loss:      {final_train_loss:.4f}")
print(f"Train accuracy:  {final_train_acc:.4f}")
print(f"Val loss:        {final_val_loss:.4f}")
print(f"Val accuracy:    {final_val_acc:.4f}")

#evalute on test set
test_loss_mlp, test_acc_mlp = mlp_model.evaluate(x_test_mlp, y_test, verbose=0)
print("\n=== MLP Test Metrics ===")
print(f"Test loss:       {test_loss_mlp:.4f}")
print(f"Test accuracy:   {test_acc_mlp:.4f}")

#confusion matrix for MLP 
#Predict class probabilities for test data
y_pred_probs_mlp = mlp_model.predict(x_test_mlp)
#Convert probs to class indices
y_pred_mlp = np.argmax(y_pred_probs_mlp, axis=1)

#Compute confusion matrix
cm_mlp = confusion_matrix(y_test, y_pred_mlp)

#Plot confusion matrix
plt.figure(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp,
                              display_labels=class_names)
disp.plot(include_values=True,
          xticks_rotation=45,
          ax=plt.gca(),
          colorbar=False)
plt.title("Confusion Matrix - MLP on Fashion-MNIST")
plt.tight_layout()
plt.show()

print("\nConfusion matrix shape:", cm_mlp.shape)

