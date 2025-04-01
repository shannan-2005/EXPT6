import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Define model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Save model
model.save("mnist_model.h5")

# Load the saved model
loaded_model = keras.models.load_model("mnist_model.h5")

# Make predictions
predictions = loaded_model.predict(x_test)

# Convert predicted probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Compute MAE and MSE
mae = mean_absolute_error(y_test, predicted_labels)
mse = mean_squared_error(y_test, predicted_labels)

# Select a random test image
index = np.random.randint(0, len(x_test))
plt.imshow(x_test[index], cmap='gray')
plt.title(f"Predicted: {predicted_labels[index]}, Actual: {y_test[index]}")
plt.show()

# Write results to a text file
with open("mnist_results.txt", "w") as file:
    file.write(f"Predicted label for sample index {index}: {predicted_labels[index]}\n")
    file.write(f"Actual label: {y_test[index]}\n")
    file.write(f"Mean Absolute Error (MAE): {mae}\n")
    file.write(f"Mean Squared Error (MSE): {mse}\n")

print("Results saved to mnist_results.txt")
