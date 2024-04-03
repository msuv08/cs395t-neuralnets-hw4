# Import the necessary libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Softmax
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the MNIST data set of 28x28 images of digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the input data by reshaping and normalizing
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the second model with two fully connected layers
model_b = Sequential()

# Add a 2D convolutional layer with 20 channels (kernels) and a small 3x3 kernel as in the first model
model_b.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Add a max pooling layer as in the first model
model_b.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 3D output to 1D as in the first model
model_b.add(Flatten())

# Add the first fully connected layer with 200 gates and ReLU activation
model_b.add(Dense(200, activation='relu'))

# Add the second fully connected layer with 100 gates and ReLU activation
model_b.add(Dense(100, activation='relu'))

# Add the output layer with 10 neurons for each class and softmax activation as in the first model
model_b.add(Dense(10, activation='softmax'))

# Compile the model with the same loss function, optimizer, and metric as in the first model
model_b.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary to verify the architecture
model_b.summary()

# Train the model with the training data
model_b.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model with the test data to get the accuracy
score_b = model_b.evaluate(x_test, y_test, verbose=0)
print('Second Model Test loss:', score_b[0])
print('Second Model Test accuracy:', score_b[1])
