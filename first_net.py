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

# Define the model
model = Sequential()

# Add a 2D convolutional layer with 20 channels (kernels) and a small 3x3 kernel
model.add(Conv2D(20, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

# Add a max pooling layer to reduce the spatial dimensions of the output volume
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the 3D output to 1D for the fully connected layer
model.add(Flatten())

# Add a fully connected layer with 100 gates (neurons) and ReLU activation
model.add(Dense(100, activation='relu'))

# Add the output layer with 10 neurons for each class and softmax activation
model.add(Dense(10, activation='softmax'))

# Compile the model with a suitable loss function, optimizer, and metric for accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary to verify the architecture
model.summary()

# Train the model with the training data
model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model with the test data to get the accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('First Model Test loss:', score[0])
print('First Model Test accuracy:', score[1])
