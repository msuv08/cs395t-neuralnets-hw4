# CNN's with MNIST (HW4)

This README details the neural network architecture and training process for a simple convolutional neural network (CNN).

## 4(a) CNN-1 Architecture Summary

The neural network consists of the following layers:

- **Conv2D:** A convolutional layer with 20 channels of size 3x3, which processes the input images.
- **MaxPooling2D:** A max-pooling layer with a 2x2 window to reduce the spatial dimensions of the convolutional layer output.
- **Flatten:** A layer that converts the 2D feature maps from the convolutional layers into a 1D feature vector.
- **Dense:** A fully connected layer with 100 gates using the ReLU activation function.
- **Dense:** The output layer with 10 neurons, one for each digit (0-9), using the softmax activation function for multi-class classification.

It is represented visually in `tensorflow` as the following:
````python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 20)        200       
                                                                 
 max_pooling2d (MaxPooling2D) (None, 13, 13, 20)       0                                                                 
                                                                 
 flatten (Flatten)           (None, 3380)              0         
                                                                 
 dense (Dense)               (None, 100)               338100    
                                                                 
 dense_1 (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 339,310
Trainable params: 339,310
Non-trainable params: 0
_________________________________________________________________
````

## Training Process for CNN-1

This model is trained over just 10 epochs with the following configurations:
- **Loss function**: Categorical crossentropy, standard for multi-class classification tasks.
- **Optimizer**: Adam, a common choice for deep learning models.
- **Metric**: Accuracy, to measure correctly predicted instances.

## Performance Evaluation for CNN-1

After the training, the first model showed the following performance:
- Training accuracy: Approximately 99.76%
- Test accuracy: Approximately 98.70%
- Test loss: 0.0476

The comparable accuracy between training and testing suggests good generalization with minimal overfitting. Here's a look at the training:
````python
Epoch 1/10
469/469 [==============================] - 4s 7ms/step - loss: 0.2583 - accuracy: 0.9296 - val_loss: 0.0912 - val_accuracy: 0.9716
Epoch 2/10
469/469 [==============================] - 3s 7ms/step - loss: 0.0815 - accuracy: 0.9763 - val_loss: 0.0605 - val_accuracy: 0.9799
Epoch 3/10
469/469 [==============================] - 3s 7ms/step - loss: 0.0553 - accuracy: 0.9839 - val_loss: 0.0542 - val_accuracy: 0.9830
Epoch 4/10
469/469 [==============================] - 3s 7ms/step - loss: 0.0431 - accuracy: 0.9873 - val_loss: 0.0479 - val_accuracy: 0.9849
Epoch 5/10
469/469 [==============================] - 3s 7ms/step - loss: 0.0338 - accuracy: 0.9900 - val_loss: 0.0444 - val_accuracy: 0.9849
Epoch 6/10
469/469 [==============================] - 3s 7ms/step - loss: 0.0259 - accuracy: 0.9925 - val_loss: 0.0428 - val_accuracy: 0.9862
Epoch 7/10
469/469 [==============================] - 3s 7ms/step - loss: 0.0203 - accuracy: 0.9941 - val_loss: 0.0435 - val_accuracy: 0.9852
Epoch 8/10
469/469 [==============================] - 3s 7ms/step - loss: 0.0175 - accuracy: 0.9944 - val_loss: 0.0446 - val_accuracy: 0.9858
Epoch 9/10
469/469 [==============================] - 3s 7ms/step - loss: 0.0134 - accuracy: 0.9961 - val_loss: 0.0468 - val_accuracy: 0.9865
Epoch 10/10
469/469 [==============================] - 3s 7ms/step - loss: 0.0091 - accuracy: 0.9976 - val_loss: 0.0476 - val_accuracy: 0.9870
````

## 4(b) CNN-2 Architecture Summary

The second neural network has an additional fully connected layer and consists of the following layers:

- **Conv2D:** Same as the first model.
- **MaxPooling2D:** Same as the first model.
- **Flatten:** Same as the first model.
- **Dense:** A first fully connected layer with 200 gates using the ReLU activation function.
- **Dense:** A second fully connected layer with 100 gates using the ReLU activation function.
- **Dense:** The output layer with 10 neurons, one for each digit (0-9), using the softmax activation function.

Again, it is represented in `tensorflow` visually as the following:
````python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 20)        200       
                                                                 
 max_pooling2d (MaxPooling2D)  (None, 13, 13, 20)       0                                                                     
                                                                 
 flatten (Flatten)           (None, 3380)              0         
                                                                 
 dense (Dense)               (None, 200)               676200    
                                                                 
 dense_1 (Dense)             (None, 100)               20100     
                                                                 
 dense_2 (Dense)             (None, 10)                1010      
                                                                 
=================================================================
Total params: 697,510
Trainable params: 697,510
Non-trainable params: 0
_________________________________________________________________
````

## Training Process for CNN-2

Just like before, this second model is trained over just 10 epochs with the following configurations:
- **Loss function**: Categorical crossentropy
- **Optimizer**: Adam
- **Metric**: Accuracy

## Performance Evaluation for CNN-2

After the training, the second net showed the following performance:
- Training accuracy: Approximately 99.87%
- Test accuracy: Approximately 98.33%
- Test loss: 0.0617

Again, this is indicative of good generalization with minimal overfitting. Here's a look at the training:
````python
Epoch 1/10
469/469 [==============================] - 5s 9ms/step - loss: 0.2186 - accuracy: 0.9355 - val_loss: 0.0773 - val_accuracy: 0.9762
Epoch 2/10
469/469 [==============================] - 4s 9ms/step - loss: 0.0642 - accuracy: 0.9809 - val_loss: 0.0567 - val_accuracy: 0.9822
Epoch 3/10
469/469 [==============================] - 4s 9ms/step - loss: 0.0403 - accuracy: 0.9878 - val_loss: 0.0503 - val_accuracy: 0.9846
Epoch 4/10
469/469 [==============================] - 4s 9ms/step - loss: 0.0294 - accuracy: 0.9908 - val_loss: 0.0565 - val_accuracy: 0.9822
Epoch 5/10
469/469 [==============================] - 4s 9ms/step - loss: 0.0211 - accuracy: 0.9931 - val_loss: 0.0405 - val_accuracy: 0.9878
Epoch 6/10
469/469 [==============================] - 4s 9ms/step - loss: 0.0147 - accuracy: 0.9954 - val_loss: 0.0447 - val_accuracy: 0.9853
Epoch 7/10
469/469 [==============================] - 4s 9ms/step - loss: 0.0119 - accuracy: 0.9962 - val_loss: 0.0506 - val_accuracy: 0.9861
Epoch 8/10
469/469 [==============================] - 4s 9ms/step - loss: 0.0094 - accuracy: 0.9970 - val_loss: 0.0490 - val_accuracy: 0.9859
Epoch 9/10
469/469 [==============================] - 4s 9ms/step - loss: 0.0070 - accuracy: 0.9977 - val_loss: 0.0490 - val_accuracy: 0.9862
Epoch 10/10
469/469 [==============================] - 4s 9ms/step - loss: 0.0044 - accuracy: 0.9987 - val_loss: 0.0617 - val_accuracy: 0.9833
````

# Accuracy Comparison

The first model had a slightly higher accuracy on the test set compared to the second model. While the results for accuracy were close, I still wanted to analyze the difference between them. Despite the second neural net being more complex (should learn more), it had a slightly (marginally) lower accuracy of 98.33% versus the first net's 98.70%. Having more complexity does not always result in better generalization, and there may even have been some overfitting with the training dataset.

This was a cool learning experience as well, good to work with Tensorflow! Had some initial problems getting it up and running on Mac's silicon M1, recommend `conda` to help resolve those dependency issues (if running on ARM chips).