import numpy as np
from keras.datasets import mnist

# Adjusted vectorized_result function
def vectorized_result(y):
    e = np.zeros(10)
    e[y] = 1.0
    return e

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
# Change the shape to (784,) for each image
training_inputs = [x.reshape(784) for x in train_images]
training_results = [vectorized_result(y) for y in train_labels]

# Combine them into training_data using zip
training_data = zip(training_inputs, training_results)
training_data = list(training_data)

test_inputs = [x.reshape(784) for x in test_images]
test_results = [vectorized_result(y) for y in test_labels]

# Combine them into training_data using zip
test_data = zip(test_inputs, test_results)
test_data = list(test_data)

import neural_network
net = neural_network.Network([784,256, 10])

net.SGD(training_data, 120, 10, 0.01, test_data=test_data)