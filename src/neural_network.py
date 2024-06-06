import random
import numpy as np

class Network:
    def __init__(self,sizes):
        np.random.seed(42)
        random.seed(42)
        self.num_of_layers = len(sizes)
        self.sizes = sizes
        self.bias = [np.random.randn(y,1) for y in sizes[1:]] 
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[:-1],sizes[1:])]
        for i in range(len(self.bias)):
            print(np.shape(self.bias[i]))
            print(np.shape(self.weights[i]))

    def sigmoid(self,z):
         return 1.0/(1.0+np.exp(-z))

    def feedforward(self,a):
        # print(np.shape(a))
        for b, w in zip(self.bias, self.weights):
            a = self.sigmoid(np.dot(a,w) + np.transpose(b))
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,lmbda = 0,test_data=None):
        # if test_data: n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, lmbda)
            print("Epoch ", i+1 ,"completed!!" )
        

        if test_data:
            n_test = self.evaluate(test_data)/len(test_data) * 100
            n_training = self.evaluate(training_data)/len(training_data) * 100
            print("Training Accuracy is: ", n_training)
            print("Test Accuracy is: ", n_test)
            # print ("Epoch {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test))
        else:
            print ("Epoch {0} complete".format(i))
    
    def update_mini_batch(self, training_set, learning_rate,lmbda):
        delta_b, delta_w = self.backprop(training_set)
        for i in range(len(self.bias)):
            self.weights[i] = np.subtract((1-learning_rate*(lmbda/1000))*self.weights[i],(learning_rate/10)*delta_w[i])
            self.bias[i] = np.subtract(self.bias[i],(learning_rate/10)*delta_b[i].reshape(-1,1))
                

    def backprop(self,training_set):
        x, y = zip(*training_set)
        
        x = list(x)
        y = list(y)

        nb = [np.zeros(b.shape) for b in self.bias]
        nw = [np.zeros(w.shape) for w in self.weights]

        activation_layer = x
        activations = [activation_layer]
        zs=[]

        for b, w in zip(self.bias, self.weights):
            z= np.dot(activation_layer,w) + b.transpose()
            zs.append(z)
            activation_layer = self.sigmoid(z)
            activations.append(activation_layer)

        delta = self.cost_derivative(activations[-1], y)
        b = np.array(delta.sum(axis=0))
        nb[-1] = b.reshape((10,1))
        nw[-1] = np.dot(np.transpose(activations[-2]), delta)

        for l in range(2, self.num_of_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.multiply(np.dot(delta,self.weights[-l+1].transpose()), sp)
            nb[-l] = delta.sum(axis=0)
            nw[-l] = np.dot(np.transpose(activations[-l-1]), delta)

        return (nb, nw)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(np.array_equal(x, y)) for (x, y) in test_results)
    


    def sigmoid_prime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def cost_derivative(self, output_activations, y):
        return (output_activations-y) 
        
        