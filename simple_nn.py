import numpy as np
import pickle as pkl
import requests
import idx2numpy
import random

def downloadData():
    url = "https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true"
    filename = url.split("/")[-1]
    with open("mnist.pkl.gz", "wb") as f:
        r = requests.get(url)
        f.write(r.content)

def same(a, b):
    return np.array_equal(a, b)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoidPrime(z):
        return sigmoid(z) * (1 - sigmoid(z))

def sigmoid_prime(z):
        return sigmoid(z) * (1 - sigmoid(z))

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.nLayers = len(self.layers)
        self.weights = [np.random.randn(layers[i + 1], layers[i]) for i in range(len(layers) - 1)] 
        self.biases = [np.random.randn(layers[i + 1], 1) for i in range(len(layers) - 1)]

        """ Hyperparameters """
        self.numberEpochs = 30 
        self.batchSize = 10 
        self.learningRate = 3.0 

    def compute(self, a):
        for i in range(self.nLayers - 1):
            a = sigmoid(np.add(self.biases[i], self.weights[i].dot(a)))
        return a

    def train(self, data):
        for epoch in range(self.numberEpochs):
            # print("At {0:0.2f}%".format(float(epoch * 100) / float(self.numberEpochs))) 
            # self.computeAccuracy(data)
            random.shuffle(data)
            batches = [data[i:i+self.batchSize] for i in range(0, len(data), self.batchSize)]
            for batch in batches:
                self.runBatch(batch)

    def runBatch(self, batch):
        nabla_w = [np.zeros((self.layers[i + 1], self.layers[i])) for i in range(len(self.layers) - 1)] 
        nabla_b = [np.zeros((self.layers[i + 1], 1)) for i in range(len(self.layers) - 1)] 

        eta_hat = self.learningRate / len(batch)

        for image, label in batch:
            x = image
            y = label 

            # returns the derivative of C with respect to w and b
            # delta_nabla_w, delta_nabla_b = self.backprop(x, y) 
            delta_nabla_w, delta_nabla_b = self.backprop(x, y) 
            # assert same(delta_nabla_w, mdelta_nabla_w)
            # assert same(delta_nabla_b, mdelta_nabla_b)

            nabla_w = [np.add(nabla_w[i], -eta_hat * delta_nabla_w[i]) for i in range(len(self.layers) - 1)]
            nabla_b = [np.add(nabla_b[i], -eta_hat * delta_nabla_b[i]) for i in range(len(self.layers) - 1)]

        self.weights = [np.add(self.weights[i], nabla_w[i]) for i in range(len(self.layers) - 1)]
        self.biases = [np.add(self.biases[i], nabla_b[i]) for i in range(len(self.layers) - 1)]

    def mbackprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = []           # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = np.subtract(activations[-1], y) * sigmoidPrime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, 3):
            z = zs[-l]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            # print("M First delta", delta) 

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

            # print("M First nabla w", nabla_w[-l])

        return (nabla_w, nabla_b)

    def backprop(self, x, y):
        delta_nabla_w = [np.zeros((self.layers[i + 1], self.layers[i])) for i in range(len(self.layers) - 1)] 
        delta_nabla_b = [np.zeros((self.layers[i + 1], 1)) for i in range(len(self.layers) - 1)] 

        z1 = np.add(self.biases[0], self.weights[0].dot(x))
        activation1 = sigmoid(z1)
        z2 = np.add(self.biases[1], self.weights[1].dot(activation1))
        activation2 = sigmoid(z2)

        outputDer = np.subtract(activation2, y) * sigmoidPrime(z2)
        
        delta_nabla_b[-1] = outputDer.copy()
        delta_nabla_w[-1] = np.dot(outputDer, np.transpose(activation1))

        # print("Weights -1 are", self.weights[-1])
        # print("Output der -1 is", outputDer)
        # print("Output of dot is", np.dot(np.transpose(self.weights[-1]), outputDer)) 

        prevOutputDer = outputDer.copy()

        outputDer = np.dot(np.transpose(self.weights[-1]), outputDer) * sigmoidPrime(z1)

        # print("First delta is", outputDer)
        
        # print("z is ", np.add(self.biases[0], self.weights[0].dot(x)))

        # print("Output of dot is", np.dot(np.transpose(self.weights[-1]), prevOutputDer)) 
        # print("OutputDer is", outputDer)

        # print("Fist output der is", outputDer)

        delta_nabla_b[-2] = outputDer 
        delta_nabla_w[-2] = np.dot(outputDer, np.transpose(x)) 

        # print("First nabla w is", delta_nabla_w[-2])

        return delta_nabla_w, delta_nabla_b

    def computeAccuracy(self, data):
        correct = 0
        for image, label in data:
            if np.argmax(self.compute(image)) == np.argmax(label):
                correct += 1
        return float(100 * correct) / len(data)


if __name__ == "__main__":
    net = Network([28 * 28, 30, 10])

