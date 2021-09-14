# from network import Network
from simple_nn import Network
import idx2numpy 
import numpy as np

def oneHot(y):
    y_hat = np.zeros((10, 1))
    y_hat[y] = 1
    return y_hat

images = [image.flatten().reshape([784, 1]) for image in idx2numpy.convert_from_file("./data/train-images")]
labels = idx2numpy.convert_from_file("./data/train-labels")
training_data = [(image * (1.0 / 256.0), oneHot(label)) for image, label in zip(images, labels)]

net = Network([784, 30, 10])

print("Before training acc is {0:2}%".format(net.computeAccuracy(training_data)))

net.train(training_data)
 
print("After training is {0:.2}%".format(net.computeAccuracy(training_data)))

images = [image.flatten().reshape([784, 1]) for image in idx2numpy.convert_from_file("./data/test-images")]
labels = idx2numpy.convert_from_file("./data/test-labels")
test_data = [(image * (1.0 / 256.0), oneHot(label)) for image, label in zip(images, labels)]

print("Accuracy in the test set is {0}%".format(net.computeAccuracy(test_data)))

