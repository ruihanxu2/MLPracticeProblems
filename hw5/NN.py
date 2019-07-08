import scipy.io
import numpy as np
from sklearn.utils import shuffle
import random

import matplotlib
import matplotlib.pyplot as plt

"""
The code below referenced Michael Nielsen's code for his book 'Neural Networks
and Deep Learning' (found at https://github.com/mnielsen/neural-networks-and-deep-learning).

Also with a reference from elijahanderson
(found at https://github.com/elijahanderson/From-Scratch-MNIST/blob/master/network.py)

"""



def sigmoid(z):
    s = 1.0 / (1 + np.exp(-z))
    return s


def derivitive_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # print('The shape of weights is ',np.array(self.weights).shape)
        # print('The shape of biaes is ', np.array(self.biases).shape)
        # print(self.weights[1][0])
        # # print('shape is ', np.array(self.weights[0][0]).shape())
        # print(self.biases)
        """
            list of matrices
            weights[layer][left layer neuron][right layer neuron]
            biases[layer][right layer neuron][0]
            
        """
    def forward(self, n):
        #get the out put of the nn

        for bias,weight in zip(self.biases, self.weights):

            n = sigmoid(np.dot(weight, n)+bias)

        return n

    def cost_derivative(self, output, y, n):
        return (output - y)/(output*(1-output))
        #return (output - y)

    def backpropagate(self, x, y, n):
        # updated weights and biases

        new_weights = [np.zeros(w.shape) for w in self.weights]
        new_biases = [np.zeros(b.shape) for b in self.biases]

        #################forward

        activation = x
        activations = [x]

        z_vectors = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            z_vectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #################backword

        #right most
        delta = self.cost_derivative(activations[-1], y, n) * derivitive_sigmoid(z_vectors[-1])

        #partial derivative of the cost function with regard to W and b
        new_biases[-1] = delta
        new_weights[-1] = np.dot(delta, activations[-2].transpose())

        #hidden layers
        for l in range(2, self.num_layers):
            z = z_vectors[-l]
            sp = derivitive_sigmoid(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta)*sp
            new_biases[-l] = delta
            new_weights[-l] = np.dot(delta, activations[-l-1].transpose())

        #print("dW1 is: ", np.linalg.norm(new_weights[0]))
        return (new_biases, new_weights, np.linalg.norm(new_weights[0])*np.linalg.norm(new_weights[0]))


    def update_mini_batch(self, mini_batch, learning_rate):
        # updated weights and biases

        new_weights = [np.zeros(w.shape) for w in self.weights]
        new_biases = [np.zeros(b.shape) for b in self.biases]

        # cycle through each one
        norms = []
        for x, y in mini_batch:
            delta_new_biases, delta_new_weights , norm= self.backpropagate(x, y, len(mini_batch))
            norms.append(norm)
            # update
            new_biases = [nb + dnb for nb, dnb in zip(new_biases, delta_new_biases)]
            new_weights = [nw + dnw for nw, dnw in zip(new_weights, delta_new_weights)]



        # gradient descent
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, new_weights)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, new_biases)]
        return sum(norms)


    def stochastic(self, training_data, epochs, mini_batch_size, learning_rate, test_data):
        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
            print('Testing set, Number correct before network is implemented: {} / {}'.format(self.evaluate(test_data), n_test))

        print('Training set, Number correct before network is implemented: {} / {}'.format(self.evaluate(training_data), n))
        for j in range(epochs):
            print("epoches", j)
            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            normss = []
            for mini_batch in mini_batches:
                norms = self.update_mini_batch(mini_batch, learning_rate)
                normss.append(norms)


            if test_data:
                num_correct = self.evaluate(test_data)
                print("Testing set : {} / {}".format(num_correct, n_test))
                print("Testing set accuracy :", num_correct / n_test)
            else:
                print("Epoch {} complete".format(j))

            num_correct_training = self.evaluate(training_data)
            print("Training set : {} / {}".format( num_correct_training, n))
            print("Training set accuracy :", num_correct_training/n)

            print("The average gradient is: ",sum(normss)/12665)




    def evaluate(self, test_data):
        results = [(self.forward(x), y) for (x, y) in test_data]

        sum_ = 0
        for res in results:
            # print('res is ', res)
            if (res[0][0]<=0.5 and res[1][0] == 0):
                sum_ += 1
            elif res[0][0]>0.5 and res[1][0] == 1:
                sum_ += 1

        return sum_









mnist = scipy.io.loadmat('mnist_all.mat')

X_train0 = mnist['train0']

# X_train0 = [np.reshape(x, (784, 1)) for x in X_train0]
# X_train0 = np.array(X_train0)


# print(X_train0.shape)
y_train0 = np.zeros(X_train0.shape[0])



X_test0 = mnist['test0']
y_test0 = np.zeros(X_test0.shape[0])

X_train1 = mnist['train1']
y_train1 = np.ones(X_train1.shape[0])

X_test1 = mnist['test1']
y_test1 = np.ones(X_test1.shape[0])

X_train = np.concatenate((X_train0, X_train1))
y_train = np.concatenate((y_train0, y_train1))

X_test = np.concatenate((X_test0, X_test1))
y_test = np.concatenate((y_test0, y_test1))

X_train, y_train = shuffle(X_train, y_train, random_state=0)
X_test, y_test = shuffle(X_test, y_test, random_state=0)

X_train = np.true_divide(X_train, 255)

X_test = np.true_divide(X_test, 255)

X_train = [np.reshape(x, (784, 1)) for x in X_train]
X_train = np.array(X_train)
# print(X_train.shape)

X_test = [np.reshape(x, (784, 1)) for x in X_test]
X_test = np.array(X_test)
# print(X_test.shape)


y_train = np.reshape(y_train, (-1,1))
y_test = np.reshape(y_test, (-1,1))

# print(X_train.shape) # (12665, 784)
# print(y_train.shape) # (12665,)


train_data = [(x, y) for x, y in zip(X_train[:], y_train[:])]
test_data = [(x, y) for x, y in zip(X_test[:], y_test[:])]




"""train _data is a list of 12665 tuples(x, y), where x is an (784,1) shaped ndarray and y is an (1,)"""


net = Network([784, 16, 16, 16, 16, 1])
print("Network config: [784, 16, 16, 16, 16, 1]" )
net.stochastic(train_data, 1, 100, 1, test_data)

#stochastic(self, training_data, epochs, mini_batch_size, learning_rate, test_data):


