__author__ = 'gabriel'
import numpy as np
from sklearn import datasets

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_prime(x):
    sigm = sigmoid(x)
    return sigm * (1.0 - sigm)


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)
        print()

    def fit(self, X, y, learning_rate=0.2, epochs=70000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        for k in range(epochs):
            if k % 10000 == 0:
                print('epochs:', k)
            i = np.random.randint(X.shape[0])
            activations = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(activations[l], self.weights[l])
                    activation = self.activation(dot_value)
                    activations.append(activation)
            # output layer
            error = y[i] - activations[-1]
            deltas = [error * self.activation_prime(activations[-1])]
            # we need to begin at the second to last layer
            # (activations layer before the output layer)
            for l in range(len(activations) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(activations[l]))
            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # back-propagation
            # 1. Multiply its output delta and input activation
            #    to get the gradient of the weight.
            # 2. Subtract activations ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(activations[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        result = self.predict_proba(x)
        return abs(result.round())

    def predict_proba(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':
    data = datasets.load_digits()
    nn = NeuralNetwork([64, 15, 10, 10])
    data = list(zip(data.data, data.target))
    np.random.shuffle(data)
    x, y = zip(*data)
    x_train, x_test = np.array(x[:-100]), np.array(x[-100:])
    y_train, y_test = np.array(y[:-100]), np.array(y[-100:])

    nn.fit(x_train, y_train)
    pr = []
    for i, e in enumerate(x_test):
        predict = nn.predict(e)
        pr.append(predict[0])
    total = sum([1 for i in range(len(pr)) if pr[i] != y_test[i]])

    print(abs(total))
