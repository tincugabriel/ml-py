__author__ = 'gabriel'
import numpy as np


class GradientDescent(object):
    def __init__(self, alpha, niter):
        self.alpha = alpha
        self.niter = niter
        self.tetha = None

    def train(self, x, y):
        m = len(x)
        self.tetha = np.zeros(len(x.T))
        for i in range(self.niter):
            hypothesis = np.dot(x, self.tetha)
            loss = hypothesis - y
            gradient = np.dot(x.T, loss) / m
            self.tetha -= self.alpha * gradient
            if all(abs(gradient) < 0.000001):
                print(iter())
                break

    def predict(self, x):
        return np.dot(x, self.tetha)


if __name__ == '__main__':
    import pickle
    regr = pickle.load(open('/tmp/grad.pck', 'rb'))[1]
    test_x = regr[0][-10:]
    test_y = regr[1][-10:]
    train_x = regr[0][:-10]
    train_y = regr[1][:-10]
    regr = GradientDescent(0.0001, 100000)
    regr.train(train_x, train_y)
    pred = regr.predict(test_x)
    print(pred)
    print(test_y)
    print(abs(pred-test_y)/test_y)