__author__ = 'gabriel'
import numpy as np


class LogisticRegression(object):
    def __init__(self, alpha, niter):
        self.alpha = alpha
        self.niter = niter
        self.tetha = None

    def train(self, x, y):
        m = len(x.T)
        self.tetha = np.random.rand(len(x.T))
        for i in range(self.niter):
            hypothesis = LogisticRegression.sigmoid(np.dot(x, self.tetha))
            loss = hypothesis - y
            gradient = np.dot(x.T, loss) / m
            self.tetha -= self.alpha * gradient
            if all(self.tetha < 0.000001):
                break

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def predict(self, x):
        return LogisticRegression.sigmoid(np.dot(x, self.tetha))

if __name__ == '__main__':
    import pickle
    clf = pickle.load(open('/tmp/grad.pck', 'rb'))[0]
    test_x = clf[0][-30:]
    test_y = clf[1][-30:]
    train_x = clf[0][:-30]
    train_y = clf[1][:-30]
    regr = LogisticRegression(0.01, 100000)
    regr.train(train_x, train_y)
    pred = regr.predict(test_x)
    print(pred, test_y)
    print(abs(pred - test_y))