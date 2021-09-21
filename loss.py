# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# The following Criterion class will be used again as the basis for a number
# of loss functions (which are in the form of classes so that they can be
# exchanged easily (it's how PyTorch and other ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        self.sum = 0
        for j in range(x.shape[1]):
            self.sum += np.exp(x[0][j])
        self.loss = 0
        for j in range(y.shape[1]):
            self.loss += -y[0][j]*np.log(np.exp(x[0][j])/self.sum)
        for i in range(1, x.shape[0]):
            a = 0
            b = 0
            for j in range(x.shape[1]):
                b += np.exp(x[i][j])
            self.sum = np.hstack((self.sum,b))
            for j in range(y.shape[1]):
                a += -y[i][j]*np.log(np.exp(x[i][j])/b)
            self.loss = np.hstack((self.loss,a))
        return self.loss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """
        x = self.logits
        y = self.labels
        self.sum = 0
        for j in range(x.shape[1]):
            self.sum += np.exp(x[0][j])
        for i in range(1, x.shape[0]):
            b = 0
            for j in range(x.shape[1]):
                b += np.exp(x[i][j])
            self.sum = np.hstack((self.sum,b))
        z = self.sum
        k = np.zeros((x.shape[0],x.shape[1]))
        for j in range(x.shape[0]):
            for i in range(x.shape[1]):
                k[j][i] = np.exp(x[j][i])/z[j]-y[j][i]
        return k


