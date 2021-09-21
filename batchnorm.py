# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """

        # if eval:
        #    # ???
        if eval:
            self.mean = x - self.running_mean
            self.norm = self.mean/((np.sqrt(self.running_var+self.eps)))

        else:
            self.x = x
            
            self.mean = x-np.sum(x, axis = 0, keepdims=True)/self.x.shape[0] 

            self.var = 1./self.x.shape[0] * np.sum(self.mean**2, axis = 0, keepdims=True)

            self.norm = (self.mean)/ np.sqrt(self.var + self.eps)

            self.running_mean = self.alpha*self.running_mean+(1-self.alpha)*self.mean
            self.running_var = self.alpha*self.running_var+(1-self.alpha)*self.var
            
        self.out= self.gamma*self.norm+ self.beta
        
        # self.mean = # ???
        # self.var = # ???
        # self.norm = # ???
        # self.out = # ???

        # Update running batch statistics
        # self.running_mean = # ???
        # self.running_var = # ???

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        self.dxhead = delta * self.gamma
        self.dbeta = np.sum(delta, axis=0, keepdims=True)
        self.dgamma = np.sum(delta*self.norm, axis=0, keepdims=True)
        
        dx1 =  self.dxhead/ np.sqrt(self.var + self.eps)
        
        dx2 =  -self.mean * np.ones((delta.shape[0],delta.shape[1])) * np.sum(self.dxhead * self.mean, axis=0, keepdims=True)/((np.sqrt(self.var + self.eps)**3)*delta.shape[0])
        dx3 =-(np.sum(dx1+dx2, axis=0, keepdims=True) * np.ones((delta.shape[0],delta.shape[1]))) /delta.shape[0] 
        self.dx = dx1 + dx2 + dx3
        
        
        return self.dx, self.dbeta, self.dgamma
'''
x=np.array([[0,0,0,0,0,0,1,0,0,0],[0.01,0.09,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])
y=np.array([[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0]])
BatchNorm=BatchNorm(2)
BatchNorm(x,y)

'''
