# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.x = x
        batch_size, in_channel, input_size = x.shape
        input_size_new = int(1+(input_size-self.kernel_size)/self.stride)
        out = np.zeros((batch_size, self.out_channel, input_size_new))
        for i in range(input_size_new):
             x_windows = x[:, :, i * self.stride : i * self.stride + self.kernel_size]
             for j in range(batch_size):
                    for k in range(self.out_channel):
                        out[j, k, i] = np.sum(x_windows[j] * self.W[k])
                        out[j, k, i] += self.b[k]
        return out
    
        
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        for i in range(self.out_channel):
            for j in range(self.in_channel):
                for k in range(self.kernel_size):
                    for l in range(delta.shape[2]):
                        for q in range(delta.shape[0]):
                            self.dW[i,j,k] += self.x[q,j,k + l * self.stride] * delta[q, i, l]
        self.db = np.sum((np.sum(delta, axis=-1)), axis=0)
        dx = np.zeros(self.x.shape)

        newW=np.zeros(self.W.shape)
        for i in range(self.out_channel):
            for j in range(self.in_channel):
                for k in range(self.kernel_size):
                    newW[i,j,k] = self.W[i,j,self.W.shape[2]-k-1]
        pad = np.zeros((delta.shape[0],delta.shape[1],(self.stride*(delta.shape[2]-1)+1)))
        for i in range(delta.shape[0]):
                for j in range(delta.shape[1]):
                    for k in range(delta.shape[2]):
                        pad[i,j,self.stride*k]=delta[i,j,k]

        batch, inchannel, inputsize = dx.shape
        for i in range(batch):
            for j in range(inchannel):
                for t in range(inputsize):
                    for k in range(self.kernel_size):
                        if (int(t+k-self.kernel_size+1))>=0 and (int(t+k-self.kernel_size+1))<=(int(self.stride*(delta.shape[2]-1))):
                            dx[i,j,t] += np.sum(newW[:,j,k] * pad[i,:,t+k-self.kernel_size+1])
        return dx


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        batch_size, in_channel, input_width, input_height = x.shape
        input_width_new = int(1+(input_width-self.kernel_size)/self.stride)
        input_height_new = int(1+(input_height-self.kernel_size)/self.stride)
        out = np.zeros((batch_size, self.out_channel, input_width_new, input_height_new))
        for i in range(input_width_new):
            for j in range(input_height_new):
               x_windows = x[:, :, i * self.stride : i * self.stride + self.kernel_size, j * self.stride : j * self.stride + self.kernel_size]
               for k in range(batch_size):
                    for l in range(self.out_channel):
                        out[k, l, i, j] = np.sum(x_windows[k] * self.W[l])
                        out[k, l, i, j] += self.b[l]
        return out
        
    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        for i in range(self.out_channel):
            for j in range(self.in_channel):
                for k in range(self.kernel_size):
                    for l in range(self.kernel_size):
                        for m in range(delta.shape[2]):
                            for n in range(delta.shape[3]):
                                for q in range(delta.shape[0]):
                                    self.dW[i,j,k,l] += self.x[q,j,k + m * self.stride,l + n * self.stride] * delta[q, i, m, n]
        self.db = np.sum(np.sum(np.sum(delta, axis=-1), axis=-1), axis=0) 
        dx = np.zeros(self.x.shape)
        newW=np.zeros(self.W.shape)
        for i in range(self.out_channel):
            for j in range(self.in_channel):
                for k in range(self.kernel_size):
                    for l in range(self.kernel_size):
                        newW[i,j,k,l] = self.W[i,j,self.W.shape[2]-k-1,self.W.shape[3]-l-1]
        pad = np.zeros((delta.shape[0],delta.shape[1],(self.stride*(delta.shape[2]-1)+1),(self.stride*(delta.shape[3]-1)+1)))
        for i in range(delta.shape[0]):
                for j in range(delta.shape[1]):
                    for k in range(delta.shape[2]):
                        for l in range(delta.shape[3]):
                            pad[i,j,self.stride*k,self.stride*l]=delta[i,j,k,l]

        batch, inchannel, inputhight, inputwidth = dx.shape
        for i in range(batch):
            for j in range(inchannel):
                for t in range(inputhight):
                    for m in range(inputwidth):
                        for k in range(self.kernel_size):
                            for l in range(self.kernel_size):
                                if (int(t+k-self.kernel_size+1))>=0 and (int(t+k-self.kernel_size+1))<=(int(self.stride*(delta.shape[2]-1))) and (int(m+l-self.kernel_size+1))>=0 and (int(m+l-self.kernel_size+1))<=(int(self.stride*(delta.shape[3]-1))):
                                    dx[i,j,t,m] += np.sum(newW[:,j,k,l] * pad[i,:,t+k-self.kernel_size+1,m+l-self.kernel_size+1])
        return dx


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.x = x
        self.b, self.c, self.w = x.shape
        return x.reshape((x.shape[0], -1))



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        
        return delta.reshape(self.x.shape)
    




