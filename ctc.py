import numpy as np


class CTC(object):
    """CTC class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument
        --------
        blank: (int, optional)
                blank label index. Default 0.

        """
        self.BLANK = BLANK

    def targetWithBlank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = 1)
                target output

        Return
        ------
        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks
        skipConnect: (np.array, dim = 1)
                    skip connections

        """
        extSymbols = None
        skipConnect = None

        # -------------------------------------------->

        # Your Code goes here
        extSymbols = np.zeros(2*len(target)+1)
        for i in range(len(target)):
            extSymbols[2*i+1] = target[i]
        skipConnect = np.zeros(2*len(target)+1)
        for i in range(len(target)-1):
            skipConnect[2*(i+1)+1] = 1
        # <---------------------------------------------

        return extSymbols, skipConnect

    def forwardProb(self, logits, extSymbols, skipConnect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (output len, out channel))
                forward probabilities

        """
        S, T = len(extSymbols), len(logits)
        alpha = np.zeros(shape=(T, S))
        #print(logits.shape)
        # -------------------------------------------->
        alpha[0,0] = logits[0,int(extSymbols[0])]
        alpha[0,1] = logits[0,int(extSymbols[1])]
        for t in range(1, T):
            for s in range(S):
                a = alpha[t-1,s]
                if s - 1 >= 0:
                    a += alpha[t-1,s-1]
                if skipConnect[s] == 1:
                    a += alpha[t-1,s-2]
                alpha[t, s] = a * logits[t,int(extSymbols[s])]
        # Your Code goes here
        # <---------------------------------------------

        return alpha

    def backwardProb(self, logits, extSymbols, skipConnect):
        """Compute backward probabilities.

        Input
        -----

        logits: (np.array, dim = (input_len, channel))
                predict (log) probabilities

        extSymbols: (np.array, dim = 1)
                    extended label sequence with blanks

        skipConnect: (np.array, dim = 1)
                    skip connections

        Return
        ------
        beta: (np.array, dim = (output len, out channel))
                backward probabilities



        """
        S, T = len(extSymbols), len(logits)
        beta = np.zeros(shape=(T, S))
        self.a = np.zeros(shape=(T, S))
        # -------------------------------------------->
        # Your Code goes here
        beta[T-1,S-1] = 1
        beta[T-1,S-2] = 1
        for t in range(T-1):
            for s in range(S):
                b = beta[T-t-1,s]*logits[T-t-1,int(extSymbols[s])]       
                if s+1 <= S-1:
                    b += beta[T-t-1,s+1]*logits[T-t-1,int(extSymbols[s+1])]      
                if s+2<S and skipConnect[s+2] == 1:
                    b += beta[T-t-1,s+2]*logits[T-t-1,int(extSymbols[s+2])]   
                beta[T-t-2,s] = b       
        # <---------------------------------------------

        return beta

    def postProb(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array)
                forward probability

        beta: (np.array)
                backward probability

        Return
        ------
        gamma: (np.array)
                posterior probability

        """
        gamma = None

        # -------------------------------------------->

        # Your Code goes here
        gamma = np.zeros(alpha.shape)
        for i in range(len(gamma)):
            s = 0
            for j in range(len(gamma[i])):
                gamma[i,j] = alpha[i,j]*beta[i,j]
                s += gamma[i,j]
            for j in range(len(gamma[i])):
                gamma[i,j] = gamma[i,j]/s
        # <---------------------------------------------

        return gamma
