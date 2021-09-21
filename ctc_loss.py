import numpy as np
from ctc import *


class CTCLoss(object):
    """CTC Loss class."""

    def __init__(self, BLANK=0):
        """Initialize instance variables.

        Argument:
                blank (int, optional) – blank label index. Default 0.
        """
        # -------------------------------------------->
        # Don't Need Modify
        super(CTCLoss, self).__init__()
        self.BLANK = BLANK
        self.gammas = []
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # -------------------------------------------->
        # Don't Need Modify
        return self.forward(logits, target, input_lengths, target_lengths)
        # <---------------------------------------------

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward.

        Computes the CTC Loss.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        loss: scalar
            (avg) divergence between the posterior probability γ(t,r) and the input symbols (y_t^r)

        """
        # -------------------------------------------->
        # Don't Need Modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths
        # <---------------------------------------------

        #####  Attention:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # -------------------------------------------->
        # Don't Need Modify
        B, _ = target.shape
        totalLoss = np.zeros(B)
        self.extSymbols = []
        # <---------------------------------------------
        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Extend Sequence with blank ->
            #     Compute forward probabilities ->
            #     Compute backward probabilities ->
            #     Compute posteriors using total probability function
            #     Compute Expected Divergence and take average on batches
            # <---------------------------------------------
            # -------------------------------------------->
            extSymbols, skipConnect = CTC().targetWithBlank(self.target[b][0:self.target_lengths[b]])
            alpha = CTC().forwardProb((self.logits[0:int(self.input_lengths[b]),b,:]), extSymbols, skipConnect)
            beta = CTC().backwardProb((self.logits[0:int(self.input_lengths[b]),b,:]), extSymbols, skipConnect)
            gamma = CTC().postProb(alpha, beta)
            for i in range(len(gamma)):
                for j in range(len(gamma[i])):
                    totalLoss[b] += -gamma[i,j]*(np.log(self.logits[i,b,int(extSymbols[j])]))
            #totalLoss[b] /=target_lengths[b]
            self.gammas.append(gamma)
            self.extSymbols.append(extSymbols)
        totalLoss = np.sum(totalLoss)/(B)
            # Your Code goes here
            # <---------------------------------------------

        return totalLoss

    def backward(self):
        """CTC loss backard.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        logits: (seqlength, batch_size, len(Symbols))
                log probabilities (output sequence) from the RNN/GRU

        target: (batch_size, paddedtargetlen)
                target sequences.

        input_lengths: (batch_size,)
                        lengths of the inputs.

        target_lengths: (batch_size,)
                        lengths of the target.

        Returns
        -------
        dY: scalar
            derivative of divergence wrt the input symbols at each time.

        """
        # -------------------------------------------->
        # Don't Need Modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)
        # <---------------------------------------------

        for b in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # <---------------------------------------------

            # -------------------------------------------->

            # Your Code goes here
            for i in range(len(self.gammas)):
                for j in range(len(dY[i,b])):
                    for k in range(len(self.extSymbols[b])):
                        if j == int(self.extSymbols[b][k]):
                            dY[i,b,j] += -self.gammas[b][i,k]/self.logits[i,b,j]
            # <---------------------------------------------
        return dY
