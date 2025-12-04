import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        """
      
        self.input_shape = A.shape
        
       
        self.A = A

       
        A_flat = A.reshape(-1, A.shape[-1])

       
        Z_flat = np.dot(A_flat, self.W.T) + self.b

        
        new_shape = self.input_shape[:-1] + (self.W.shape[0],)
        Z = Z_flat.reshape(new_shape)
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        
        dLdZ_flat = dLdZ.reshape(-1, dLdZ.shape[-1])
        
        
        A_flat = self.A.reshape(-1, self.A.shape[-1])

       
        
        
        self.dLdA = np.dot(dLdZ_flat, self.W)

        
        self.dLdW = np.dot(dLdZ_flat.T, A_flat)

        self.dLdb = dLdZ_flat.sum(axis=0).reshape(-1, 1)

        
        return self.dLdA.reshape(self.input_shape)
