import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        dim = self.dim if self.dim >= 0 else self.dim + Z.ndim

        # Numerically stable softmax:
        # subtract max along dim so exp() does not blow up
        Z_max = np.max(Z, axis=dim, keepdims=True)
        Z_shifted = Z - Z_max
        expZ = np.exp(Z_shifted)
        sum_expZ = np.sum(expZ, axis=dim, keepdims=True)
        self.A = expZ / sum_expZ

        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        A = self.A
        shape = A.shape
        dim = self.dim if self.dim >= 0 else self.dim + len(shape)
        A_moved = np.moveaxis(A, dim, -1)
        dLdA_moved = np.moveaxis(dLdA, dim, -1)
        N = int(np.prod(A_moved.shape[:-1]))
        C = A_moved.shape[-1]
        A_flat = A_moved.reshape(N, C)
        dLdA_flat = dLdA_moved.reshape(N, C)
        dot = np.sum(dLdA_flat * A_flat, axis=1, keepdims=True)
        dLdZ_flat = A_flat * (dLdA_flat - dot)
        dLdZ_moved = dLdZ_flat.reshape(A_moved.shape)
        dLdZ = np.moveaxis(dLdZ_moved, -1, dim)
        return dLdZ
 

    