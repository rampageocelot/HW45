import numpy as np
from .activation import Softmax
class ScaledDotProductAttention:
    def __init__(self):
        self.eps = 1e10
        self.softmax = Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        self.Q = Q
        self.K = K
        self.V = V
        dk = Q.shape[-1]
        self.dk = dk
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(dk)
        if mask is not None:
            scores = scores + (-self.eps) * mask
        self.attention_scores = self.softmax.forward(scores)
        output = np.matmul(self.attention_scores, V)
        return output
    
    def backward(self, d_output):
        A = self.attention_scores
        d_V = np.matmul(np.swapaxes(A, -1, -2), d_output)
        d_attention_scores = np.matmul(d_output, np.swapaxes(self.V, -1, -2))
        d_scaled_dot_product = self.softmax.backward(d_attention_scores)
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(self.dk)
        d_Q = np.matmul(d_scaled_dot_product, self.K)
        d_K = np.matmul(np.swapaxes(d_scaled_dot_product, -1, -2), self.Q)
        return d_Q, d_K, d_V
