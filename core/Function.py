import numpy as np
import matplotlib.pyplot as plt
class Activation:
    @staticmethod
    def get_activation(activation_name):
        """
        Get activation function by name.
        """
        if activation_name is None:
            return no_activation()
        elif activation_name == "relu":
            return relu()
        elif activation_name == "sigmoid":
            return sigmoid()
        elif activation_name == "tanh":
            return tanh()
        elif activation_name == "softmax":
            return softmax()
        elif activation_name == "leaky_relu":
            return leaky_relu()
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")

class tanh(Activation):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1 - np.tanh(x) ** 2

class sigmoid(Activation):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x):
        return self.forward(x) * (1 - self.forward(x))

class softmax(Activation):
    def forward(self, x, axis=1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def backward(self, x):
        # Softmax derivative is handled separately in the loss function
        return np.ones_like(x)

class relu(Activation):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        dx = np.ones_like(x)
        dx[x < 0] = 0
        return dx

class leaky_relu(Activation):
    def forward(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def backward(self, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx

class no_activation(Activation):
    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)