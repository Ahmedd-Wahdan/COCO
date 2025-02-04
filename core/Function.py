import numpy as np
import matplotlib.pyplot as plt


@staticmethod
def get_activation(activation_name):
    """
    Retrieve an activation function instance based on its name.

    Args:
        activation_name (str): Name of the activation function.

    Returns:
        Activation: Instance of the corresponding activation function class.

    Raises:
        ValueError: If the activation function name is not recognized.
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
    elif activation_name == "none" or activation_name == None:
        return no_activation()
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")
    

class Activation:
    """
    Base class for activation functions.
    """    
    
    def __init__(self, dropout=None):
        """
        Initialize the activation function with optional dropout.

        Args:
            dropout (float, optional): Dropout rate. Defaults to None.
        """
        self.dropout = dropout
        self.output = None
        self.test=False

    def to_train(self):
        self.test = False

    def to_eval(self):
        self.test = True

    def apply_dropout(self):
        """
        Apply dropout to the output if dropout is set and not in test mode.

        Args:
            test (bool): Flag to indicate test mode (dropout is not applied in test mode).
        """
        if self.dropout is not None and not self.test:
            self.mask = (np.random.rand(*self.output.shape) < (1 - self.dropout)).astype(float)
            self.output *= self.mask
            self.output /= (1 - self.dropout)


class tanh(Activation):
    """
    Hyperbolic tangent activation function.
    """
    def __init__(self, dropout=None):
        super().__init__(dropout)

    def __call__(self, x, **args):
        """
        Compute the __call__ pass using the tanh function.

        Args:
            x (ndarray): Input tensor.
            test (bool, optional): Flag for test mode. Defaults to False.

        Returns:
            ndarray: Transformed tensor using tanh.
        """
        self.output = np.tanh(x)
        
        self.apply_dropout()
        
        return self.output

    def backward(self, grad, **args):
        """
        Compute the backward pass (derivative of tanh).

        Args:
            grad (ndarray): Gradient flowing from the next layer.

        Returns:
            ndarray: Gradient after applying the derivative of tanh.
        """
        return grad * (1 - self.output ** 2)


class sigmoid(Activation):
    """
    Sigmoid activation function.
    """
    def __init__(self, dropout=None):
        super().__init__(dropout)
    
    def __call__(self, x, **args):
        """
        Compute the __call__ pass using the sigmoid function.

        Args:
            x (ndarray): Input tensor.

        Returns:
            ndarray: Transformed tensor in the range (0, 1).
        """
        self.output = 1 / (1 + np.exp(-x))

        self.apply_dropout()
        return self.output

    def backward(self, grad, **args):
        """
        Compute the backward pass (derivative of sigmoid).

        Args:
            grad (ndarray): Gradient flowing from the next layer.

        Returns:
            ndarray: Gradient after applying the derivative of sigmoid.
        """
        return grad * (self.output * (1 - self.output))


class softmax(Activation):
    """
    Softmax activation function.
    """
    
    def __init__(self, dropout=None):
        super().__init__(dropout)    

    def __call__(self, x, **args):
        """
        Compute the __call__ pass using the softmax function.

        Args:
            x (ndarray): Input tensor.

        Returns:
            ndarray: Transformed tensor where values sum to 1 along the specified axis.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        self.apply_dropout()
        return self.output

    def backward(self, grad, **args):
        """
        Compute the backward pass for softmax.

        Returns:
            ndarray: Gradient (placeholder implementation).
        """
        return grad


class relu(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.
    """
    def __init__(self, dropout=None):
        super().__init__(dropout)

    def __call__(self, x, **args):
        """
        Compute the __call__ pass using ReLU.

        Args:
            x (ndarray): Input tensor.

        Returns:
            ndarray: Tensor with negative values replaced by 0.
        """
        self.output = np.maximum(0, x)
        self.apply_dropout()
        return self.output

    def backward(self, grad, **args):
        """
        Compute the backward pass (derivative of ReLU).

        Args:
            grad (ndarray): Gradient flowing from the next layer.

        Returns:
            ndarray: Gradient after applying the derivative of ReLU.
        """
        dx = np.ones_like(self.output)
        dx[self.output < 0] = 0
        return grad * dx


class leaky_relu(Activation):
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    """
    def __init__(self, dropout=None):
        super().__init__(dropout)
    
    def __call__(self, x, alpha=0.01, **args):
        """
        Compute the __call__ pass using Leaky ReLU.
        """
        self.output = np.where(x > 0, x, alpha * x)
        self.apply_dropout()
        return self.output

    def backward(self, grad, alpha=0.01, **args):
        """
        Compute the backward pass (derivative of Leaky ReLU).
        """
        dx = np.ones_like(self.output)
        dx[self.output < 0] = alpha
        return grad * dx


class no_activation(Activation):
    """
    No activation (identity function).
    """
    def __init__(self, dropout=None):
        super().__init__(dropout)
    
    def __call__(self, x):
        """Returns input as-is."""
        return x

    def backward(self, grad):
        """Returns gradient as-is."""
        return grad
