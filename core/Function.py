import numpy as np
import matplotlib.pyplot as plt

class Activation:
    """
    A utility class to fetch activation function objects by name.
    """

    @staticmethod
    def get_activation(activation_name):
        """
        Get an activation function instance by its name.

        Args:
            activation_name (str): Name of the activation function. Supported names are:
                - "relu"
                - "sigmoid"
                - "tanh"
                - "softmax"
                - "leaky_relu"
                - "none" (no activation function)

        Returns:
            Activation: An instance of the corresponding activation function class.

        Raises:
            ValueError: If the activation_name is not recognized.
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


class tanh(Activation):
    """
    Hyperbolic tangent activation function.
    """

    def forward(self, x, test=False, **args):
        """
        Forward pass of the tanh activation function.

        Args:
            x (ndarray): Input tensor.
            test (bool, optional): Flag for test mode. Defaults to False.

        Returns:
            ndarray: Transformed tensor.
        """
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad, **args):
        """
        Backward pass to compute the gradient.

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

    def forward(self, x, **args):
        """
        Forward pass of the sigmoid activation function.

        Args:
            x (ndarray): Input tensor.

        Returns:
            ndarray: Transformed tensor in the range (0, 1).
        """
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad, **args):
        """
        Backward pass to compute the gradient.

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

    def forward(self, x, axiss=1):
        """
        Forward pass of the softmax activation function.

        Args:
            x (ndarray): Input tensor.
            axis (int, optional): Axis along which to compute softmax. Defaults to 1.

        Returns:
            ndarray: Transformed tensor where values sum to 1 along the specified axis.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, grad, **args):
        """
        Backward pass for softmax is generally handled with the loss function.

        Returns:
            ndarray: A tensor of ones (placeholder implementation).
        """
        return grad


class relu(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.
    """

    def forward(self, x, test=False, **args):
        """
        Forward pass of the ReLU activation function.

        Args:
            x (ndarray): Input tensor.
            test (bool, optional): Flag for test mode. Defaults to False.

        Returns:
            ndarray: Tensor with negative values replaced by 0.
        """
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad, **args):
        """
        Backward pass to compute the gradient.

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

    def forward(self, x, alpha=0.01, **args):
        """
        Forward pass of the Leaky ReLU activation function.

        Args:
            x (ndarray): Input tensor.
            alpha (float, optional): Slope for negative values. Defaults to 0.01.

        Returns:
            ndarray: Transformed tensor with small values for negatives.
        """
        self.output = np.where(x > 0, x, alpha * x)
        return self.output

    def backward(self, grad, alpha=0.01, **args):
        """
        Backward pass to compute the gradient.

        Args:
            x (ndarray): Input tensor.
            alpha (float, optional): Slope for negative values. Defaults to 0.01.

        Returns:
            ndarray: Gradient after applying the derivative of Leaky ReLU.
        """
        dx = np.ones_like(self.output)
        dx[self.output < 0] = alpha
        return dx


class no_activation(Activation):
    """
    No activation (identity function).
    """

    def forward(self, x):
        """
        Forward pass with no transformation.

        Args:
            x (ndarray): Input tensor.

        Returns:
            ndarray: The input tensor as-is.
        """
        return x

    def backward(self, grad):
        """
        Backward pass with no transformation.

        Args:
            x (ndarray): Input tensor.

        Returns:
            ndarray: A tensor of ones (no change to gradients).
        """
        return grad
