
import numpy as np
import matplotlib.pyplot as plt
from core.wrapper import timing_decorator
import core.Function as fn
import core.optim as opt
import core.loss as ls
from core.operations import FastConvolver
from numpy.lib.stride_tricks import sliding_window_view
from core.Datasets import Dataset
import inspect
import re

class Module:
    def __init__(self):
        pass

class Layer(Module):
    def __init__(self):
        """
        
        """
        self.initialize_type = None
        self.loss_wrt_output = None
        self.loss_wrt_input = None
        self.weights=None
        self.bias=None
        self.grad_w = None
        self.grad_b = None
        self.momentum_w = None
        self.momentum_b = None
        self.Accumelated_Gsquare_w = None
        self.Accumelated_Gsquare_b = None
        self.t = 1 #used in ADAM and NADAM
        self.eps = 1e-7
        
        # self.dropout = None
    def forward():
        pass

    def backward():
        pass

    def initialize_weights(self, shape,initialize_type):
        pass

    

class Linear(Layer):
    """
    A fully connected linear layer that performs an affine transformation: 
    output = input * weights + bias. Supports different weight initialization methods
    and optional dropout for regularization.
    """
    
    def __init__(self, shape: tuple, initialize_type="random"):
        """
        Initializes a linear layer with specified weight initialization and optional dropout.
        
        Args:
            shape (tuple): A tuple of (input_dim, output_dim).
            initialize_type (str, optional): Weight initialization method. Defaults to "random".
            dropout (float, optional): Dropout rate. Defaults to None.
        """
        super().__init__()
        
        self.weights, self.bias = self.initialize_weights(shape, initialize_type)


    def initialize_weights(self, shape, initialize_type):
        """
        Initializes weights and biases based on the specified initialization method.
        
        Args:
            shape (tuple): (input_dim, output_dim) specifying layer dimensions.
            initialize_type (str): Initialization method (zero, random, xavier, he, lecun).
            
        Returns:
            tuple: Initialized weight matrix and bias vector.
        """
        self.momentum_w = np.zeros(shape)
        self.momentum_b = np.zeros((1, shape[1]))
        self.Accumelated_Gsquare_w = np.zeros(shape)
        self.Accumelated_Gsquare_b = np.zeros((1, shape[1]))

        input_dim, output_dim = shape

        if initialize_type == 'zero':
            w = np.zeros((input_dim, output_dim))

        elif initialize_type == 'random':
            w = np.random.randn(input_dim, output_dim) * 0.01

        elif initialize_type == 'xavier':  # Also known as Glorot
            fan_in, fan_out = input_dim, output_dim
            bound = np.sqrt(6 / (fan_in + fan_out))
            w = np.random.uniform(-bound, bound, size=(input_dim, output_dim))

        elif initialize_type == 'he':  # He (Kaiming) initialization for ReLU
            std = np.sqrt(2 / input_dim)
            w = np.random.normal(0, std, size=(input_dim, output_dim))

        elif initialize_type == 'lecun':  # LeCun initialization for tanh/sigmoid
            std = np.sqrt(1 / input_dim)
            w = np.random.normal(0, std, size=(input_dim, output_dim))

        else:
            raise ValueError(f"Unknown initialization method: {initialize_type}")

        b = np.zeros((1, output_dim))  # Bias is always initialized to zeros
        return w, b

    def __call__(self, input,**kwargs):
        """
        Performs the forward pass of the linear layer.
        
        Args:
            input (np.ndarray): Input tensor.
            test (bool, optional): Whether the model is in test mode (disables dropout). Defaults to False.
            
        Returns:
            np.ndarray: Output of the affine transformation.
        """
        self.input = input
        if self.input.ndim == 4:  # If input is 4D (e.g., [B, C, H, W])
            B, C, H, W = self.input.shape
            self.input = self.input.reshape(B, -1)  # Flatten to [B, C*H*W]
        elif self.input.ndim != 2:  # If input is not 2D or 4D, raise an error
            raise ValueError(f"Linear layer received input of unsupported shape {self.input.shape}")
        
        self.out = (self.input @ self.weights) + self.bias  # Affine transformation
            
        
        return self.out

    def backward(self, error_wrt_output, **kwargs):
        """
        Performs the backward pass, computing gradients of the loss with respect to input, weights, and bias.
        
        Args:
            error_wrt_output (np.ndarray): Gradient of loss with respect to layer output.
            **kwargs: Optional L1 and L2 regularization parameters.
        
        Returns:
            np.ndarray: Gradient of loss with respect to layer input.
        """
        l1 = kwargs.get('l1', None)
        l2 = kwargs.get('l2', None)
        
        # Gradient of loss w.r.t. weights
        self.grad_w = self.input.T @ error_wrt_output

        if l1 is not None:
            self.grad_w += (l1 * np.sign(self.weights))
        if l2 is not None:
            self.grad_w += (l2 * self.weights)

        # Gradient of loss w.r.t. bias
        self.grad_b = np.sum(error_wrt_output, axis=0, keepdims=True)
        
        # Gradient of loss w.r.t. input
        self.loss_wrt_input = error_wrt_output @ self.weights.T
        
        assert self.grad_w.shape == self.weights.shape
        assert self.grad_b.shape == self.bias.shape
        assert self.loss_wrt_input.shape == self.input.shape
        
        return self.loss_wrt_input

class Conv2d(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, initialize_type="xavier"):
        """
        Initializes a 2D convolutional layer.
        
        Args:
            input_channels (int): Number of channels in the input tensor.
            output_channels (int): Number of filters (output channels) in the layer.
            kernel_size (int): Size of the square convolutional kernel.
            stride (int, optional): Step size for moving the convolutional kernel. Defaults to 1.
            padding (int, optional): Number of pixels to pad around the input. Defaults to 0.
            initialize_type (str, optional): Method for initializing weights. Defaults to "xavier".
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Convolution operation utility
        self.convolver = FastConvolver()

        # Shape of the weight tensor (filters)
        self.kernels_shape = (output_channels, input_channels, kernel_size, kernel_size)

        # Initialize weights and biases based on selected method
        self.weights, self.bias = self.initialize_weights(initialize_type)

    def initialize_weights(self, initialize_type="xavier"):
        """
        Initializes weights and biases for the Conv2D layer.

        Args:
            initialize_type (str): Type of initialization ('zero', 'random', 'xavier', 'he', 'lecun').

        Returns:
            tuple: Initialized weight and bias tensors.
        """
        if initialize_type == 'zero':
            w = np.zeros(self.kernels_shape)
        elif initialize_type == 'random':
            w = np.random.randn(*self.kernels_shape) * 0.01
        elif initialize_type == 'xavier':
            fan_in = self.input_channels * self.kernel_size * self.kernel_size
            fan_out = self.output_channels * self.kernel_size * self.kernel_size
            bound = np.sqrt(6 / (fan_in + fan_out))
            w = np.random.uniform(-bound, bound, size=self.kernels_shape)
        elif initialize_type == 'he':
            w = np.random.randn(*self.kernels_shape) * np.sqrt(2 / (self.input_channels * self.kernel_size * self.kernel_size))
        elif initialize_type == 'lecun':
            w = np.random.randn(*self.kernels_shape) * np.sqrt(1 / (self.input_channels * self.kernel_size * self.kernel_size))
        else:
            raise ValueError(f"Unknown initialization method: {initialize_type}")

        # Bias is usually initialized to zero
        b = np.zeros((1, self.output_channels, 1, 1))

        self.grad_w = np.zeros_like(w)
        self.grad_b = np.zeros_like(b)

        # Momentum storage (for optimizers like Adam, Momentum SGD, etc.)
        self.momentum_w = np.zeros_like(w)
        self.momentum_b = np.zeros_like(b)

        # Storage for squared gradients (for Adam, RMSprop, etc.)
        self.Accumelated_Gsquare_w = np.zeros_like(w)
        self.Accumelated_Gsquare_b = np.zeros_like(b)

        return w, b

    def __call__(self, input,**kwargs):
        """
        Performs the forward pass of the convolution operation.
        
        Args:
            input (np.ndarray): Input tensor of shape (batch_size, input_channels, height, width).
            test (bool, optional): Whether the model is in test mode (disables dropout). Defaults to False.
        
        Returns:
            np.ndarray: Output tensor after convolution.
        """
        if input.ndim != 4:
            raise ValueError(f"Expected 4D input (batch_size, channels, height, width), got shape {input.shape}")
        if input.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {input.shape[1]}")

        self.input = input
        self.output, self.col_matrix = self.convolver.convolve(self.input, self.weights, stride=self.stride, padding=self.padding)
        self.output += self.bias

        return self.output

    def backward(self, output_grad, **kwargs):
        """
        Computes the backward pass of the convolution operation.
        
        Args:
            output_grad (np.ndarray): Gradient of the loss with respect to the output tensor.
        
        Returns:
            np.ndarray: Gradient of the loss with respect to the input tensor.
        """
        B, F, H_out, W_out = self.output.shape
        
        # Gradient wrt biases
        output_grad = output_grad.reshape(self.output.shape)
        self.grad_b = np.sum(output_grad, axis=(0, 2, 3), keepdims=True)
        assert self.grad_b.shape == self.bias.shape
        
        # Gradient wrt weights
        grad_reshaped = output_grad.transpose(1, 0, 2, 3).reshape(self.output_channels, -1)  # Shape: (F, B * H_out * W_out)
        grad_kernel_matrix = grad_reshaped @ self.col_matrix  # Use cached `col_matrix`
        self.grad_w = grad_kernel_matrix.reshape(self.output_channels, self.input_channels, self.kernel_size, self.kernel_size)
        assert self.grad_w.shape == self.weights.shape
        
        # Gradient wrt input
        kernel_matrix = self.weights.reshape(self.output_channels, -1).T  # Shape: (C*k*k, F)
        dout_matrix = output_grad.transpose(0, 2, 3, 1).reshape(B * H_out * W_out, F)
        dX_col = dout_matrix @ kernel_matrix.T

        # Use col2im_accumulation to fold dX_col back to the padded input shape.
        dInput_padded = self.convolver.col2im_accumulation(
            dX_col=dX_col,
            input_shape=self.input.shape, 
            filter_height=self.kernel_size,
            filter_width=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        
        # Remove the padding to recover gradient w.r.t. the original input.
        if self.padding > 0:
            dInput = dInput_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dInput = dInput_padded

        assert dInput.shape == self.input.shape
        return dInput



class MaxPool2d(Module):
    """
    Implements a 2D max pooling layer.

    Args:
        kernel_size (int or tuple): Size of the pooling window.
        stride (int or tuple, optional): Stride of the pooling operation. Defaults to kernel_size.
        padding (int, optional): Amount of zero-padding added to the input. Defaults to 0.
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding
        self.cache = {}

    def __call__(self, X, **kwargs):
        """
        Forward pass for max pooling.

        Args:
            X (np.ndarray): Input tensor of shape (batch_size, channels, height, width).
            test (bool, optional): Whether the model is in test mode. Defaults to False.

        Returns:
            np.ndarray: Output tensor after max pooling.
        """
        B, C, H, W = X.shape
        H_k, W_k = self.kernel_size
        stride_h, stride_w = self.stride

        padded_X = np.pad(
            X,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant'
        )

        windows = sliding_window_view(padded_X, (H_k, W_k), axis=(2, 3))
        windows = windows[:, :, ::stride_h, ::stride_w, :, :]

        max_windows = windows.reshape(B, C, -1, H_k * W_k)
        max_vals = max_windows.max(axis=-1)
        max_indices = max_windows.argmax(axis=-1)

        output = max_vals.reshape(B, C, -1)
        H_out = int(np.sqrt(output.shape[2]))
        output = output.reshape(B, C, H_out, H_out)

        self.cache['out_shape'] = output.shape
        self.cache['input'] = X
        self.cache['max_indices'] = (
            max_indices,
            windows.shape,
            (stride_h, stride_w)
        )
        return output

    def backward(self, grad_output, **kwargs):
        """
        Backward pass for max pooling.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        X = self.cache['input']
        max_indices, window_shape, strides = self.cache['max_indices']
        grad_output = grad_output.reshape(*self.cache['out_shape'])
        B, C, H_out, W_out = grad_output.shape
        stride_h, stride_w = strides
        kernel_H, kernel_W = self.kernel_size
        H_in, W_in = X.shape[2], X.shape[3]
        padding = self.padding

        max_indices_4d = max_indices.reshape(B, C, H_out, W_out)

        # Generate all indices for B, C, H_out, W_out
        b, c, i, j = np.indices((B, C, H_out, W_out))

        # Compute starting positions and offsets
        h_start = i * stride_h
        w_start = j * stride_w
        h_offset = max_indices_4d // kernel_W
        w_offset = max_indices_4d % kernel_W

        # Calculate positions in padded and original input
        h_padded = h_start + h_offset
        w_padded = w_start + w_offset
        h_original = h_padded - padding
        w_original = w_padded - padding

        # Mask for valid positions within original input dimensions
        valid = (h_original >= 0) & (h_original < H_in) & (w_original >= 0) & (w_original < W_in)

        # Extract valid indices and corresponding gradients
        valid_b = b[valid]
        valid_c = c[valid]
        valid_h = h_original[valid]
        valid_w = w_original[valid]
        valid_grad = grad_output[valid]

        # Accumulate gradients using vectorized scatter-add
        grad_input = np.zeros_like(X)
        np.add.at(grad_input, (valid_b, valid_c, valid_h, valid_w), valid_grad)

        return grad_input


class batchnorm1d(Layer):
    """
    Implements Batch Normalization for 1D input.
    This layer normalizes the input batch-wise, stabilizing the training and improving convergence speed.
    
    Attributes:
        eps (float): Small constant for numerical stability.
        betanorm (float): Momentum factor for moving averages.
        input_normalized (np.ndarray or None): Stores the normalized input.
        input (np.ndarray or None): Stores the original input for backward pass.
        mean (np.ndarray or None): Stores the mean of the batch.
        var (np.ndarray or None): Stores the variance of the batch.
        running_mean (np.ndarray): Running mean used during inference.
        running_variance (np.ndarray): Running variance used during inference.
        weights (np.ndarray): Scaling factor (gamma).
        bias (np.ndarray): Shift factor (beta).
        momentum_w (np.ndarray): Momentum storage for weight updates.
        momentum_b (np.ndarray): Momentum storage for bias updates.
        Accumelated_Gsquare_w (np.ndarray): Storage for squared weight gradients.
        Accumelated_Gsquare_b (np.ndarray): Storage for squared bias gradients.
    """
    def __init__(self, dim, activation="none", initialize_type="zero", dropout=None):
        """
        Initializes the BatchNorm1D layer.
        
        Args:
            dim (int): Number of input features.
            activation (str, optional): Activation type (default is "none").
            initialize_type (str, optional): Initialization method (default is "zero").
            dropout (float, optional): Dropout rate (not used in batch norm).
        """
        super().__init__()
        self.eps = 1e-5
        self.input_normalized = None
        self.input = None
        self.betanorm = 0.9
        self.mean = None
        self.var = None
        self.initialize_weights(dim, initialize_type)

    def initialize_weights(self, dims, initialize_type):
        """
        Initializes weights and bias parameters.
        
        Args:
            dims (int): Number of input features.
            initialize_type (str): Type of initialization (not used for batch norm).
        """
        self.running_mean = np.zeros((1, dims))
        self.running_variance = np.ones((1, dims))
        self.weights = np.ones((1, dims))  # Gamma initialized to 1
        self.bias = np.zeros((1, dims))  # Beta initialized to 0
        self.momentum_w = np.zeros((1, dims))
        self.momentum_b = np.zeros((1, dims))
        self.Accumelated_Gsquare_w = np.zeros((1, dims))
        self.Accumelated_Gsquare_b = np.zeros((1, dims))
        

    def __call__(self, input,test=False,**kwargs):
        """
        Forward pass of batch normalization.
        
        Args:
            input (np.ndarray): Input tensor of shape (batch_size, features).
            test (bool, optional): If True, uses running statistics for inference.

        Returns:
            np.ndarray: Normalized and scaled output tensor.
        """
        self.input = input
        if not test:
            self.mean = np.mean(self.input, axis=0, keepdims=True)
            self.var = np.var(self.input, axis=0, keepdims=True)
            
            # Update running statistics using exponential moving average
            self.running_variance = (self.betanorm * self.running_variance) + ((1 - self.betanorm) * self.var)
            self.running_mean = (self.betanorm * self.running_mean) + ((1 - self.betanorm) * self.mean)
            
            # Normalize input
            self.input_normalized = (self.input - self.mean) / np.sqrt(self.var + self.eps)
            
            # Apply scale and shift
            self.out = self.weights * self.input_normalized + self.bias
        else:
            # Use precomputed running statistics during inference
            self.input_normalized = (self.input - self.running_mean) / np.sqrt(self.running_variance + self.eps)
            self.out = self.weights * self.input_normalized + self.bias
        
        return self.out

    def backward(self, error_wrt_output,**kwargs):
        """
        Backward pass for batch normalization.
        
        Args:
            error_wrt_output (np.ndarray): Gradient of the loss with respect to output.
            l1 (float): L1 regularization coefficient (not used in batch norm).
            l2 (float): L2 regularization coefficient (not used in batch norm).
        
        Returns:
            np.ndarray: Gradient with respect to input.
        """
        batch_size = error_wrt_output.shape[0]

        # Gradient w.r.t. gamma (weights) and beta (bias)
        normalized_input_grad = error_wrt_output * self.weights  # Gradients w.r.t. gamma (scale)
        variance_grad = np.sum(
            normalized_input_grad * (self.input - self.mean) * (-0.5) * np.power((self.var + self.eps), -1.5),
            axis=0, keepdims=True
        )
        mean_grad = np.sum(normalized_input_grad * (-1 / np.sqrt(self.var + self.eps)), axis=0, keepdims=True) + (
            variance_grad * np.mean(-2 * (self.input - self.mean), axis=0, keepdims=True)
        )

        self.loss_wrt_input = (
            normalized_input_grad * (1 / np.sqrt(self.var + self.eps)) +
            (variance_grad * 2 * (self.input - self.mean) / batch_size) +
            (mean_grad / batch_size)
        )

        self.grad_w = np.sum(error_wrt_output * self.input_normalized, axis=0, keepdims=True)
        self.grad_b = np.sum(error_wrt_output, axis=0, keepdims=True) / batch_size

        return self.loss_wrt_input

        