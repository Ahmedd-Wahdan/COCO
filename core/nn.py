
import numpy as np
import matplotlib.pyplot as plt
from core.wrapper import timing_decorator
import core.Function as fn
import core.optim as opt
import core.loss as ls
from core.operations import FastConvolver
from numpy.lib.stride_tricks import sliding_window_view
from core.Datasets import Dataset
import time


class Layer:
    def __init__(self,activation="none", initialize_type="he"):
        """
        
        """
        self.activation = fn.Activation.get_activation(activation)
        self.initialize_type = initialize_type
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

    def initialize_weights(self, dims,initialize_type):
        self.momentum_w = np.zeros(dims)
        self.momentum_b = np.zeros((1, dims[1]))
        self.Accumelated_Gsquare_w = np.zeros(dims)
        self.Accumelated_Gsquare_b = np.zeros((1, dims[1]))
        input_dim, output_dim = dims
        if initialize_type == 'zero':
            w = np.zeros((input_dim, output_dim))
        elif initialize_type == 'random':
            w = np.random.randn(input_dim, output_dim) * 0.01
        elif initialize_type == 'xavier':
            limit = np.sqrt(6 / (input_dim + output_dim))
            w = np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif initialize_type == 'he':
            w = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        elif initialize_type == 'lecun':
            w = np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
        else:
            raise ValueError(f"Unknown initialization method: {initialize_type}")

        b = np.zeros((1, output_dim))  # Bias is a row vector
        return w, b

    

class Linear(Layer):
    def __init__(self, dims: tuple, activation="none", initialize_type="he", dropout=None):
        """
        Initialize a linear layer.

        Args:
            dims (tuple): A tuple of (input_dim, output_dim).
            activation (str, optional): Activation function name. Defaults to "none".
            initialize_type (str, optional): Weight initialization method. Defaults to "he".
            dropout (float, optional): Dropout rate. Defaults to None.
        """
        super().__init__(activation, initialize_type)
        self.input_dim, self.output_dim = dims
        self.weights, self.bias = self.initialize_weights(dims, initialize_type)
        print("weights",self.weights.shape)
        print("bias",self.bias.shape)
        self.dropout = dropout

    def initialize_weights(self, dims, initialize_type):
        self.momentum_w = np.zeros(dims)
        self.momentum_b = np.zeros((1, dims[1]))
        self.Accumelated_Gsquare_w = np.zeros(dims)
        self.Accumelated_Gsquare_b = np.zeros((1, dims[1]))
        input_dim, output_dim = dims
        if initialize_type == 'zero':
            w = np.zeros((input_dim, output_dim))
        elif initialize_type == 'random':
            w = np.random.randn(input_dim, output_dim) * 0.01
        elif initialize_type == 'xavier':
            limit = np.sqrt(6 / (input_dim + output_dim))
            w = np.random.uniform(-limit, limit, (input_dim, output_dim))
        elif initialize_type == 'he':
            w = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        elif initialize_type == 'lecun':
            w = np.random.randn(input_dim, output_dim) * np.sqrt(1 / input_dim)
        else:
            raise ValueError(f"Unknown initialization method: {initialize_type}")

        b = np.zeros((1, output_dim))  # Bias is a row vector
        return w, b

    def forward(self, input, test=False):
        # start_time = time.time()
  
        self.input = input
        if self.input.ndim == 4:  # If input is 4D (e.g., [B, C, H, W])
            B, C, H, W = self.input.shape
            self.input = self.input.reshape(B, -1)  # Flatten to [B, C*H*W]
        elif self.input.ndim != 2:  # If input is not 2D or 4D, raise an error
            raise ValueError(f"Linear layer received input of unsupported shape {self.input.shape}")
        self.z = (self.input @ self.weights) + self.bias  # Linear transformation
        self.out = self.activation.forward(self.z)  # Apply activation

        # Apply dropout during training
        if self.dropout is not None and not test:
            self.mask = (np.random.rand(*self.out.shape) < (1 - self.dropout)).astype(float)
            self.out *= self.mask
            self.out /= (1 - self.dropout)

        # print("output_of_forward",self.out.shape)
        # end_time = time.time()
        # print(f"Linear forward took {end_time - start_time:.4f}")
        return self.out

    def backward(self, error_wrt_output,**kwargs):
        # start_time = time.time()
        l1 = kwargs.get('l1', None)
        l2 = kwargs.get('l2', None)
    
        da_dz = self.activation.backward(self.z)
        batch_size = error_wrt_output.shape[0]
        # Gradient of loss w.r.t. pre-activation (z)
        d_z = error_wrt_output * da_dz

        # Gradient of loss w.r.t. weights
        self.grad_w = np.dot(self.input.T, d_z)
        if l1 is not None:
            self.grad_w += (l1*np.sign(self.weights))

        if l2 is not None:
            self.grad_w += (l2*self.weights)

        # Gradient of loss w.r.t. bias
        self.grad_b = np.sum(d_z, axis=0, keepdims=True)

        # print("grad_w",self.grad_w)
        # Gradient of loss w.r.t. input
        self.loss_wrt_input = np.dot(d_z, self.weights.T)

        assert self.grad_w.shape == self.weights.shape
        assert self.grad_b.shape == self.bias.shape
        assert self.loss_wrt_input.shape == self.input.shape
        # end_time = time.time()
        # print(f"Linear backward took {end_time - start_time:.4f}")
        return self.loss_wrt_input

class Conv2d(Layer):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.convolver = FastConvolver()
        
        self.kernels_shape = (output_channels, input_channels, kernel_size, kernel_size) #(number of filters, number of channels, filter height, filter width)

        self.weights = np.random.randn(*self.kernels_shape)
        self.grad_w = np.zeros_like(self.weights)
        self.bias  = np.random.randn(self.output_channels)
        self.bias = self.bias.reshape(1, -1,1,1)
        self.grad_b = np.zeros_like(self.bias)
        self.momentum_w = np.zeros(self.kernels_shape)
        self.momentum_b = np.zeros_like(self.bias)
        self.Accumelated_Gsquare_w = np.zeros(self.kernels_shape)
        self.Accumelated_Gsquare_b = np.zeros_like(self.bias)


    def forward(self, input, test=False):
        # start_time = time.time()
        if input.ndim != 4:
            raise ValueError(f"Expected 4D input (batch_size, channels, height, width), got shape {input.shape}")
        if input.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {input.shape[1]}")

        self.input = input
        batch_size,channels,h_in,w_in = input.shape

        # Output Dims
        h_out = ((h_in + 2 * self.padding - self.kernel_size) // self.stride) + 1
        w_out = ((w_in + 2 * self.padding - self.kernel_size) // self.stride) + 1

        # Output Tensor
        output = np.zeros((batch_size, self.output_channels, h_out, w_out))

        # Processing Batch
        output = self.convolver.convolve(self.input, self.weights, self.stride, self.padding)

        self.output = output
        self.output += self.bias  # Add biases to each output channel
        # end_time = time.time()
        # print(f"conv forward took {end_time - start_time:.4f}")
        return self.output

    def backward(self, output_grad,**kwargs):
        # start_time = time.time()
        B, C, H, W = self.input.shape
        B, F, H_out, W_out = output_grad.shape
        _, C_k, H_k, W_k = self.weights.shape
        output_grad.reshape(self.output.shape)
        #grad wrt biases
        self.grad_b = self.dBiases = np.sum(output_grad, axis=(0, 2, 3), keepdims=True)

        assert self.dBiases.shape == self.bias.shape
       #grad wrt kernels [we multiply the gradient of each filter with the corresponding patch that was multipled by the filter in the forward pass] 
        col_matrix, _, _ = self.convolver._im2col(self.input, (C_k, H_k, W_k), self.stride, self.padding)
       #col_matrix is of shape (B*H_out*W_out, C*H_k*W_k) (number of patches, number of elements in each patch)
        grad_reshaped = output_grad.reshape(F, -1)  
        #grad_reshaped is of shape (F, B*H_out*W_out) (number of filters, number of patches)
       

        #how each patch contributed to each filter
        grad_kernel_matrix = grad_reshaped @ col_matrix

        #reshape to original Filter shape
        self.grad_w = self.dKernels = grad_kernel_matrix.reshape(F, C_k, H_k, W_k)

        assert self.dKernels.shape == self.weights.shape

        #flip kernels and transpose to make the output channels of the convolution matches the input channels
        flipped_kernels = np.flip(self.weights, axis=(2, 3)).transpose(1, 0, 2, 3)


        # padd to compensate padding in the forward
        pad_h = (self.kernel_size - 1 - self.padding)
        pad_w = (self.kernel_size - 1 - self.padding)
        if self.stride > 1:
            pad_h += (self.stride - 1)
            pad_w += (self.stride - 1)

        
       
        #pad output_grad
        output_grad_padded = np.pad(output_grad,((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),mode='constant')
        #gradient wrt input (prev layer output)
        dInput = self.convolver.convolve(output_grad_padded,flipped_kernels,stride=1,padding=0)
        
        assert dInput.shape == self.input.shape
        # end_time = time.time()
        # print(f"conv backward took {end_time - start_time:.4f}")
        return dInput

class MaxPool2d:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding
        self.convolver = FastConvolver()
        self.cache = {}

    def forward(self, X, test=False):
        # start_time = time.time()
        """Vectorized max pooling forward pass"""
        B, C, H, W = X.shape
        H_k, W_k = self.kernel_size
        stride_h, stride_w = self.stride

        # Pad input
        padded_X = np.pad(
            X,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            mode='constant'
        )

        # Extract sliding windows
        windows = sliding_window_view(padded_X, (H_k, W_k), axis=(2, 3))
        windows = windows[:, :, ::stride_h, ::stride_w, :, :]

        # Compute max and indices
        max_windows = windows.reshape(B, C, -1, H_k * W_k)
        max_vals = max_windows.max(axis=-1)
        max_indices = max_windows.argmax(axis=-1)

        # Reshape output to 4D
        output = max_vals.reshape(B, C, -1)
        H_out = output.shape[-1]
        W_out = int(np.sqrt(H_out))
        output = output.reshape(B, C, H_out // W_out, W_out)

        # Store cache for backward
        self.cache['out_shape'] = output.shape
        self.cache['input'] = X
        self.cache['windows'] = windows
        self.cache['max_indices'] = (
            max_indices,
            windows.shape,
            (stride_h, stride_w)
        )
        # end_time = time.time()
        # print(f"maxpool forward took {end_time - start_time:.4f}")
        return output

    def backward(self, grad_output,**kwargs):
        # start_time = time.time()
        """Vectorized max pooling backward pass"""
        X = self.cache['input']
        max_indices, window_shape, strides = self.cache['max_indices']
        grad_output = grad_output.reshape(*self.cache['out_shape'])
        # Initialize gradient
        grad_input = np.zeros_like(X)
        B, C, H_out, W_out = grad_output.shape
        stride_h, stride_w = strides

        # Compute gradient indices
        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        # Compute base position
                        h_start = i * stride_h
                        w_start = j * stride_w

                        # Compute local max index
                        local_max_idx = max_indices[b, c, i * W_out + j]
                        h_offset = local_max_idx // self.kernel_size[1]
                        w_offset = local_max_idx % self.kernel_size[1]

                        # Accumulate gradient
                        grad_input[b, c, h_start + h_offset, w_start + w_offset] += grad_output[b, c, i, j]
        # end_time = time.time()
        # print(f"maxpool forward took {end_time - start_time:.4f}")
        return grad_input

class batchnorm1d(Layer):
    def __init__(self, dim, activation="none", initialize_type="zero", dropout=None):
        super().__init__()
        self.eps = 1e-5
        self.input_normalized = None
        self.input = None
        self.betanorm = 0.9
        self.mean = None
        self.var = None
        self.initialize_weights(dim, initialize_type)

    def initialize_weights(self, dims, initialize_type):
        self.running_mean = np.zeros((1, dims))
        self.running_variance = np.ones((1, dims))
        self.weights = np.ones((1, dims))  # Initialize gamma to 1
        self.bias = np.zeros((1, dims))  # Initialize beta to 0
        self.momentum_w = np.zeros((1, dims))
        self.momentum_b = np.zeros((1, dims))
        self.Accumelated_Gsquare_w = np.zeros((1, dims))
        self.Accumelated_Gsquare_b = np.zeros((1, dims))

    def forward(self, input, test=False):
        self.input = input
        if not test:
            self.mean = np.mean(self.input, axis=0, keepdims=True)
            self.var = np.var(self.input, axis=0, keepdims=True)
            
            # Update running mean and variance using exponential moving average
            self.running_variance = (self.betanorm * self.running_variance) + ((1 - self.betanorm) * self.var)
            self.running_mean = (self.betanorm * self.running_mean) + ((1 - self.betanorm) * self.mean)
            
            # Normalize input
            self.input_normalized = (self.input - self.mean) / np.sqrt(self.var + self.eps)
            
            # Scale and shift with gamma and beta
            self.out = self.weights * self.input_normalized + self.bias
        else:  # Use running statistics during inference
            self.input_normalized = (self.input - self.running_mean) / np.sqrt(self.running_variance + self.eps)
            self.out = self.weights * self.input_normalized + self.bias
        
        return self.out

    def backward(self, error_wrt_output,l1,l2):
        batch_size = error_wrt_output.shape[0]

        # Gradient with respect to gamma (weights) and beta (bias)
        normalized_input_grad = error_wrt_output * self.weights  # Gradients wrt γ (scale)
        variance_grad = np.sum(normalized_input_grad * (self.input - self.mean) * (-0.5) * np.power((self.var + self.eps), -1.5), axis=0, keepdims=True)
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
        


    
    
class Network:
    def __init__(self, Layers,classification=False):
        self.layers = Layers
        self.classification = classification
        self.losses = []
        self.train_loss = 0
        self.val_loss = 0
        self.learning_rate=0.001
        self.beta1=0.9
        self.beta2=0.999
        self.EMA=False
        self.clip_value=10
        self.l1=None
        self.l2=None
        self.t=0
        self.featuremaps=[]


    def forward(self, X, test=False):
        for layer in self.layers:
            X = layer.forward(X, test)
            if isinstance(layer,Conv2d) and test==True:
                self.featuremaps.append(X)
        self.last_out = X


    def backward(self, error_grad):
        self.t+=1
        for layer in reversed(self.layers):
            error_grad = layer.backward(error_grad,l1=self.l1,l2=self.l2)
            # print("error_grad",error_grad)
            if isinstance(layer,Layer):
                self.optimizer_step(layer)


    def optimizer_step(self,layer):
        if not isinstance(layer,Layer):
            raise ValueError("layer must be an instance of Layer class")
        
        self.optimizer(layer)



    def one_hot_encode(labels, num_classes=None):
        if labels.ndim == 2:
        # Check if all rows have exactly one `1` and the rest `0`
            is_one_hot = np.all(np.isin(labels, [0, 1])) and np.all(labels.sum(axis=1) == 1)
        if num_classes is not None:
            is_one_hot = is_one_hot and (labels.shape[1] == num_classes)
        if is_one_hot:
            return labels.astype(int)  # Ensure integer type
    
        # Proceed to encode if not one-hot
        if num_classes is None:
            num_classes = np.max(labels) + 1  # Infer from integer labels
        return np.eye(num_classes, dtype=int)[labels]
    @timing_decorator
    def train(
        self, x_train, y_train, epochs=10, batch_size=32, learning_rate=0.001, 
        classification=True, verbose=True, L1=0, L2=0, optimizer="momentum", 
        beta1=0.9, beta2=0.999, EMA=False, clip_value=10, shuffle=True
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.EMA = EMA
        self.clip_value = clip_value
        self.l1 = L1
        self.l2 = L2
        self.optimizer = opt.Optimizer.get_optimizer(optimizer)
        
        # Initialize losses for tracking
        self.losses = []  # Store average loss per epoch
        if isinstance(self.layers[0], Conv2d) and x_train.ndim == 2:
                side = int(np.sqrt(x_train.shape[1]))
                x_train = x_train.reshape(-1, 1, side, side)
                y_train = self.one_hot_encode(y_train, num_classes=y_train.max() + 1)
        # Create Dataset instance
        dataset = Dataset(x_train, y_train, batch_size, shuffle=shuffle)

        for epoch in range(1, epochs + 1):
            # Reset dataset (shuffle indices if shuffle=True)
            dataset.reset()
            epoch_loss = 0.0
            batches_len = 0

            # Iterate over batches using the Dataset
            # cnt=0
            for X_batch, y_batch in dataset:
                # if cnt==1:
                #     raise ValueError("one iteration")
                # cnt+=1

                # Forward pass
                self.forward(X_batch, test=False)
                
                # Compute loss and gradient
                batch_loss, error_grad = ls.sparse_categorical_cross_entropy(y_batch, self.last_out, axis=1)
                epoch_loss += batch_loss
                batches_len += 1

                # Backward pass and parameter update
                self.backward(error_grad)

            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / batches_len
            self.losses.append(avg_epoch_loss)

            # Verbose logging
            if verbose:
                percent = (epoch / epochs) * 100
                bar = '||' * int(percent // 2) + '-' * (50 - int(percent // 2))
                print(f'\rEpoch {epoch}/{epochs} | [{bar}] {percent:.2f}% | Train Loss = {avg_epoch_loss:.4f}', end='')

        print()



    def plot_loss(self):

        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.ylim(0)
        plt.show()
    def predict(self, X):
        """
        RETURNS: LOGITS (BATCH_SIZE,NUM_CLASSES)
        
        """
        self.forward(X, test=True)
        out = self.last_out
        return out