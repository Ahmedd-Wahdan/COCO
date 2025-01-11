
import numpy as np
import matplotlib.pyplot as plt
from core.wrapper import timing_decorator
import core.Function as fn
import core.optim as opt
import core.loss as ls
class Layer:
    def __init__(self,activation=None, initialize_type="he"):
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
        """
        Forward pass through the linear layer.

        Args:
            input (ndarray): Input of shape (batch_size, input_dim).
            test (bool, optional): Whether to use dropout during testing. Defaults to False.

        Returns:
            ndarray: Output of shape (batch_size, output_dim).
        """
        self.input = input
        self.z = np.dot(self.input, self.weights) + self.bias  # Linear transformation
        self.out = self.activation.forward(self.z)  # Apply activation

        # Apply dropout during training
        if self.dropout is not None and not test:
            self.mask = (np.random.rand(*self.out.shape) < (1 - self.dropout)).astype(float)
            self.out *= self.mask
            self.out /= (1 - self.dropout)

        # print("output_of_forward",self.out.shape)
        return self.out

    def backward(self, error_wrt_output,l1=None,l2=None):
        """
        Backward pass through the linear layer.

        Args:
            error_wrt_output (ndarray): Gradient of the loss w.r.t. the output of shape (batch_size, output_dim).

        Returns:
            ndarray: Gradient of the loss w.r.t. the input of shape (batch_size, input_dim).
        """
        # Gradient of activation
        # self.grad_w = np.zeros_like(self.weights)  # zero gradients before accumulating
        # self.grad_b = np.zeros_like(self.bias)



        da_dz = self.activation.backward(self.z)
        batch_size = error_wrt_output.shape[0]
        # Gradient of loss w.r.t. pre-activation (z)
        d_z = error_wrt_output * da_dz

        # Gradient of loss w.r.t. weights
        self.grad_w = np.dot(self.input.T, d_z)/batch_size
        if l1 is not None:
            self.grad_w += (l1*np.sign(self.weights))

        if l2 is not None:
            self.grad_w += (l2*2*self.weights)

        # Gradient of loss w.r.t. bias
        self.grad_b = np.sum(d_z, axis=0, keepdims=True)/batch_size

        # print("grad_w",self.grad_w)
        # Gradient of loss w.r.t. input
        self.loss_wrt_input = np.dot(d_z, self.weights.T)

        assert self.grad_w.shape == self.weights.shape
        assert self.grad_b.shape == self.bias.shape
        assert self.loss_wrt_input.shape == self.input.shape

        # print("loss_wrt_input",self.loss_wrt_input.shape)


        """
        
        UPDATE THE GRADIENTS HERE TO AVOID A SECOND LOOP IN THE TRAINING FUNCTION
        
        """

        return self.loss_wrt_input



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

        # Ensure gradients are correctly computed
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


    def forward(self, X, test=False):
        for layer in self.layers:
            X = layer.forward(X, test)
        self.last_out = X


    def backward(self, error_grad):
        self.t+=1
        for layer in reversed(self.layers):
            # print("error_grad",error_grad)
            error_grad = layer.backward(error_grad,l1=self.l1,l2=self.l2)
            self.optimizer_step(layer)


    def optimizer_step(self,layer):
        if not isinstance(layer,Layer):
            raise ValueError("layer must be an instance of Layer class")
        
        self.optimizer(layer)



    def one_hot_encode(labels, num_classes=None):
        """
        Convert integer labels to one-hot encoded labels.

        Args:
            labels (np.ndarray): Integer labels of shape (N,), where N is the number of samples.
            num_classes (int, optional): Number of classes. If None, it is inferred from the labels.

        Returns:
            np.ndarray: One-hot encoded labels of shape (N, num_classes).
        """
        if num_classes is None:
            num_classes = np.max(labels) + 1  # Infer number of classes from labels
        return np.eye(num_classes)[labels]
    @timing_decorator
    def train(self, x_train, y_train, epochs=10, batch_size=32, learning_rate=0.001, classification=True, verbose=True, L1=0, L2=0, optimizer="momentum", beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        """
        This function performs training for the neural network:
        - Forward pass
        - Calculate loss
        - Backward pass and update gradients
        """
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

        for epoch in range(1, epochs + 1):
            # Reset epoch metrics
            epoch_loss = 0.0  # Accumulate loss for all batches in the epoch
            batches_len = 0   # Track the number of batches

            # Shuffle the data
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)
            x = x_train[indices, :]
            y = y_train[indices, :]

            # Iterate over batches
            for i in range(0, x.shape[0], batch_size):
                X_batch = x[i:i + batch_size, :]
                y_batch = y[i:i + batch_size, :]
              
                self.forward(X_batch, test=False)

                batch_loss, error_grad = ls.sparse_categorical_cross_entropy(y_batch, self.last_out, axis=1)
                
                epoch_loss += batch_loss
                
                batches_len += 1

                # Backward pass and gradient update
                self.backward(error_grad)

            # Calculate average loss for the epoch
            avg_epoch_loss = epoch_loss / batches_len
            self.losses.append(avg_epoch_loss)  # Store average epoch loss

            # Verbose logging
            if verbose:
                percent = (epoch / epochs) * 100
                bar = '||' * int(percent // 2) + '-' * (50 - int(percent // 2))
                print(f'\rEpoch {epoch}/{epochs} | [{bar}] {percent:.2f}% | Train Loss = {avg_epoch_loss:.4f}', end='')

        print()



    def plot_loss(self):
        """
        Plots the loss curve of the neural network model.

        Parameters:
            None

        Returns:
            None
        """
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.ylim(0)
        plt.show()
    def predict(self, X):
        """
        Predicts the output for a given input using the trained neural network model.

        Parameters:
            X (ndarray): The input data for prediction. It should be a 2D array where each row represents a sample.

        Returns:
            If the model is a binary classification model:
                - 1 or 0, depending on the predicted class for the corresponding sample in X.
            If the model is a multi-class classification model:
                - the class number
            If the model is a regression model:
                - scalar continuous value
        """
        self.forward(X, test=True)
        out = self.last_out
        return out