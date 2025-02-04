from core.nn import *
class Model:
    def __init__(self):
        """
        Initialize the model.

        Attributes:
            layers (dict): Dictionary to store layers with their variable names.
            losses (list): Store training losses.
            last_out (ndarray or None): Store the last output for loss computation.
        """
        self.layers = {}  # Dictionary to store layers with their variable names
        self.losses = []  # Store training losses
        self.last_out = None  # Store the last output for loss computation

    def __setattr__(self, name, value):
        """
        Override the __setattr__ method to track layers.
        """
        super().__setattr__(name, value)
        if isinstance(value, Layer) or isinstance(value, Module) or isinstance(value, fn.Activation):
            self.layers[name] = value  # Add it to the layers dictionary with its variable name


    def to_train(self):
        """
        Set the model to training mode. This will propagate the training flag to all layers
        and enable the computation of gradients. Note that this method should be called
        before training the model.

        """
        self.train_mode = True
        self.update_train_mode()

    def to_eval(self):
        """
        Set the model to evaluation mode. This will propagate the evaluation flag to all layers
        and disable the computation of gradients. Note that this method should be called
        before evaluating the model.

        """
        self.train_mode = False
        self.update_train_mode()
    def update_train_mode(self):
        for layer in self.layers.values():
            if isinstance(layer, fn.Activation):
                if self.train_mode:
                    layer.to_train()
                else:
                    layer.to_eval()

    def forward(self, x):
        """
        Users must override this method to define the forward pass.
        """
        raise NotImplementedError("Forward method must be implemented in subclasses")

    def backward(self, error_grad, l1=None, l2=None):
        """
        Perform the backward pass through all layers.
        """
        for layer_name, layer in reversed(self.layers.items()):
            
            error_grad = layer.backward(error_grad, l1=l1, l2=l2)
            if isinstance(layer, Layer):
                self.optimizer_step(layer)

    def optimizer_step(self, layer):
        """
        Perform a single optimization step for a layer.
        """
        if not isinstance(layer, Layer):
            raise ValueError("layer must be an instance of Layer class")
        self.optimizer(layer)

    @timing_decorator
    def train(self, x_train, y_train, epochs=10, batch_size=32, learning_rate=0.001, 
              verbose=True, L1=0, L2=0, optimizer="momentum", loss="categorical_bce", 
              beta1=0.9, beta2=0.999, EMA=False, clip_value=10, shuffle=True):
        """
        Default training loop. Users can override this method for custom training behavior.
        """
        # Set optimizer and loss function
        self.optimizer = opt.get_optimizer(
        optimizer, 
        learning_rate=learning_rate, 
        beta1=beta1, 
        beta2=beta2, 
        EMA=EMA, 
        clip_value=clip_value
        )
        
        self.loss_fn = ls.get_loss_fn(loss)

        # Ensure input format
        if isinstance(next(iter(self.layers.values())), Conv2d) and x_train.ndim == 2:
            side = int(np.sqrt(x_train.shape[1]))
            x_train = x_train.reshape(-1, 1, side, side)
            y_train = self.one_hot_encode(y_train, num_classes=y_train.max() + 1)

        # Create Dataset instance
        dataset = Dataset(x_train, y_train, batch_size, shuffle=shuffle)

        for epoch in range(1, epochs + 1):
            dataset.reset()  # Shuffle if needed
            epoch_loss = 0.0
            total_samples = 0  # Track total processed samples

            for X_batch, y_batch in dataset:
                batch_size_actual = len(y_batch)  # May be smaller in last batch

                # Forward pass
                self.last_out = self.forward(X_batch)

                # Compute loss and gradient
                batch_loss, error_grad = self.loss_fn(y_batch, self.last_out, axis=1)

                # Backward pass and parameter update
                self.backward(error_grad, l1=L1, l2=L2)

                # Scale loss
                epoch_loss += batch_loss * batch_size_actual
                total_samples += batch_size_actual

            # Compute average epoch loss (like PyTorch)
            avg_epoch_loss = epoch_loss / total_samples
            self.losses.append(avg_epoch_loss)

            # Verbose logging
            if verbose:
                percent = (epoch / epochs) * 100
                print(f'\rEpoch {epoch}/{epochs} | Epoch Loss = {avg_epoch_loss:.4f}')

        print()

    def __call__(self, x):
        """
        ARGS: 
            x (ndarray): Input data if image then (B, C, H, W) else (B, D)
        RETURNS:
            ndarray: Output probabilities (B,num_output neurons)
        """
        self.last_out = self.forward(x)
        return self.last_out

    def plot_loss(self):
        """
        Plot the training loss curve.
        """
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.ylim(0)
        plt.show()

    def visualize_feature_maps(self, image):
        """
        Visualize feature maps for convolutional layers.
        """
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, :, :]
        elif image.ndim == 3:
            image = image[np.newaxis, :, :, :]
        elif image.ndim == 4 and image.shape[0] != 1:
            raise ValueError("Only single image batches are supported.")

        self.featuremaps = []
        self.forward(image, test=True, visualize=True)

        num_conv_layers = len(self.featuremaps)
        if num_conv_layers == 0:
            print("No convolutional layers found.")
            return

        # Create a separate figure for each convolutional layer
        for layer_idx, fm in enumerate(self.featuremaps):
            fm = fm[0]  # Remove batch dimension -> (C, H, W)
            num_channels = fm.shape[0]

            # Create a new figure for this layer
            plt.figure(figsize=(16, 8))
            plt.suptitle(f"Layer {layer_idx+1} Feature Maps", fontsize=14, y=1.02)

            # Calculate grid dimensions
            cols = 8  # Max 8 filters per row
            rows = int(np.ceil(num_channels / cols))

            # Plot each channel
            for channel_idx in range(num_channels):
                plt.subplot(rows, cols, channel_idx + 1)
                channel_data = fm[channel_idx]

                # Normalize to [0, 1] for better contrast
                channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)

                plt.imshow(channel_data, cmap='gray')
                plt.axis('off')
                plt.title(f'Ch{channel_idx+1}', fontsize=8)

            plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)  # Increase spacing
            plt.show()  # Show layer-specific figure (will create multiple windows)

    @staticmethod
    def one_hot_encode(labels, num_classes=None):
        """
        Convert integer labels to one-hot encoded vectors.
        """
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
    

    def summary(self):
        """
        Print a summary of the model architecture.
        """
        print("Model Summary:")
        print("=" * 50)
        for name, layer in self.layers.items():
            print(f"{name}: {layer.__class__.__name__}")
            if hasattr(layer, 'weights'):
                print(f"  Weights: {layer.weights.shape}")
            if hasattr(layer, 'bias'):
                print(f"  Bias: {layer.bias.shape}")
        print("=" * 50)