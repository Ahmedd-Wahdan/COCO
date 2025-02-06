import numpy as np


def get_optimizer(name, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
    """
    Factory function to create an optimizer instance based on its name.

    Args:
        name (str): Name of the optimizer (e.g., "sgd", "momentum", "nag", "adam", "nadam", "adagrad", "rmsprop").
        learning_rate (float): Learning rate for the optimizer.
        beta1 (float): First momentum decay factor.
        beta2 (float): Second momentum decay factor (used by some optimizers).
        EMA (bool): Flag indicating whether to use Exponential Moving Average for momentum.
        clip_value (float): Maximum allowed absolute value for gradients (gradient clipping).

    Returns:
        Optimizer: An instance of the specified optimizer.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    if name == "sgd":
        return sgd(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "momentum":
        return momentum(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "nag":
        return nag(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "adam":
        return adam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "nadam":
        return NAdam(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "adagrad":
        return AdaGrad(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    elif name == "rmsprop":
        return RMSprop(learning_rate=learning_rate, beta1=beta1, beta2=beta2, EMA=EMA, clip_value=clip_value)
    else:
        raise ValueError("Unknown optimizer: " + name)


class Optimizer:
    """
    Base class for optimizers.

    This class defines the interface and holds common attributes for all optimizers.

    Attributes:
        learning_rate (float): Step size for parameter updates.
        beta1 (float): Decay factor for momentum.
        beta2 (float): Decay factor for second moment estimates.
        EMA (bool): Whether to use Exponential Moving Average for momentum.
        clip_value (float): Maximum allowed absolute value for gradients.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.EMA = EMA
        self.clip_value = clip_value

    def __call__(self, layer, **kwargs):
        """
        Placeholder for the optimizer update logic.
        Subclasses must implement this method.

        Args:
            layer: Layer object containing parameters (weights, biases) and their gradients.
            **kwargs: Additional parameters for the update.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class momentum(Optimizer):
    """
    Momentum optimizer for gradient descent.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)

    def __call__(self, layer, **kwargs):
        """
        Perform a momentum-based parameter update.

        Args:
            layer: Layer object with attributes:
                - weights, bias: Parameters.
                - grad_w, grad_b: Gradients.
                - momentum_w, momentum_b: Previous momentum values.
            **kwargs: Optional overrides for learning_rate, beta1, etc.
        """
        # Apply gradient clipping if needed.
        if self.clip_value:
            layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value)

        if self.EMA:
            layer.momentum_w = self.beta1 * layer.momentum_w + (1 - self.beta1) * layer.grad_w
            layer.momentum_b = self.beta1 * layer.momentum_b + (1 - self.beta1) * layer.grad_b
            layer.weights -= self.learning_rate * layer.momentum_w
            layer.bias -= self.learning_rate * layer.momentum_b
        else:
            layer.momentum_w = self.beta1 * layer.momentum_w + self.learning_rate * layer.grad_w
            layer.momentum_b = self.beta1 * layer.momentum_b + self.learning_rate * layer.grad_b
            layer.weights -= layer.momentum_w
            layer.bias -= layer.momentum_b


class nag(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG) optimizer.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)

    def __call__(self, layer, **kwargs):
        """
        Perform a parameter update using Nesterov Accelerated Gradient.

        Args:
            layer: Layer object with attributes:
                - weights, bias, grad_w, grad_b, momentum_w, momentum_b.
            **kwargs: Optional overrides.
        """
        if self.clip_value:
            layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value)

        if self.EMA:
            layer.momentum_w = self.beta1 * layer.momentum_w + (1 - self.beta1) * layer.grad_w
            layer.momentum_b = self.beta1 * layer.momentum_b + (1 - self.beta1) * layer.grad_b
        else:
            layer.momentum_w = self.beta1 * layer.momentum_w + self.learning_rate * layer.grad_w
            layer.momentum_b = self.beta1 * layer.momentum_b + self.learning_rate * layer.grad_b

        # Standard update.
        layer.weights -= layer.momentum_w
        layer.bias -= layer.momentum_b

        # Lookahead step.
        layer.weights -= self.beta1 * layer.momentum_w
        layer.bias -= self.beta1 * layer.momentum_b


class sgd(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)

    def __call__(self, layer, **kwargs):
        """
        Perform a parameter update using SGD.

        Args:
            layer: Layer object with attributes:
                - weights, bias, grad_w, grad_b.
            **kwargs: Optional overrides.

        Raises:
            ValueError: If the weight gradients are all zero.
        """
        if np.all(layer.grad_w == 0):
            raise ValueError("Gradient for weights is zero, cannot perform update.")

        if self.clip_value:
            layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value)

        layer.weights -= self.learning_rate * layer.grad_w
        layer.bias -= self.learning_rate * layer.grad_b


class adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)
        self.eps = 1e-8  # Small epsilon to avoid division by zero

    def __call__(self, layer, **kwargs):
        """
        Perform a parameter update using Adam.

        Args:
            layer: Layer object with attributes:
                - weights, bias, grad_w, grad_b,
                  momentum_w, momentum_b, Accumelated_Gsquare_w, Accumelated_Gsquare_b, and t (time step).
            **kwargs: Optional overrides.
        """
        if self.clip_value:
            layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value)

        # Update time step.
        layer.t += 1

        # Update biased first moment estimate.
        layer.momentum_w = self.beta1 * layer.momentum_w + (1 - self.beta1) * layer.grad_w
        layer.momentum_b = self.beta1 * layer.momentum_b + (1 - self.beta1) * layer.grad_b

        # Update biased second raw moment estimate.
        layer.Accumelated_Gsquare_w = self.beta2 * layer.Accumelated_Gsquare_w + (1 - self.beta2) * (layer.grad_w ** 2)
        layer.Accumelated_Gsquare_b = self.beta2 * layer.Accumelated_Gsquare_b + (1 - self.beta2) * (layer.grad_b ** 2)

        # Compute bias-corrected estimates.
        vw_corrected = layer.momentum_w / (1 - self.beta1 ** layer.t)
        vb_corrected = layer.momentum_b / (1 - self.beta1 ** layer.t)
        Gw_corrected = layer.Accumelated_Gsquare_w / (1 - self.beta2 ** layer.t)
        Gb_corrected = layer.Accumelated_Gsquare_b / (1 - self.beta2 ** layer.t)

        ita_w = self.learning_rate / (np.sqrt(Gw_corrected) + self.eps)
        ita_b = self.learning_rate / (np.sqrt(Gb_corrected) + self.eps)

        layer.weights -= ita_w * vw_corrected
        layer.bias -= ita_b * vb_corrected


class NAdam(Optimizer):
    """
    Nesterov-accelerated Adaptive Moment Estimation (NAdam) optimizer.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)
        self.eps = 1e-8

    def __call__(self, layer, **kwargs):
        """
        Perform a parameter update using NAdam.

        Args:
            layer: Layer object with attributes:
                - weights, bias, grad_w, grad_b,
                  momentum_w, momentum_b, Accumelated_Gsquare_w, Accumelated_Gsquare_b, t, and eps.
            **kwargs: Optional overrides.
        """
        if self.clip_value:
            layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value)

        layer.t += 1
        # Update biased first moment estimates.
        layer.momentum_w = self.beta1 * layer.momentum_w + (1 - self.beta1) * layer.grad_w
        layer.momentum_b = self.beta1 * layer.momentum_b + (1 - self.beta1) * layer.grad_b

        # Update biased second moment estimates.
        layer.Accumelated_Gsquare_w = self.beta2 * layer.Accumelated_Gsquare_w + (1 - self.beta2) * (layer.grad_w ** 2)
        layer.Accumelated_Gsquare_b = self.beta2 * layer.Accumelated_Gsquare_b + (1 - self.beta2) * (layer.grad_b ** 2)

        # Bias corrections.
        Gw_corrected = layer.Accumelated_Gsquare_w / (1 - np.power(self.beta2, layer.t))
        Gb_corrected = layer.Accumelated_Gsquare_b / (1 - np.power(self.beta2, layer.t))
        vw_corrected = layer.momentum_w / (1 - np.power(self.beta1, layer.t))
        vb_corrected = layer.momentum_b / (1 - np.power(self.beta1, layer.t))

        ita_w = self.learning_rate / (np.sqrt(Gw_corrected) + layer.eps)
        ita_b = self.learning_rate / (np.sqrt(Gb_corrected) + layer.eps)

        layer.weights -= ita_w * vw_corrected
        layer.bias -= ita_b * vb_corrected

        #TODO: Lookahead step need to be checked again (poor implementation)

        # Lookahead step.
        layer.weights -= self.beta1 * layer.momentum_w
        layer.bias -= self.beta1 * layer.momentum_b


class AdaGrad(Optimizer):
    """
    AdaGrad optimizer.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)
        # Note: AdaGrad typically does not use beta parameters.

    def __call__(self, layer, **kwargs):
        """
        Perform a parameter update using AdaGrad.

        Args:
            layer: Layer object with attributes:
                - weights, bias, grad_w, grad_b,
                  accumelated_Gsquare_w, accumelated_Gsquare_b, and eps.
            **kwargs: Optional overrides.
        """
        if self.clip_value:
            layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value)

        # Accumulate squared gradients.
        layer.accumelated_Gsquare_w += layer.grad_w ** 2
        layer.accumelated_Gsquare_b += layer.grad_b ** 2

        ita_w = self.learning_rate / (np.sqrt(layer.accumelated_Gsquare_w) + layer.eps)
        ita_b = self.learning_rate / (np.sqrt(layer.accumelated_Gsquare_b) + layer.eps)

        layer.weights -= ita_w * layer.grad_w
        layer.bias -= ita_b * layer.grad_b


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)
        self.eps = 1e-8

    def __call__(self, layer, **kwargs):
        """
        Perform a parameter update using RMSprop.

        Args:
            layer: Layer object with attributes:
                - weights, bias, grad_w, grad_b,
                  accumelated_Gsquare_w, accumelated_Gsquare_b, and eps.
            **kwargs: Optional overrides.
        """
        if self.clip_value:
            layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value)

        layer.accumelated_Gsquare_w = self.beta2 * layer.accumela




#TODO: ADAMW optimizers

