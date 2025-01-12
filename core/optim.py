import numpy as np

class Optimizer:
    """
    Base class for optimizers in the framework. Provides a static method to fetch specific optimizer instances.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_optimizer(name):
        """
        Factory function to create an optimizer instance based on the name.

        Args:
            name (str): Name of the optimizer (e.g., "sgd", "momentum", "adam").

        Returns:
            Optimizer: An instance of the specified optimizer.

        Raises:
            ValueError: If the optimizer name is not recognized.
        """
        if name == "sgd":
            return sgd()
        elif name == "momentum":
            return momentum()
        elif name == "nag":
            return nag()
        elif name == "adam":
            return adam()
        elif name == "nadam":
            return NAdam()
        elif name == "adagrad":
            return AdaGrad()
        elif name == "rmsprop":
            return RMSprop()
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def __call__(self, grad_w, grad_b, learning_rate, beta1=0.9, beta2=0.999):
        """
        Placeholder for the optimizer update logic. Subclasses must implement this method.
        """
        pass


class momentum(Optimizer):
    """
    Momentum optimizer for gradient descent.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, layer, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        """
        Perform a momentum-based parameter update.

        Args:
            layer: Layer object containing weights, biases, and gradients.
            learning_rate (float): Step size for the update.
            beta1 (float): Momentum decay factor.
            beta2 (float): Not used in this optimizer.
            EMA (bool): Use Exponential Moving Average for momentum.
            clip_value (float): Maximum allowed value for gradients (gradient clipping).
        """
        layer.grad_w = np.clip(layer.grad_w, -clip_value, clip_value) if clip_value else layer.grad_w
        if EMA:
            layer.momentum_w = beta1 * layer.momentum_w + (1 - beta1) * layer.grad_w
            layer.momentum_b = beta1 * layer.momentum_b + (1 - beta1) * layer.grad_b
        else:
            layer.momentum_w = beta1 * layer.momentum_w + learning_rate * layer.grad_w
            layer.momentum_b = beta1 * layer.momentum_b + learning_rate * layer.grad_b
        layer.weights -= layer.momentum_w
        layer.bias -= layer.momentum_b


class nag(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG) optimizer.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, layer, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        """
        Perform a parameter update using Nesterov Accelerated Gradient.

        Args:
            layer: Layer object containing weights, biases, and gradients.
            learning_rate (float): Step size for the update.
            beta1 (float): Momentum decay factor.
            beta2 (float): Not used in this optimizer.
            EMA (bool): Use Exponential Moving Average for momentum.
            clip_value (float): Maximum allowed value for gradients (gradient clipping).
        """
        layer.grad_w = np.clip(layer.grad_w, -clip_value, clip_value) if clip_value else layer.grad_w
        if EMA:
            layer.momentum_w = beta1 * layer.momentum_w + (1 - beta1) * layer.grad_w
            layer.momentum_b = beta1 * layer.momentum_b + (1 - beta1) * layer.grad_b
        else:
            layer.momentum_w = beta1 * layer.momentum_w + learning_rate * layer.grad_w
            layer.momentum_b = beta1 * layer.momentum_b + learning_rate * layer.grad_b
        layer.weights -= layer.momentum_w
        layer.bias -= layer.momentum_b

        # Lookahead step
        layer.weights -= (beta1 * layer.momentum_w)
        layer.bias -= (beta1 * layer.momentum_b)


class sgd(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, layer, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        """
        Perform a parameter update using SGD.

        Args:
            layer: Layer object containing weights, biases, and gradients.
            learning_rate (float): Step size for the update.
            clip_value (float): Maximum allowed value for gradients (gradient clipping).
        """
        layer.grad_w = np.clip(layer.grad_w, -clip_value, clip_value) if clip_value else layer.grad_w
        layer.weights -= learning_rate * layer.grad_w
        layer.bias -= learning_rate * layer.grad_b


class adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.t = 0  # Time step
        self.eps = eps  # Small epsilon to avoid division by zero

    def __call__(self, layer, learning_rate=1e-3, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        """
        Perform a parameter update using Adam.

        Args:
            layer: Layer object containing weights, biases, and gradients.
            learning_rate (float): Step size for the update.
            beta1 (float): Decay rate for first moment estimates.
            beta2 (float): Decay rate for second moment estimates.
            clip_value (float): Maximum allowed value for gradients (gradient clipping).
        """
        if clip_value:
            np.clip(layer.grad_w, -clip_value, clip_value, out=layer.grad_w)

        layer.t += 1
        layer.momentum_w = beta1 * layer.momentum_w + (1 - beta1) * layer.grad_w
        layer.momentum_b = beta1 * layer.momentum_b + (1 - beta1) * layer.grad_b

        layer.Accumelated_Gsquare_w = beta2 * layer.Accumelated_Gsquare_w + (1 - beta2) * layer.grad_w**2
        layer.Accumelated_Gsquare_b = beta2 * layer.Accumelated_Gsquare_b + (1 - beta2) * layer.grad_b**2

        vw_corrected = layer.momentum_w / (1 - beta1**layer.t)
        vb_corrected = layer.momentum_b / (1 - beta1**layer.t)
        Gw_corrected = layer.Accumelated_Gsquare_w / (1 - beta2**layer.t)
        Gb_corrected = layer.Accumelated_Gsquare_b / (1 - beta2**layer.t)

        ita_w = learning_rate / (np.sqrt(Gw_corrected) + self.eps)
        ita_b = learning_rate / (np.sqrt(Gb_corrected) + self.eps)

        layer.weights -= ita_w * vw_corrected
        layer.bias -= ita_b * vb_corrected




class NAdam(Optimizer):
    def __init__(self):
        super().__init__()

    def __call__(self,layer, learning_rate:float=0.001,beta1:float=0.9,beta2:float=0.999,EMA:bool=False,clip_value:float=10)->None:
        if clip_value:
            np.clip(layer.grad_w, -clip_value, clip_value, out=layer.grad_w)
        
        layer.t+=1        
        layer.momentum_w = beta1 * layer.momentum_w + (1 - beta1) * layer.grad_w  # Momentum with exponential decaying moving average
        layer.momentum_b = beta1 * layer.momentum_b + (1 - beta1) * layer.grad_b  

        layer.Accumelated_Gsquare_w = (beta2 * layer.Accumelated_Gsquare_w) + ((1 - beta2) * layer.grad_w**2)  # accumulation of squared gradient with exponential decaying moving average
        layer.Accumelated_Gsquare_b = (beta2* layer.Accumelated_Gsquare_b) + ((1 - beta2) * layer.grad_b**2)

        Gw_corrected = layer.Accumelated_Gsquare_w/((1-np.power(beta2, layer.t)))
        Gb_corrected = layer.Accumelated_Gsquare_b/((1-np.power(beta2, layer.t)))

        vw_corrected = layer.momentum_w/((1-np.power(beta1, layer.t)))   #bias correction  (1 - np.power(beta1, layer.t)) becomes closer to 1 as t increases
        vb_corrected = layer.momentum_b/((1-np.power(beta1, layer.t)))


        ita_w = learning_rate/(np.sqrt(Gw_corrected)+layer.eps)
        ita_b = learning_rate/(np.sqrt(Gb_corrected)+layer.eps)


        layer.weights -= ita_w*vw_corrected
        layer.bias -= ita_b*vb_corrected


        layer.weights -= (beta1 * layer.momentum_w)          # look ahead step (make the momentum step first then adjust)
        layer.bias -= (beta1 * layer.momentum_b)

        


class AdaGrad(Optimizer):
    def __init__(self):
        super().__init__()

    def __call__(self,layer, learning_rate:float=0.001,beta1:float=0.9,beta2:float=0.999,EMA:bool=False,clip_value:float=10)->None:
        layer.grad_w = np.clip(layer.grad_w, -10, 10)
        layer.grad_b = np.clip(layer.grad_b, -10, 10)

        layer.accumelated_Gsquare_w+=layer.grad_w**2
        layer.accumelated_Gsquare_b+=layer.grad_b**2

        ita_w = learning_rate/(np.sqrt(layer.accumelated_Gsquare_w)+layer.eps)
        ita_b = learning_rate/(np.sqrt(layer.accumelated_Gsquare_b)+layer.eps)


        layer.weights -= ita_w*layer.grad_w
        layer.bias -= ita_b*layer.grad_b

       


class RMSprop(Optimizer):
    def __init__(self):
        super().__init__()

    def __call__(self,layer, learning_rate:float=0.001,beta1:float=0.9,beta2:float=0.999,EMA:bool=False,clip_value:float=10)->None:
        layer.grad_w = np.clip(layer.grad_w, -clip_value, clip_value) if clip_value else layer.grad_w

        layer.accumelated_Gsquare_w = (beta2 * layer.accumelated_Gsquare_w) + ((1 - beta2) * layer.grad_w**2)
        layer.accumelated_Gsquare_b = (beta2 * layer.accumelated_Gsquare_b) + ((1 - beta2) * layer.grad_b**2)

        ita_w = learning_rate/(np.sqrt(layer.accumelated_Gsquare_w)+self.eps)
        ita_b = learning_rate/(np.sqrt(layer.accumelated_Gsquare_b)+self.eps)


        weights -= ita_w*layer.grad_w
        bias -= ita_b*layer.grad_b

    