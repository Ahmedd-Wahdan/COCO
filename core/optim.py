import numpy as np

@staticmethod
def get_optimizer(name, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
    """
    Factory function to create an optimizer instance based on the name.

    Args:
        name (str): Name of the optimizer (e.g., "sgd", "momentum", "adam").
        learning_rate (float): Learning rate for the optimizer.
        beta1 (float): Momentum decay factor for optimizers like Adam.
        beta2 (float): Second momentum decay factor for optimizers like Adam.
        EMA (bool): Whether to use Exponential Moving Average for momentum.
        clip_value (float): Maximum allowed value for gradients (gradient clipping).

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
        raise ValueError(f"Unknown optimizer: {name}")
    
class Optimizer:
    """
    Base class for optimizers in the framework. Provides a static method to fetch specific optimizer instances.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.EMA = EMA
        self.clip_value = clip_value

    

    def __call__(self, grad_w, grad_b, learning_rate, beta1=0.9, beta2=0.999):
        """
        Placeholder for the optimizer update logic. Subclasses must implement this method.
        """
        pass


class momentum(Optimizer):
    """
    Momentum optimizer for gradient descent.
    """
    def __init__(self ,learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)

    def __call__(self, layer,**kwargs):
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
        layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value) if self.clip_value else layer.grad_w
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
    def __init__(self ,learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)
        

    def __call__(self, layer,**kwargs):
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
        layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value) if self.clip_value else layer.grad_w
        if self.EMA:
            layer.momentum_w = self.beta1 * layer.momentum_w + (1 - self.beta1) * layer.grad_w
            layer.momentum_b = self.beta1 * layer.momentum_b + (1 - self.beta1) * layer.grad_b
        else:
            layer.momentum_w = self.beta1 * layer.momentum_w + self.learning_rate * layer.grad_w
            layer.momentum_b = self.beta1 * layer.momentum_b + self.learning_rate * layer.grad_b
        layer.weights -= layer.momentum_w
        layer.bias -= layer.momentum_b

        # Lookahead step
        layer.weights -= (self.beta1 * layer.momentum_w)
        layer.bias -=    (self.beta1 * layer.momentum_b)


class sgd(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self ,learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)

    def __call__(self, layer, **kwargs):
        """
        Perform a parameter update using SGD.

        Args:
            layer: Layer object containing weights, biases, and gradients.
            learning_rate (float): Step size for the update.
            clip_value (float): Maximum allowed value for gradients (gradient clipping).
        """
        if np.all(layer.grad_w == 0):
            raise ValueError("Gradient is zero")
        layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value) if self.clip_value else layer.grad_w
        layer.weights -= self.learning_rate * layer.grad_w
        layer.bias -= self.learning_rate * layer.grad_b


class adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    """
    def __init__(self ,learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)
        self.eps = 1e-8  # Small epsilon to avoid division by zero

    def __call__(self, layer,**kwargs):
        """
        Perform a parameter update using Adam.

        Args:
            layer: Layer object containing weights, biases, and gradients.
            learning_rate (float): Step size for the update.
            beta1 (float): Decay rate for first moment estimates.
            beta2 (float): Decay rate for second moment estimates.
            clip_value (float): Maximum allowed value for gradients (gradient clipping).
        """
        layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value) if self.clip_value else layer.grad_w

        layer.t += 1
        layer.momentum_w = self.beta1 * layer.momentum_w + (1 - self.beta1) * layer.grad_w
        layer.momentum_b = self.beta1 * layer.momentum_b + (1 - self.beta1) * layer.grad_b

        layer.Accumelated_Gsquare_w = self.beta2 * layer.Accumelated_Gsquare_w + (1 - self.beta2) * layer.grad_w**2
        layer.Accumelated_Gsquare_b = self.beta2 * layer.Accumelated_Gsquare_b + (1 - self.beta2) * layer.grad_b**2

        vw_corrected = layer.momentum_w / (1 - self.beta1**layer.t)
        vb_corrected = layer.momentum_b / (1 - self.beta1**layer.t)
        Gw_corrected = layer.Accumelated_Gsquare_w / (1 - self.beta2**layer.t)
        Gb_corrected = layer.Accumelated_Gsquare_b / (1 - self.beta2**layer.t)

        ita_w = self.learning_rate / (np.sqrt(Gw_corrected) + self.eps)
        ita_b = self.learning_rate / (np.sqrt(Gb_corrected) + self.eps)

        layer.weights -= ita_w * vw_corrected
        layer.bias -= ita_b * vb_corrected




class NAdam(Optimizer):
    def __init__(self ,learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)

    def __call__(self,layer,**kwargs)->None:
        
        layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value) if self.clip_value else layer.grad_w
        
        layer.t+=1        
        layer.momentum_w = self.beta1 * layer.momentum_w + (1 - self.beta1) * layer.grad_w  # Momentum with exponential decaying moving average
        layer.momentum_b = self.beta1 * layer.momentum_b + (1 - self.beta1) * layer.grad_b  

        layer.Accumelated_Gsquare_w = (self.beta2 * layer.Accumelated_Gsquare_w) + ((1 - self.beta2) * layer.grad_w**2)  # accumulation of squared gradient with exponential decaying moving average
        layer.Accumelated_Gsquare_b = (self.beta2* layer.Accumelated_Gsquare_b) + ((1 - self.beta2) * layer.grad_b**2)

        Gw_corrected = layer.Accumelated_Gsquare_w/((1-np.power(self.beta2, layer.t)))
        Gb_corrected = layer.Accumelated_Gsquare_b/((1-np.power(self.beta2, layer.t)))

        vw_corrected = layer.momentum_w/((1-np.power(self.beta1, layer.t)))   #bias correction  (1 - np.power(beta1, layer.t)) becomes closer to 1 as t increases
        vb_corrected = layer.momentum_b/((1-np.power(self.beta1, layer.t)))


        ita_w = self.learning_rate/(np.sqrt(Gw_corrected)+layer.eps)
        ita_b = self.learning_rate/(np.sqrt(Gb_corrected)+layer.eps)


        layer.weights -= ita_w*vw_corrected
        layer.bias -= ita_b*vb_corrected


        layer.weights -= (self.beta1 * layer.momentum_w)          # look ahead step (make the momentum step first then adjust)
        layer.bias -= (self.beta1 * layer.momentum_b)

        


class AdaGrad(Optimizer):
    def __init__(self ,learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)

    def __call__(self,layer,**kwargs)->None:
        layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value) if self.clip_value else layer.grad_w
       

        layer.accumelated_Gsquare_w+=layer.grad_w**2
        layer.accumelated_Gsquare_b+=layer.grad_b**2

        ita_w = self.learning_rate/(np.sqrt(layer.accumelated_Gsquare_w)+layer.eps)
        ita_b = self.learning_rate/(np.sqrt(layer.accumelated_Gsquare_b)+layer.eps)


        layer.weights -= ita_w*layer.grad_w
        layer.bias -= ita_b*layer.grad_b

       


class RMSprop(Optimizer):
    def __init__(self ,learning_rate=0.001, beta1=0.9, beta2=0.999, EMA=False, clip_value=10):
        super().__init__(learning_rate, beta1, beta2, EMA, clip_value)

    def __call__(self,layer,**kwargs)->None:
        layer.grad_w = np.clip(layer.grad_w, -self.clip_value, self.clip_value) if self.clip_value else layer.grad_w

        layer.accumelated_Gsquare_w = (self.beta2 * layer.accumelated_Gsquare_w) + ((1 - self.beta2) * layer.grad_w**2)
        layer.accumelated_Gsquare_b = (self.beta2 * layer.accumelated_Gsquare_b) + ((1 - self.beta2) * layer.grad_b**2)

        ita_w = self.learning_rate/(np.sqrt(layer.accumelated_Gsquare_w)+self.eps)
        ita_b = self.learning_rate/(np.sqrt(layer.accumelated_Gsquare_b)+self.eps)


        layer.weights -= ita_w*layer.grad_w
        layer.bias -= ita_b*layer.grad_b

    