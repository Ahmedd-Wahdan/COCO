import numpy as np

class Optimizer:
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


    def __call__(self,grad_w, grad_b, learning_rate,beta1=0.9,beta2=0.999):
        pass


class momentum(Optimizer):
    def __init__(self):
        super().__init__()

    
    def __call__(self,layer, learning_rate:float=0.001,beta1:float=0.9,beta2:float=0.999,EMA:bool=False,clip_value:float=10)->None:
        layer.grad_w = np.clip(layer.grad_w, -clip_value, clip_value) if clip_value else layer.grad_w
        if EMA:
            layer.momentum_w = beta1 * layer.momentum_w + (1 - beta1) * layer.grad_w  # exponential decaying moving average momentum (EMA)
            layer.momentum_b = beta1 * layer.momentum_b + (1 - beta1) * layer.grad_b
        else:
            layer.momentum_w = beta1 * layer.momentum_w + learning_rate * layer.grad_w  # polyak original momentum equation
            layer.momentum_b = beta1 * layer.momentum_b + learning_rate * layer.grad_b  
        layer.weights -= layer.momentum_w
        layer.biases -= layer.momentum_b


class nag(Optimizer):
    def __init__(self):
        super().__init__()

    def __call__(self,layer, learning_rate:float=0.001,beta1:float=0.9,beta2:float=0.999,EMA:bool=False,clip_value:float=10)->None:

        layer.grad_w = np.clip(layer.grad_w, -clip_value, clip_value) if clip_value else layer.grad_w
        if EMA:
            layer.momentum_w = beta1 * layer.momentum_w + (1 - beta1) * layer.grad_w  # exponential decaying moving average momentum (EMA)
            layer.momentum_b = beta1 * layer.momentum_b + (1 - beta1) * layer.grad_b
        else:
            layer.momentum_w = beta1 * layer.momentum_w + learning_rate * layer.grad_w  # polyak original momentum equation
            layer.momentum_b = beta1 * layer.momentum_b + learning_rate * layer.grad_b  
        layer.weights -= layer.momentum_w
        layer.biases -= layer.momentum_b
        
        layer.weights -= (beta1 * layer.momentum_w)          # look ahead step (make the momentum step first then adjust)
        layer.biases -= (beta1 * layer.momentum_b)

              

        

        
class sgd(Optimizer):
    def __init__(self):
        super().__init__()

    def __call__(self,layer, learning_rate:float=0.001,beta1:float=0.9,beta2:float=0.999,EMA:bool=False,clip_value:float=10)->None:

        # print("weights before sgd",weights)
        layer.grad_w = np.clip(layer.grad_w, -clip_value, clip_value) if clip_value else layer.grad_w
        # print("grad_w",grad_w)
        # print("lr",learning_rate)
        layer.weights -= learning_rate * layer.grad_w
        layer.biases -= learning_rate * layer.grad_b   

        # print("weights after sgd",weights)

       
    
class adam(Optimizer):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.t = 0  # Time step
        self.eps = eps  # Small epsilon to avoid division by zero

    def __call__(self,layer, learning_rate:float=1e-3,beta1:float=0.9,beta2:float=0.999,EMA:bool=False,clip_value:float=10)->None:
        # Gradient clipping
        if layer.grad_w is None or layer.grad_b is None:
            raise ValueError("Gradients (grad_w or grad_b) are None.")
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
        layer.biases -= (beta1 * layer.momentum_b)

        


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
        layer.biases -= ita_b*layer.grad_b

       


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
        biases -= ita_b*layer.grad_b

    