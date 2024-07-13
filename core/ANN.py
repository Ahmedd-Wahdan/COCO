
import numpy as np
import matplotlib.pyplot as plt
from core.wrapper import timing_decorator
class Layer:
    def __init__(self, m=1, n=1, activation="none", initialize_type="he",dropout=0):
        """
        Initializes the Layer with the specified parameters.

        Parameters:
            m (int): The number of rows in the weight matrix.
            n (int): The number of columns in the weight matrix.
            activation (str): The activation function name.
            initialize_type (str): The weight initialization type he is default.
            dropout (int): The dropout rate 0 is default.

        Returns:
            None
        """


        if m==1 and  n==1:
            pass
        else:
            self.w, self.b = self.initialize_weights(m, n,initialize_type)
            self.momentum_w = np.zeros((m, n))
            self.momentum_b = np.zeros((m, 1))
            self.Accumelated_Gsquare_w = np.zeros((m, n))
            self.Accumelated_Gsquare_b = np.zeros((m, 1))
        self.error = None
        self.activation = activation
        self.backprop_error = None
        self.is_last=False
        self.eps = 1e-8
        self.t=0
        self.dropout = dropout
        self.optimizers = {
            "sgd": self.SGD,
            "momentum": self.Momentum,
            "adam": self.Adam,
            "nag": self.NAG,
            "adagrad": self.AdaGrad,
            "rmsprop": self.RMSprop,
            "nadam": self.NAdam
        }
        self.activation_functions = {
            "sigmoid": (self.sigmoid, self.sigmoid_derivative),
            "softmax": (self.softmax, self.softmax_derivative),
            "relu": (self.relu, self.relu_derivative),
            "leaky_relu": (self.leaky_relu, self.leaky_relu_derivative),
            "none": (self.no_activation, self.no_activation_derivative),
            "tanh": (self.tanh, self.tanh_derivative)
        }

    def initialize_weights(self, m, n,initialize_type):
        """
        Initializes the weights of a neural network layer based on the specified initialization method.

        the difference between the initialization types is the distribution of the weights and the ranges
        (he)==> commonly used for relu and leaky relu
        (xavier)==> commonly used for tanh
        (lecun)==> commonly used for sigmoid


        Parameters:
            m (int): Number of input units.
            n (int): Number of output units.
            initialize_type (str): The type of weight initialization method to use.

        Returns:
            tuple: A tuple containing the initialized weights (w) and bias (b) for the layer.
        """
        if initialize_type == 'zero':
            w = np.zeros((m, n))
        elif initialize_type == 'random':
            w = np.random.randn(m, n) * 0.01
        elif initialize_type == 'xavier':
            limit = np.sqrt(6 / (m + n))
            w = np.random.uniform(-limit, limit, (m, n))
        elif initialize_type == 'he':
            w = np.random.randn(m, n) * np.sqrt(2. / n)
        elif initialize_type == 'lecun':
            w = np.random.randn(m, n) * np.sqrt(1. / n)
        else:
            raise ValueError(f"Unknown initialization method: {initialize_type}")

        b = np.zeros((m, 1))  # Bias initialization is usually zero
        return w, b

    def tanh(self, x): #ranges from -1 to 1
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def sigmoid(self, x): #ranges from 0 to 1
        return 1 / (1 + np.exp(-x))
    


    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def softmax(self, x): # outputs a propability distribution
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # subtract max for numerical stability
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    

    def softmax_derivative(self, x):
        pass


    def relu(self, x): # ranges from 0 to infinity
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def leaky_relu(self, x, alpha=0.01): 
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x < 0] = alpha
        return dx
    
    def no_activation(self, x):
        return x

    def no_activation_derivative(self, x):
        return np.ones_like(x)

    def forward(self, input,test = False):
        """
        Forward pass through the neural network.
        just activation(WX + B)  where w is (neurons in layer, neurons in previous layer) and x is (neurons in  layer, batch size) ==> z is (neurons in layer, batch size)

        Args:
            input (ndarray): The input data to the network.
            test (bool, optional): Flag indicating whether the forward pass is for testing. Defaults to False.

        Returns:
            ndarray: The output of the network after applying the activation function.
        """
        self.input = input
        self.z = np.dot(self.w, self.input) + self.b  # Storing z for use in backpropagation
        activation_fn, _ = self.activation_functions[self.activation]
        self.out = activation_fn(self.z)
        if self.dropout>0 and test == False:
            self.mask = (np.random.rand(*self.out.shape) > self.dropout).astype(float)  #binary mask on the output of the layer so we drop out random neurons activations  
            self.out *= self.mask
            self.out /= (1 - self.dropout)
        
        return self.out
  

    def backward(self,error, learning_rate, batch_size,L1,L2,beta,optimizer,**kwargs):
        """
        Computes the backward pass of the neural network for a single layer.

        Args:
            error (ndarray): The error from the next layer.
            learning_rate (float): The learning rate for the optimizer.
            batch_size (int): The batch size for the training.
            L1 (float): The L1 regularization strength.
            L2 (float): The L2 regularization strength.
            beta (float): The beta parameter for the momentum optimizer.
            optimizer (str): The name of the optimizer to use.
            **kwargs: Additional keyword arguments.
                - aout (ndarray): The output from the previous layer.
                - y (ndarray): The target values.
                - classification (bool): Flag indicating whether the task is a classification task.

        Returns:
            None

        Raises:
            KeyError: If the optimizer name is not recognized.

        Calculates the error for the current layer, computes the backpropagation error,
        and updates the weights and biases using the optimizer.

        The error for the current layer is calculated as follows:
        - If the layer is the last layer:
            - If the task is a classification task:
                - The error is the difference between the output of the last layer and the target values.
            - Otherwise:
                - The error is the difference between the output of the last layer and the target values,
                  multiplied by the derivative of the activation function at the input of the last layer.
        - Otherwise:
            - The error is the product of the error from the next layer and the derivative of the
              activation function at the input of the current layer.

        The backpropagation error is calculated as the matrix product of the transpose of the
        weights and the error.

        The gradients of the weights and biases are calculated using the L1 and L2 regularization
        strengths.

        The weights and biases are updated using the specified optimizer.
        """
        _, derivative_fn = self.activation_functions[self.activation]
        if self.is_last:
            last_out = kwargs["aout"]
            Y = kwargs["y"]
            classification = kwargs["classification"]
            if classification == True:
                self.error = last_out - Y    # loss wrt the output of the last layer

            else:
                self.error = (last_out - Y)*derivative_fn(self.z)   # dl/da * da/dz  
                
        else:
            self.error = error * derivative_fn(self.z)  # loss wrt the output of the layer  # dl/da * da/dz

           
        self.backprop_error = np.matmul(self.w.T,self.error)  # loss wrt the input of the layer  # dl/da self.error is (number of neurons, batch size) w ==> (number of neurons, number of neurons in previous layer)

        grad_w, grad_b = self.calculate_gradients(L1, L2)
        # print(self.activation,"error : ",self.error)
        # print(self.activation,"backprop_error : ",self.backprop_error)
        # print(self.activation,"grad_w",grad_w)
        # print(self.activation,"grad_b",grad_b)
        optim = self.optimizers[optimizer]
        optim(grad_w, grad_b, learning_rate)
            
    

    def calculate_gradients(self,L1=0, L2=0):
        """
        Calculate the gradients of the weights and biases for the neural network layer.

        you have to know the math of the backpropagation

        dl/dw  = dl/da * da/dz * dz/dw
        dl/da is the loss wrt the output of the layer after activation
        da/dz is the derivative of the activation function of the layer
        dz/dw is derivative of the output wrt the weights

        dl/db = dl/da * da/dz * dz/db
        dz/db = 1
        so dl/db = dl/da * da/dz
        (THE MAGIC OF CHAIN RULE)

        self.error ==> (number of neurons, batch size) , 
        self.input ==> (number of neurons in previous layer, batch size)

        matmul of them will get  w ==> (number of neurons, number of neurons in previous layer) and so on for the rest of the layers

        Parameters:
            L1 (float): L1 regularization parameter (default is 0).
            L2 (float): L2 regularization parameter (default is 0).

        Returns:
            tuple: A tuple containing the gradients of the weights and biases.
        """
        if isinstance(self,batchnorm1d):
            grad_w = np.sum((self.error*self.input_normalized),axis=1,keepdims=True) / self.input_normalized.shape[1] # gradients of gamma and beta of the batchnorm layer
            grad_b = np.sum(self.error,axis=1,keepdims=True) / self.input_normalized.shape[1]
        else:
            grad_w = np.matmul(self.error, self.input.T) / self.input.shape[1]  #(jacobian matrix) loss wrt weights (error of the layer * input of the layer)
            grad_b = np.sum(self.error, axis=1, keepdims=True) / self.input.shape[1]
            if L1 > 0:
                grad_w += L1 * np.sign(self.w) / self.input.shape[1]    # L1 regularization (derivative of L1 wrt self.w)
            if L2 > 0:
                grad_w += L2 * self.w / self.input.shape[1]             # L2 regularization (derivative of L2 wrt self.w)

        if  grad_w.shape != self.w.shape or grad_b.shape != self.b.shape:

            raise ValueError(f"Shape mismatch: {grad_w.shape} vs {self.w.shape}")
        
        return grad_w, grad_b
    
    def Momentum(self,grad_w, grad_b, learning_rate,beta=0.9): #polyak momentum implementation (adjust then make the step of the momentum)
        '''
        momentum is like speeding the convergence when the gradients arent changing too much (the error plane is smooth) so gradients are small so 
        slow convergence . momentum is then used to speed up the process



        polyak original momentum equation we can also use the commented lines instead of the original equation the difference is the commented equation uses an EMA

        exponential decaying moving average momentum (EMA) which doesnt make the big steps i make in the first of the training affect me till the end because there effect

        vanishes exponentially
        '''
        grad_w = np.clip(grad_w, -100, 100)
        grad_b = np.clip(grad_b, -100, 100)
        # self.momentum_w = beta * self.momentum_w + (1 - beta) * grad_w  # exponential decaying moving average momentum (EMA)
        # self.momentum_b = beta * self.momentum_b + (1 - beta) * grad_b
        self.momentum_w = beta * self.momentum_w + learning_rate * grad_w  # polyak original momentum equation
        self.momentum_b = beta * self.momentum_b + learning_rate * grad_b  
        self.w -= self.momentum_w
        self.b -= self.momentum_b 
        
    def NAG(self,grad_w, grad_b, learning_rate,beta=0.9,EMA=False): #nestrov accelerated gradients momentum implementation (make the step of the momentum vector then adjust)
        '''
        nag can also be used with EMA too 
        
        im not sure of this implementation this is only my approach i didnt see a direct implementation of it but what is commonly used is

        (momentum_w = beta * self.momentum_w + learning_rate * grad_w(gradient of the look ahead step(dl/dw of (w-beta*momentum)))) the grad w here is of the look ahead position 
        
        so i tried making the same idea but with a twist to fit my implementation because i cant just update the gradients twice so i made the last two lines

        self.w -= (beta * self.momentum_w)          # look ahead step (make the momentum step first then adjust)
        self.b -= (beta * self.momentum_b)  

        in the first iteration momentum will be 0 so there is no look ahead position so we are good but at any iteration after that the gradient will be of the lookahead position
        so i think this is a correct implementation
        '''
        grad_w = np.clip(grad_w, -100, 100)
        grad_b = np.clip(grad_b, -100, 100)

        if EMA:
            self.momentum_w = beta * self.momentum_w + (1 - beta) * grad_w  # Momentum with exponential decaying moving average
            self.momentum_b = beta * self.momentum_b + (1 - beta) * grad_b  # Momentum
        else:
            self.momentum_w = beta * self.momentum_w + learning_rate * grad_w  # Momentum original equation HB METHOD
            self.momentum_b = beta * self.momentum_b + learning_rate * grad_b  # Momentum

        self.w -= self.momentum_w
        self.b -= self.momentum_b
        
        self.w -= (beta * self.momentum_w)          # look ahead step (make the momentum step first then adjust)
        self.b -= (beta * self.momentum_b)         
        
    def SGD(self,grad_w, grad_b, learning_rate,beta=0.9): #sgd optimizer
        '''
        your normal sgd optimizer

        '''
        #clip large gradient values
        grad_w = np.clip(grad_w, -100, 100)
        grad_b = np.clip(grad_b, -100, 100)
         
        self.w -= learning_rate * grad_w
        self.b -= learning_rate * grad_b
        
    def Adam(self,grad_w, grad_b, learning_rate,beta1=0.9,beta2=0.999):
        '''

        Adam is just momentum + RMSprop + a bias correction 

        what is bias correction?

        lets say in the first step the gradient is 10 and the momentum is 0 and in the momentum equation we have (m = beta1 * m + (1 - beta1) * grad) so the momentum will be 1
        which is (biased towards zero) so it doesnt represent the real change and it will make the learning slow at the beginning  so to correct it we use the equation 
        ( m = m / (1 - beta1 ^ t) ) so in the first iteration

        t=1 so and beta is 0.9 so m afer bias correction will be 10 again so its not biased now and the power of this correction decreases as the iteration increases 

        so that after many iterations there will be no correction 
        

        '''
        grad_w = np.clip(grad_w, -100, 100)
        grad_b = np.clip(grad_b, -100, 100)

        self.t+=1        
        self.momentum_w = beta1 * self.momentum_w + (1 - beta1) * grad_w  # Momentum with exponential decaying moving average
        self.momentum_b = beta1 * self.momentum_b + (1 - beta1) * grad_b  

        self.Accumelated_Gsquare_w = (beta2 * self.Accumelated_Gsquare_w) + ((1 - beta2) * grad_w**2)  # accumulation of squared gradient with exponential decaying moving average
        self.Accumelated_Gsquare_b = (beta2* self.Accumelated_Gsquare_b) + ((1 - beta2) * grad_b**2)

        Gw_corrected = self.Accumelated_Gsquare_w/((1-np.power(beta2, self.t)))
        Gb_corrected = self.Accumelated_Gsquare_b/((1-np.power(beta2, self.t)))

        vw_corrected = self.momentum_w/((1-np.power(beta1, self.t)))   #bias correction  (1 - np.power(beta1, self.t)) becomes closer to 1 as t increases
        vb_corrected = self.momentum_b/((1-np.power(beta1, self.t)))


        ita_w = learning_rate/(np.sqrt(Gw_corrected)+self.eps)
        ita_b = learning_rate/(np.sqrt(Gb_corrected)+self.eps)


        self.w -= ita_w*vw_corrected
        self.b -= ita_b*vb_corrected




    def NAdam(self,grad_w, grad_b, learning_rate,beta1=0.9,beta2=0.999):
         
         '''
         same implementation of adam but we use nesterov momentum instead of polyak's n
         
         '''

         grad_w = np.clip(grad_w, -100, 100)
         grad_b = np.clip(grad_b, -100, 100)
 
         self.t+=1        
         self.momentum_w = beta1 * self.momentum_w + (1 - beta1) * grad_w  # Momentum with exponential decaying moving average
         self.momentum_b = beta1 * self.momentum_b + (1 - beta1) * grad_b  
 
         self.Accumelated_Gsquare_w = (beta2 * self.Accumelated_Gsquare_w) + ((1 - beta2) * grad_w**2)  # accumulation of squared gradient with exponential decaying moving average
         self.Accumelated_Gsquare_b = (beta2* self.Accumelated_Gsquare_b) + ((1 - beta2) * grad_b**2)
 
         Gw_corrected = self.Accumelated_Gsquare_w/((1-np.power(beta2, self.t)))
         Gb_corrected = self.Accumelated_Gsquare_b/((1-np.power(beta2, self.t)))
 
         vw_corrected = self.momentum_w/((1-np.power(beta1, self.t)))   #bias correction  (1 - np.power(beta1, self.t)) becomes closer to 1 as t increases
         vb_corrected = self.momentum_b/((1-np.power(beta1, self.t)))
 
 
         ita_w = learning_rate/(np.sqrt(Gw_corrected)+self.eps)
         ita_b = learning_rate/(np.sqrt(Gb_corrected)+self.eps)
 
 
         self.w -= ita_w*vw_corrected
         self.b -= ita_b*vb_corrected
  
         self.w -= (beta1 * self.momentum_w)          # look ahead step (make the momentum step first then adjust)
         self.b -= (beta1 * self.momentum_b) 
     

    def AdaGrad(self,grad_w, grad_b, learning_rate,beta=0.9):

        '''
        adaptive gradients is assigning a learning rate to each weight why we do that?

        suppose we have 2 features so 2 weights so the loss function will have 2 horizontal dimensions lets imagine it is like a circle but not an even circle it is more like an egg
        so the length of more than the width so when we update the weights is isnt the best decision to go the same distance in the both directions maybe i want to move
        more in the direction of w1 more than w2 because this will be closer to the minima so that what adaptive gradients is doing

        and how does it work?

        if the root of the squared sum of the gradients is big then it will slow down the learning rate why?
        large gradients mean that the slope of the loss function is large so it is going to the minima (hopefully it could be going to the maxima too haha)
        so it is like sliding on a slide so i make the learning rate smaller as it get closer to minima because i want it to stay there.

        but where is the magic?
        i can make this in only one dimension (on only one weight) and if the other weight gradient is consistent with the loss function so it roughly remains the same
        (it decreases but not with large difference)

        BUT IT HAS A PROBLEM
        it doesnt forget past mistakes (like someone i know )
        normally gradients is bigger at the first of the training so that will make the learning rate decrease fast.
        because i accumulated the gradients so even the gradients of the first iterations contribute to it and this might lead to slow learning or maybe even you dont converge
        you dont reach a minima 
        so how to solve this proplem?

        (RMSprop) 

        so conclusion:

        adaptive gradients has 2 cases:
        if im apporaching a mininma (or a maxima) i will slow down the learning rate
        if the slope is steady the learning rate will not change (change with a small value)


        
        '''
        grad_w = np.clip(grad_w, -100, 100)
        grad_b = np.clip(grad_b, -100, 100)

        self.Accumelated_Gsquare_w+=grad_w**2
        self.Accumelated_Gsquare_b+=grad_b**2

        ita_w = learning_rate/(np.sqrt(self.Accumelated_Gsquare_w)+self.eps)
        ita_b = learning_rate/(np.sqrt(self.Accumelated_Gsquare_b)+self.eps)


        self.w -= ita_w*grad_w
        self.b -= ita_b*grad_b


    def RMSprop(self,grad_w, grad_b, learning_rate,beta=0.9):
        '''
        same idea of adaGrad but with an EMA (Exponential Moving Average) (i think anyone stuck with his algorithm try EMA and it works lol)
        
        EMA is the man for the job why?
        because now we can forgive past mistakes (or mistakes that happened in the past)
        because only (1-beta) of the new Accumelated_Gsquare_w will be added and 0.9 of the past Accumelated_Gsquare_w will remain 
        and that continues which lead too exponential decaay of the first mistakes so now the learning rate depend only on the recent mistakes not the past ones

        
        '''
        
        grad_w = np.clip(grad_w, -100, 100)
        grad_b = np.clip(grad_b, -100, 100)

        self.Accumelated_Gsquare_w = (beta * self.Accumelated_Gsquare_w) + ((1 - beta) * grad_w**2)
        self.Accumelated_Gsquare_b = (beta * self.Accumelated_Gsquare_b) + ((1 - beta) * grad_b**2)

        ita_w = learning_rate/(np.sqrt(self.Accumelated_Gsquare_w)+self.eps)
        ita_b = learning_rate/(np.sqrt(self.Accumelated_Gsquare_b)+self.eps)


        self.w -= ita_w*grad_w
        self.b -= ita_b*grad_b



class batchnorm1d(Layer):
    def __init__(self,m):
        """
        batch normalization is a regularization technique that normalizes the input to the layer
        (like we do standard scaling to the features before feeding it to the model)
        we do the same thing with batchnormalization.

        we start with the output of the first layer we calc the mean and variance for every feature the output of the layer is (neurons of layer,batch_size)
        and then normalize the output (output-mean/sqrt(var+eps)) then 
        we scale the normalized output by gamma and add beta and then feed this output to the next layer as input 
        what is the benefit of that?

        it makes the model more stable and decrease the internal covariate shift what the hell is internal covariates shift?
        it is the change in the distribution of the input to each layer during training this is called internal covariate shift
        when the distribution changes alot this makes the network adapt its weights to the change in the distribution of the input not in learning the feature it self 
        so it might overfit .

        i implemented the paper from here https://arxiv.org/abs/1502.03167 it isnt a hard paper to read

        one more thing you need to know about batch normalization so what after calculating the mean and variance for every feature what about the test data do i 
        calc the mean and variance for it too?

        no you dont you keep track of the running mean and running variance for each feature 
        so when you calc the var and mean you in each step you do the following 
        running_mean = beta*running_mean + (1-beta)*mean
        running_var = beta*running_var + (1-beta)*var
        looks familiar right? yes it is the very good EMA (Exponential Moving Average)


        """
        super().__init__()
        self.w = np.ones((m,1))
        self.b = np.zeros((m,1))  #0
        self.momentum_w = np.zeros((m, 1))
        self.momentum_b = np.zeros((m, 1))
        self.running_mean = np.zeros((m,1))
        self.running_variance = np.ones((m,1))
        self.Accumelated_Gsquare_w = np.zeros((m, 1))
        self.Accumelated_Gsquare_b = np.zeros((m, 1))
        self.eps = 1e-5
        self.input_normalized = None
        self.input = None
        self.betanorm = 0.9
        self.mean = None
        self.var = None
        
    def forward(self, input,test=False):
        """
        Calculates the output of the batchnorm layer for a given input.

        Args:
            input (numpy.ndarray): The input to the layer.
            test (bool, optional): Whether the layer is in test mode. Defaults to False.

        Returns:
            numpy.ndarray: The output of the layer.
        """
        self.input = input
        if test == False:
            self.mean = np.mean(self.input, axis=1, keepdims=True)
            self.var = np.var(self.input, axis=1, keepdims=True)
            
            self.running_variance = (self.betanorm*self.running_variance) + ((1-self.betanorm)*self.var) #EMA of mean and variance that will be used in testing mode
            
            self.running_mean = (self.betanorm*self.running_mean) + ((1-self.betanorm)*self.mean)
            
            
            self.input_normalized = (self.input - self.mean) / np.sqrt(self.var + self.eps)
            
            self.out = self.w * self.input_normalized + self.b
            
        elif test == True:  #use the running mean and variance in inference
            self.input_normalized =  (self.input - self.running_mean) / np.sqrt(self.running_variance + self.eps)
            self.out = self.w * self.input_normalized + self.b
        return self.out
    
    def backward(self,error, learning_rate, batch_size,L1,L2,beta,optimizer,**kwargs):
        '''
        this equations is provided in the paper i just vectorized them and implemented them
        
        '''

        if error is None:
            raise ValueError("prev_w and prev_error must be provided")
        if batch_size == 0:
            raise ValueError("batch_size cannot be zero")
        
        self.error = error #error wrt output of the batch norm
        normalizedipnut_grad = self.error * self.w

        variance_grad = np.sum(normalizedipnut_grad*(self.input-self.mean)*(-0.5)*np.power((self.var+self.eps),-1.5),axis=1,keepdims=True)
        
        mean_grad = np.sum(normalizedipnut_grad*(-1/(np.sqrt(self.var+self.eps))),axis=1,keepdims=True) + ((variance_grad)*(np.mean((-2*(self.input-self.mean)),axis=1,keepdims=True)))     

        self.backprop_error = (normalizedipnut_grad *(1/np.sqrt(self.var+self.eps))) +(variance_grad*(2*(self.input-self.mean)/batch_size))+(mean_grad/batch_size) # loss wrt the input of the batch layer
        
        

        grad_w, grad_b = self.calculate_gradients(L1, L2)
        optim = self.optimizers[optimizer]
        optim(grad_w, grad_b, learning_rate)
    
    
class Network:
    def __init__(self, neurons_each_layer, activations,dropout, classification=False, threshold=0.5,batchnorm = False,initialize_type="he"):
        """
        Initializes a Network object with the given parameters.

        Args:
            neurons_each_layer (List[int]): A list of integers representing the number of neurons in each layer of the network. The last element is the number of output neurons.
            activations (List[str]): A list of strings representing the activation functions to be used for each layer. The length of this list should be one less than the length of `neurons_each_layer`.
            dropout (List[float]): A list of floats representing the dropout rates for each layer. The length of this list should be one less than the length of `neurons_each_layer`. If a value is `None`, dropout is not applied to that layer.
            classification (bool, optional): A boolean indicating whether the network is a classification network. Defaults to False.
            threshold (float, optional): A float representing the threshold for classification networks. Defaults to 0.5.
            batchnorm (bool, optional): A boolean indicating whether batch normalization should be applied to the network. Defaults to False.
            initialize_type (str, optional): A string representing the type of initialization to be used for the weights. Defaults to "he".

        Returns:
            None

        Raises:
            ValueError: If the length of `neurons_each_layer` is less than 2.
            ValueError: If the length of `activations` is not one less than the length of `neurons_each_layer`.
            ValueError: If the length of `dropout` is not one less than the length of `neurons_each_layer`.

        Note:
            - The last layer of the network has dropout applied to it.
            - If `batchnorm` is True and the layer is not the last layer, batch normalization is applied to the layer.
        """
        self.activations = activations
        self.layers = []
        self.loss = 0
        self.classification = classification
        self.threshold = threshold
        self.batchnorm = False
        self.losses = []
        self.last_out = None
        self.Y = None
        
        for i in range(len(neurons_each_layer) - 1):   
            dropout_rate = dropout[i] if dropout and dropout[i] is not None else 0
            self.layers.append(Layer(neurons_each_layer[i + 1], neurons_each_layer[i], activations[i], initialize_type, dropout_rate))
            if batchnorm == True and i != len(neurons_each_layer) - 2:
                self.layers.append(batchnorm1d(neurons_each_layer[i + 1]))
            
       

        
        self.layers[-1].is_last = True

        if self.layers[-1].dropout != 0:
            print("last layer has dropout just reminding you that dropout is not applied to the last layer")

            

    def forward(self, X,test = False):
        '''
        Your normal forward pass

        output of each layer becomes input to the next layer
        '''
        for layer in self.layers:
            X = layer.forward(X,test)

        assert X is not None

        self.last_out = X
                

    def binary_cross_entropy(self,Y,last_out):
        '''
        assumes y is of size (1,N) where N is batch size

        '''
        return -np.mean(Y * np.log(last_out) + (1 - Y) * np.log(1 - last_out))
    
    def sparse_categorical_cross_entropy(self,Y,last_out):
        '''
        assumes y is of size (num_of_classes,N) where N is batch size

        '''
        return -np.mean(np.sum(Y * np.log(last_out), axis=0))
    
    def mean_squared_error(self,Y,last_out):
        return np.mean(np.square(Y - last_out))
    

    def compute_loss(self,Y):
        if self.classification==True:   #binary cross entropy and sparse categorical cross entropy
           if Y.shape[0] == 1:
               self.loss += self.binary_cross_entropy(Y,self.last_out)
           else:
               self.loss += self.sparse_categorical_cross_entropy(Y,self.last_out)
        
        else:
            self.loss += self.mean_squared_error(Y,self.last_out)


    def backward(self, X, Y, learning_rate, batch_size,L1,L2,beta,optimizer):
        '''
        make the forward pass
        calculate the loss
        calc the loss wrt the output of each layer
        backprop the error
        update the weights
        
        '''

        
        self.forward(X) # forward pass
        

        if self.last_out.shape != Y.shape:
            raise ValueError("y and aout must have the same shape check the shape of the input ")
        if batch_size == 0:
            raise ValueError("batch_size cannot be zero")
        

        self.compute_loss(Y)
        # print("last out",self.last_out)
        # print("last out shape",self.last_out.shape)
        # print("Y",Y.shape)
        # print("Y",Y)

        for i in range(len(self.layers) - 1, -1, -1):
            if self.layers[i].is_last == True:  #backprop the last layer
                self.layers[i].backward(0, learning_rate, batch_size,L1,L2,beta,optimizer,y=Y,aout=self.last_out,classification=self.classification)
            else:
                self.layers[i].backward(self.layers[i + 1].backprop_error,learning_rate,batch_size,L1,L2,beta,optimizer)# backprop any  layer pass to the function the back propagated error

    @timing_decorator
    def train(self, x_train, y_train, epochs=10, batch_size=32, learning_rate=0.001, verbose=True,L1=0,L2=0,beta=0.9,optimizer="momentum"):
        """
        Trains the neural network model on the given training data for a specified number of epochs.
        if multi class classification the labels must be one hot encoded
        Parameters:
            x_train (ndarray): The input training data. each row is an example
            y_train (ndarray): The target training data. each row is an example input must be one hot encoded for multi classification
            epochs (int, optional): The number of epochs to train for. Defaults to 10.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            learning_rate (float, optional): The learning rate for training. Defaults to 0.001.
            verbose (bool, optional): Whether to print the training progress. Defaults to True.
            L1 (float, optional): The L1 regularization strength. Defaults to 0.
            L2 (float, optional): The L2 regularization strength. Defaults to 0.
            beta (float, optional): The momentum factor for the optimizer. Defaults to 0.9.
            optimizer (str, optional): The optimizer to use for training. Defaults to "momentum".

        Returns:
            None
        """
        #calc time of training
        batches_len=0
        for epoch in range(1, epochs + 1):
            indices = np.arange(x_train.shape[0])
            np.random.shuffle(indices)   #randomize the batch
            x = x_train[indices, :] 
            y = y_train[indices, :]
            
            # cnt=0
            for i in range(0, x.shape[0], batch_size):
                # cnt+=1
                # if cnt ==2: # this is to run it for one iteration to check correctness
                #     break
                X_batch = x[i:i + batch_size, :]
                y_batch = y[i:i + batch_size, :]
                self.backward(X_batch.T, y_batch.T, learning_rate, batch_size, L1, L2, beta,optimizer) #network takes a column vector as input
                batches_len+=1

            if verbose:
                percent = (epoch / epochs) * 100
                bar = '█' * int(percent // 2) + '-' * (50 - int(percent // 2))
                epoch_loss = self.loss/batches_len
                self.losses.append(epoch_loss)
                print(f'\rEpoch {epoch}/{epochs} | [{bar}] {percent:.2f}% |  Train Loss = {epoch_loss:.4f}', end='')

            

       

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
        X = X.T
        self.forward(X, test=True)
        out = self.last_out.T
        if self.classification:
            if out.shape[1] == 1:  # Binary classification
                return (out >= self.threshold).astype(int).flatten()
            else:  
                return np.argmax(out, axis=1).flatten() # the output has one dimension which is the class number
        else:
            return out.flatten()  #output of regression