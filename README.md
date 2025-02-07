# COCO: A Mini Educational Deep Learning Framework

COCO is a mini deep learning framework built **from scratch** in NumPy for educational purposes. It helps you understand the inner workings of neural network components—from layers and activation functions to optimizers and convolution operations. Use it to build and experiment with your own models.

---

## Code Structure

- **nn.py**: Contains the base classes (`Module`, `Layer`) and implementations of layers:
  - **Linear**: Implements an affine transformation with various weight initialization options (e.g., `xavier`, `he`, `lecun`).
  - **Conv2d**: Performs 2D convolutions using the `FastConvolver` class from `operations.py`.
  - **MaxPool2d**: Implements max pooling with a backward pass for gradient propagation.
  - **batchnorm1d**: Implements 1D batch normalization, including updating running statistics and backpropagation.
- **optim.py**: Implements a variety of optimizers including **SGD**, **momentum**, **Nesterov Accelerated Gradient (NAG)**, **Adam**, **NAdam**, **AdaGrad**, and **RMSprop**. Each optimizer can be selected by name when training your model.
- **operations.py**: Implements the `FastConvolver` class, which leverages the im2col method to perform efficient convolution operations. It includes methods for:
  - Converting images to column format (`_im2col`)
  - Reshaping filters for matrix multiplication (`_transform_kernels`)
  - Reconstructing images from column data (`col2im_accumulation`)

---

## Usage Example

Below is an example of a Convolutional Neural Network (CNN) built with COCO to classify the MNIST dataset.

### Model Definition

```python
from coco.nn import Model, Conv2d, MaxPool2d, Linear, tanh, softmax

class MyModel(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(input_channels=1, output_channels=8, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.tanh1 = tanh()
        self.max1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(input_channels=8, output_channels=16, kernel_size=3, stride=1, padding=1, initialize_type='xavier')
        self.tanh2 = tanh()
        self.max2 = MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = Linear((16 * 7 * 7, 50), initialize_type='xavier')
        self.tanh3 = tanh()
        self.linear2 = Linear((50, 10), initialize_type='xavier')
        self.softmaxx = softmax()

    def forward(self, x):
        # Example forward pass with an optional skip connection:
        i = x
        x = self.conv1(x)
        x = self.tanh1(x)
        # Adding a skip connection (if the dimensions match)
        x = x + i  
        x = self.max1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.max2(x)
        x = self.linear1(x)
        x = self.tanh3(x)
        x = self.linear2(x)
        x = self.softmaxx(x)
        return x

# Instantiate your model:
model = MyModel()
model.summary()  # Displays a summary of the model’s architecture

# Train your model (ensure you have loaded train_images and train_labels):
model.train(train_images, train_labels, epochs=10, batch_size=64, learning_rate=0.001, optimizer="sgd", verbose=1)
```

---

## Key Points

- **Layer Customization**: Modify any layer (e.g., change initialization or activation functions) to see how it impacts model performance.
- **Optimizer Flexibility**: Choose from multiple optimizers by simply passing their name (e.g., `"adam"` or `"nadam"`) during training.
- **Fast Convolution**: The convolution operation is handled by a highly optimized `FastConvolver` class using im2col, ensuring efficient computation even when implemented purely in NumPy.

---

## Roadmap

Future enhancements include:
- **Pretrained Models**: Integrate models like ResNet50, MobileNet, and VGG.
- **Static Computational Graphs**: Optimize execution with static graphs.
- **Advanced Convolution Techniques**: Implement state-of-the-art convolution algorithms and GPU support.
- **Custom CUDA Kernels**: Accelerate convolution with custom CUDA implementations.
- **Extended Layer and Loss Function Support**: Expand the library with more components.

---

## Contributing

COCO is open-source and built for educational purposes. We welcome contributions:
- **Fork** the repository.
- Check out `tasks.md` for upcoming features or bug fixes.
- **Submit a pull request** with your improvements.

By contributing, you not only deepen your understanding of AI but also help empower engineers worldwide.

---

## Mission

Our mission is to democratize deep learning education by giving everyone—from beginners to experts—the tools to understand and experiment with AI at a fundamental level.

**contribution**

- Fork the repo
- check TASKS.md 
- implement a feature
- submit a pull request 
