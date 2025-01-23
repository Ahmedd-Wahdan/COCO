import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time
class FastConvolver:
    def __init__(self):
        pass

    def _im2col(self, input_data, kernel_shape, stride=1, padding=0):
        """
        Transform the input image into column form for efficient matrix multiplication.

        Parameters:
            input_data (numpy.ndarray): Input image of shape (C, H, W).
            kernel_shape (tuple): Shape of the kernel (C, H_k, W_k).
            stride (int): Stride of the convolution.
            padding (int): Padding to apply to the input image.

        Returns:
            numpy.ndarray: Column-transformed input.
        """
        C, H, W = input_data.shape
        _, H_k, W_k = kernel_shape

        # padding
        padded_input = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)), mode='constant')

        # output dimensions
        H_out = (H + 2 * padding - H_k) // stride + 1
        W_out = (W + 2 * padding - W_k) // stride + 1

        # extracting patches
        col_matrix = []
        for i in range(H_out):
            for j in range(W_out):
                start_row = i * stride
                start_col = j * stride
                patch = padded_input[:, start_row:start_row + H_k, start_col:start_col + W_k]
                col_matrix.append(patch.flatten())

        return np.array(col_matrix).T, H_out, W_out

    def _transform_kernels(self, kernels):
        """
        Transform the kernels into row form for matrix multiplication.

        Parameters:
            kernels (numpy.ndarray): Kernels of shape (F, C, H_k, W_k).

        Returns:
            numpy.ndarray: Row-transformed kernels.
        """
        F, C, H_k, W_k = kernels.shape
        return kernels.reshape(F, -1)

    def _col2im(self, col_matrix, H_out, W_out, num_filters):
        """
        Transform the result back to the convolution output dimensions.

        Parameters:
            col_matrix (numpy.ndarray): Resultant matrix from matrix multiplication.
            H_out (int): Height of the output feature map.
            W_out (int): Width of the output feature map.
            num_filters (int): Number of filters (output channels).

        Returns:
            numpy.ndarray: Reshaped convolution output.
        """
        return col_matrix.reshape(num_filters, H_out, W_out)

    def convolve(self, input_data, kernels, stride=1, padding=0):
        """
        Perform 2D convolution on a batch of images using im2col approach.

        Parameters:
            input_data (numpy.ndarray): Batch of images of shape (N, C, H, W).
            kernels (numpy.ndarray): Kernels of shape (F, C, H_k, W_k).
            stride (int): Stride of the convolution.
            padding (int): Padding to apply to the input image.

        Returns:
            numpy.ndarray: Convolution output of shape (N, F, H_out, W_out).
        """
        N, C, H, W = input_data.shape
        F, _, H_k, W_k = kernels.shape

        outputs = []
        for i in range(N):
            col_matrix, H_out, W_out = self._im2col(input_data[i], (C, H_k, W_k), stride, padding)
            kernel_matrix = self._transform_kernels(kernels)
            result_matrix = kernel_matrix @ col_matrix
            conv_output = self._col2im(result_matrix, H_out, W_out, F)
            outputs.append(conv_output)

        return np.stack(outputs)  # Shape: (N, F, H_out, W_out
    



class FastestConvolver:
    def __init__(self):
        pass



    def _im2col(self, input_data, kernel_shape, stride=1, padding=0):
        """
        Vectorized im2col for batches using sliding_window_view.
        """
        B, C, H, W = input_data.shape
        input_channels, H_k, W_k = kernel_shape

        # Pad input (only spatial dimensions)
        padded_input = np.pad(
            input_data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )

        # Compute output dimensions
        H_out = (H + 2 * padding - H_k) // stride + 1
        W_out = (W + 2 * padding - W_k) // stride + 1

        # Extract patches using sliding_window_view (vectorized)
        # Shape: (B, C, H_out, W_out, H_k, W_k)
        windows = sliding_window_view(padded_input, (H_k, W_k), axis=(2, 3))
        
        # Slice windows according to stride
        windows = windows[:, :, ::stride, ::stride, :, :]

        # Reshape to (B * H_out * W_out, C * H_k * W_k)
        col_matrix = windows.transpose(0, 2, 3, 1, 4, 5).reshape(-1, C * H_k * W_k)#here each row is a patch and the patches of all the images are stacked on each others
        #col matrix if of shape (B*H_out*W_out, C*H_k*W_k) 

        return col_matrix, H_out, W_out
    

    def _transform_kernels(self, kernels):
        """
        Transform the kernels into row form for matrix multiplication.

        Parameters:
            kernels (numpy.ndarray): Kernels of shape (F, C, H_k, W_k).

        Returns:
            numpy.ndarray: Row-transformed kernels.
        """
        F, C, H_k, W_k = kernels.shape
        return kernels.reshape(F,-1).T

    def _col2im(self, result_matrix,B, H_out, W_out, num_filters):
        """
        Transform the result back to the convolution output dimensions.

        Parameters:
            col_matrix (numpy.ndarray): Resultant matrix from matrix multiplication.
            H_out (int): Height of the output feature map.
            W_out (int): Width of the output feature map.
            num_filters (int): Number of filters (output channels).

        Returns:
            numpy.ndarray: Reshaped convolution output.
        """
        

        return result_matrix.reshape(B, H_out, W_out, num_filters).transpose(0, 3, 1, 2)

    def convolve(self, input_data, kernels, stride=1, padding=0):
        """
        Perform 2D convolution using im2col approach.

        Parameters:
            input_data (numpy.ndarray): Input image of shape (C, H, W).
            kernels (numpy.ndarray): Kernels of shape (F, C, H_k, W_k).
            stride (int): Stride of the convolution.
            padding (int): Padding to apply to the input image.

        Returns:
            numpy.ndarray: Convolution output of shape (B, output_channels, H_out, W_out)
        """

        output_channels, C, H_k, W_k = kernels.shape
        B = input_data.shape[0]

        col_matrix, H_out, W_out = self._im2col(input_data, (C, H_k, W_k), stride, padding)


        kernel_matrix = self._transform_kernels(kernels)

        #kernel_matrix is of shape (C*H_k*W_k, F) each column is a kernel
        #col_matrix is of shape (B*H_out*W_out, C*H_k*W_k) each row is a patch


        result_matrix =  col_matrix @ kernel_matrix

        #result_matrix is of shape (B*H_out*W_out, F) each row is the result of the convolution of a patch with all the kernels

        conv_output = self._col2im(result_matrix,B, H_out, W_out, kernels.shape[0])

        #conv_output is of shape (B, output_channels, H_out, W_out)

        return conv_output
    

# Define input data and kernels
np.random.seed(0)
input_data = np.random.rand(16, 3, 32, 32)
kernels = np.random.rand(5, 3, 3, 3)

# Initialize FastConvolver and FastestConvolver
fast_convolver = FastConvolver()
fastest_convolver = FastestConvolver()

# Perform convolution using FastConvolver
start1 = time.time()
conv_output_fast = fast_convolver.convolve(input_data, kernels, stride=1, padding=0)
end1 = time.time()
print("fast convolver shape",conv_output_fast.shape)


# Perform convolution using FastestConvolver
start2 = time.time()
conv_output_fastest = fastest_convolver.convolve(input_data, kernels, stride=1, padding=0)
end2 = time.time()
print("fastest convolver shape",conv_output_fastest.shape)

print("Time taken by FastConvolver: ", end1 - start1)
print("Time taken by FastestConvolver: ", end2 - start2)
# Compare the results