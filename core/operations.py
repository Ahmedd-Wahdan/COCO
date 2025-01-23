import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
class FastConvolver:
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