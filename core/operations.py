import numpy as np

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
        Perform 2D convolution using im2col approach.

        Parameters:
            input_data (numpy.ndarray): Input image of shape (C, H, W).
            kernels (numpy.ndarray): Kernels of shape (F, C, H_k, W_k).
            stride (int): Stride of the convolution.
            padding (int): Padding to apply to the input image.

        Returns:
            numpy.ndarray: Convolution output of shape (F, H_out, W_out).
        """

        _, C, H_k, W_k = kernels.shape


        col_matrix, H_out, W_out = self._im2col(input_data, (C, H_k, W_k), stride, padding)


        kernel_matrix = self._transform_kernels(kernels)


        result_matrix = kernel_matrix @ col_matrix


        conv_output = self._col2im(result_matrix, H_out, W_out, kernels.shape[0])

        return conv_output