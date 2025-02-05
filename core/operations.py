import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class FastConvolver:
    """
    Implements efficient 2D convolution using the im2col approach.

    This class transforms input feature maps into column format to enable
    matrix multiplication-based convolution, significantly improving performance
    compared to naive implementations.

    Features:
    - Supports multi-channel inputs and multiple filters.
    - Allows custom stride and padding configurations.
    - Provides utility functions for im2col and col2im transformations.

    Example usage:
        convolver = FastConvolver()
        output, col_matrix = convolver.convolve(input_data, kernels, stride=1, padding=1)

    """
    def __init__(self):
        pass

    def _im2col(self, input_data, kernel_shape, stride=1):
        """
                Converts an input batch into a columnized matrix for efficient convolution.

                This function extracts patches from the input tensor and flattens them into
                rows, enabling efficient matrix multiplication.

                Parameters:
                    input_data (numpy.ndarray): Padded input data of shape (B, C, H, W).
                    kernel_shape (tuple): Tuple (C, H_k, W_k) describing the kernel dimensions.
                    stride (int): Stride of the convolution (applied to height & width).

                Returns:
                    col_matrix (numpy.ndarray): 2D array of shape (B * H_out * W_out, C * H_k * W_k)
                        where each row corresponds to a flattened receptive field.

                Notes:
                    - Assumes input_data is already padded.
                    - Uses `sliding_window_view` for efficient patch extraction.
                """
        B, C, H, W = input_data.shape
        # kernel_shape is expected to be (C, H_k, W_k)
        _, H_k, W_k = kernel_shape

        # Extract sliding windows. This returns an array of shape:
        # (B, C, H - H_k + 1, W - W_k + 1, H_k, W_k)
        windows = sliding_window_view(input_data, (H_k, W_k), axis=(2, 3))

        # Apply the stride by slicing the spatial dimensions:
        windows = windows[:, :, ::stride, ::stride, :, :]
        # Now, windows.shape is (B, C, H_out, W_out, H_k, W_k) where:
        # H_out = (H - H_k + 1) // stride  and  W_out = (W - W_k + 1) // stride

        B, C, H_out, W_out, H_k, W_k = windows.shape

        # Rearrange axes so that each patch becomes a row:
        # from (B, C, H_out, W_out, H_k, W_k) to (B, H_out, W_out, C, H_k, W_k)
        # and then reshape to (B * H_out * W_out, C * H_k * W_k)
        col_matrix = windows.transpose(0, 2, 3, 1, 4, 5).reshape(B * H_out * W_out, C * H_k * W_k)
        return col_matrix

    def _transform_kernels(self, kernels):
        """
        Converts convolution filters into a matrix form for multiplication.

        Parameters:
            kernels (numpy.ndarray): Convolution filters of shape (F, C, H_k, W_k),
                                     where F is the number of filters.

        Returns:
            numpy.ndarray: Reshaped kernel matrix of shape (C * H_k * W_k, F),
                           where each column represents a flattened kernel.

        Notes:
            - This transformation allows convolution to be performed using
              standard matrix multiplication.
        """
        F, C, H_k, W_k = kernels.shape
        # Reshape each kernel to a vector and then transpose so that
        # the kernel matrix is of shape (C*H_k*W_k, F)
        return kernels.reshape(F, C * H_k * W_k).T

    def _col2im(self, result_matrix, B, H_out, W_out, num_filters):
        """
        Reshape the result matrix back to the convolution output dimensions.
        (Note: This version is for the forward pass and only performs reshaping,
         not the full accumulation needed for a backward col2im with overlaps.)

        Parameters:
            result_matrix (numpy.ndarray): Matrix of shape (B * H_out * W_out, num_filters).
            B (int): Batch size.
            H_out (int): Output height.
            W_out (int): Output width.
            num_filters (int): Number of filters (output channels).

        Returns:
            numpy.ndarray: Convolution output of shape (B, num_filters, H_out, W_out).
        """
        return result_matrix.reshape(B, H_out, W_out, num_filters).transpose(0, 3, 1, 2)
    
    def col2im_accumulation(self,dX_col, input_shape, filter_height, filter_width, stride, padding):
        """
        Reconstructs the gradient tensor by summing overlapping regions.

        Used in backpropagation to redistribute gradients from the im2col representation
        back to the original input shape.

        Parameters:
            dX_col (numpy.ndarray): 2D array of shape (B * H_out * W_out, C * filter_height * filter_width),
                                    representing gradient patches.
            input_shape (tuple): Original input shape (B, C, H, W).
            filter_height (int): Height of the convolution filter.
            filter_width (int): Width of the convolution filter.
            stride (int): Stride used in the forward pass.
            padding (int): Amount of zero-padding applied to the input.

        Returns:
            numpy.ndarray: Reconstructed gradient of shape (B, C, H + 2*padding, W + 2*padding).

        Notes:
            - Accumulates overlapping regions correctly.
            - This version assumes simple summation without additional normalization.
        """
        B, C, H, W = input_shape
        H_padded = H + 2 * padding
        W_padded = W + 2 * padding

        # Calculate output spatial dims for padded input:
        H_out = (H_padded - filter_height) // stride + 1
        W_out = (W_padded - filter_width) // stride + 1

        # Initialize the gradient array for padded input.
        dInput_padded = np.zeros((B, C, H_padded, W_padded))

        # Reshape dX_col into (B, H_out, W_out, C, filter_height, filter_width)
        dX_col_reshaped = dX_col.reshape(B, H_out, W_out, C, filter_height, filter_width)
        # Permute to (B, C, H_out, W_out, filter_height, filter_width)
        dX_col_reshaped = dX_col_reshaped.transpose(0, 3, 1, 2, 4, 5)

        # Accumulate gradients into the padded input.
        for i in range(filter_height):
            for j in range(filter_width):
                dInput_padded[:, :, i: i + stride * H_out: stride, j: j + stride * W_out: stride] += dX_col_reshaped[:, :, :, :, i, j]

        return dInput_padded

    def convolve(self, input_data, kernels, stride=1, padding=0):
        """
        Performs 2D convolution using the im2col approach.

        This method extracts patches from the input feature map,
        converts them into a column matrix, and performs convolution using
        matrix multiplication.

        Parameters:
            input_data (numpy.ndarray): Input tensor of shape (B, C, H, W).
            kernels (numpy.ndarray): Filters of shape (F, C, H_k, W_k),
                                     where F is the number of filters.
            stride (int): Stride value for convolution.
            padding (int): Amount of zero-padding applied before convolution.

        Returns:
            conv_output (numpy.ndarray): Output tensor of shape (B, F, H_out, W_out).
            col_matrix (numpy.ndarray): Columnized patches from the padded input.

        Notes:
            - Uses `im2col` for efficient patch extraction.
            - Output dimensions are computed as:
                H_out = (H + 2 * padding - H_k) // stride + 1
                W_out = (W + 2 * padding - W_k) // stride + 1
        """
        B, C, H, W = input_data.shape
        F, C, H_k, W_k = kernels.shape

        # Compute output dimensions based on padded input size.
        H_out = (H + 2 * padding - H_k) // stride + 1
        W_out = (W + 2 * padding - W_k) // stride + 1

        # Pad the input (only on spatial dimensions)
        padded_input = np.pad(
            input_data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode='constant'
        )

        # Convert the padded input into columns
        col_matrix = self._im2col(padded_input, kernel_shape=(C, H_k, W_k), stride=stride)

        # Transform kernels into a matrix for multiplication
        kernel_matrix = self._transform_kernels(kernels)

        # Matrix multiplication:
        #   (B*H_out*W_out, C*H_k*W_k) @ (C*H_k*W_k, F) -> (B*H_out*W_out, F)
        result_matrix = col_matrix @ kernel_matrix

        # Reshape the result into (B, F, H_out, W_out)
        conv_output = self._col2im(result_matrix, B, H_out, W_out, F)
        return conv_output, col_matrix
