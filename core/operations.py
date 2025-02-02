import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class FastConvolver:
    def __init__(self):
        pass

    def _im2col(self, input_data, kernel_shape, stride=1):
        """
        Vectorized im2col for batches using sliding_window_view.

        Parameters:
            input_data (numpy.ndarray): Padded input data of shape (B, C, H, W).
            kernel_shape (tuple): Tuple (C, H_k, W_k) describing the kernel dimensions.
            stride (int): Stride of the convolution.

        Returns:
            col_matrix (numpy.ndarray): 2D array of shape (B * H_out * W_out, C * H_k * W_k)
                where each row is a flattened patch.
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
        Transform the kernels into a 2D matrix for matrix multiplication.

        Parameters:
            kernels (numpy.ndarray): Kernels of shape (F, C, H_k, W_k).

        Returns:
            numpy.ndarray: Transformed kernels of shape (C * H_k * W_k, F).
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
        Efficiently fold the columns back into the image shape, summing over overlaps.
        Parameters:
            dX_col (numpy.ndarray): 2D array of shape (B*H_out*W_out, C*filter_height*filter_width)
            input_shape (tuple): Original input shape (B, C, H, W)
            filter_height (int): Height of the filter.
            filter_width (int): Width of the filter.
            stride (int): Stride.
            padding (int): Padding that was applied.
        Returns:
            dInput: Gradient w.r.t the padded input of shape (B, C, H + 2*padding, W + 2*padding).
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
        Perform 2D convolution using the im2col approach.

        Parameters:
            input_data (numpy.ndarray): Input data of shape (B, C, H, W).
            kernels (numpy.ndarray): Kernels of shape (F, C, H_k, W_k).
            stride (int): Stride of the convolution.
            padding (int): Amount of zero-padding to add on the spatial dimensions.

        Returns:
            conv_output (numpy.ndarray): Output data of shape (B, F, H_out, W_out).
            col_matrix (numpy.ndarray): Columnized patches from the padded input.
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
