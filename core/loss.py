import numpy as np

def binary_cross_entropy(Y,last_out):
    '''
    assumes y is of size (1,N) where N is batch size

    '''
    B = last_out.shape[0]
    return -np.mean(Y * np.log(last_out) + (1 - Y) * np.log(1 - last_out),axis=0) ,(last_out -Y)/B


def sparse_categorical_cross_entropy(Y, last_out, axis=1):
    '''
    assumes y is of size (N, num_classes) where N is batch size.
    This handles sparse matrix conversion before calculating the loss.
    '''
    # Convert sparse matrix to dense format
    if isinstance(Y, np.ndarray):
        Y_dense = Y
    else:
        Y_dense = Y.toarray()

    B = last_out.shape[0]


    return -np.mean(np.sum(Y_dense * np.log(last_out), axis=1), axis=0), (last_out - Y_dense)/B


def mean_squared_error(Y,last_out):
    B = last_out.shape[0]
    return np.mean(np.square(Y - last_out)), ((last_out - Y)*last_out)/B