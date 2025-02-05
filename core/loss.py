import numpy as np

# Global constant for numerical stability in logarithms.
EPSILON = 1e-12


def binary_cross_entropy(y_true, y_pred):
    """
    Compute the binary cross entropy loss and its gradient.

    Assumes:
        - y_true is an array of shape (B, 1), where B is the batch size.
        - y_pred is an array of shape (B, 1) containing predicted probabilities,
          with values in (0, 1).

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth labels.
    y_pred : numpy.ndarray
        Predicted probabilities.

    Returns
    -------
    loss : numpy.ndarray
        The average binary cross entropy loss (averaged over the batch).
    grad : numpy.ndarray
        The gradient of the loss with respect to y_pred.

    Notes
    -----
    A small constant EPSILON is added to y_pred (and 1 - y_pred) to avoid log(0) issues.
    """
    B = y_pred.shape[0]
    # Clip predictions to ensure they are within [EPSILON, 1 - EPSILON]
    y_pred_clipped = np.clip(y_pred, EPSILON, 1 - EPSILON)
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped), axis=0)
    grad = (y_pred_clipped - y_true) / B
    return loss, grad


def sparse_categorical_cross_entropy(y_true, y_pred, axis=1):
    """
    Compute the sparse categorical cross entropy loss and its gradient.

    This function converts a sparse matrix (if applicable) to dense format before
    calculating the loss.

    Assumes:
        - y_true is either a dense numpy array or a sparse matrix (with a .toarray() method)
          of shape (B, num_classes), where B is the batch size.
        - y_pred is a numpy array of shape (B, num_classes) containing probability
          distributions along the specified axis.

    Parameters
    ----------
    y_true : numpy.ndarray or sparse matrix
        Ground truth labels. If not a numpy array, it is converted using .toarray().
    y_pred : numpy.ndarray
        Predicted probability distributions.
    axis : int, optional
        The axis along which the classes are defined (default is 1).

    Returns
    -------
    loss : numpy.ndarray
        The average sparse categorical cross entropy loss.
    grad : numpy.ndarray
        The gradient of the loss with respect to y_pred.

    Notes
    -----
    A small constant EPSILON is added to y_pred before taking the logarithm to avoid log(0).
    """
    # Convert sparse matrix to dense if needed.
    if isinstance(y_true, np.ndarray):
        y_dense = y_true
    else:
        try:
            y_dense = y_true.toarray()
        except AttributeError:
            raise ValueError("y_true must be a numpy array or have a 'toarray' method.")

    B = y_pred.shape[0]
    loss = -np.mean(np.sum(y_dense * np.log(y_pred + EPSILON), axis=axis), axis=0)
    grad = (y_pred - y_dense) / B
    return loss, grad


def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error (MSE) loss and its gradient.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground truth values.
    y_pred : numpy.ndarray
        Predicted values.

    Returns
    -------
    loss : numpy.ndarray
        The average mean squared error loss.
    grad : numpy.ndarray
        The gradient of the MSE loss with respect to y_pred.

    Notes
    -----
    The gradient here is computed as 2*(y_pred - y_true)/B, where B is the batch size.
    """
    B = y_pred.shape[0]
    loss = np.mean(np.square(y_true - y_pred))
    grad = 2 * (y_pred - y_true) / B
    return loss, grad


def get_loss_fn(loss_fn_name):
    """
    Retrieve the loss function based on the provided name.

    Parameters
    ----------
    loss_fn_name : str
        Name of the loss function. Options are:
            - 'bce' for binary cross entropy
            - 'categorical_bce' for sparse categorical cross entropy
            - 'mse' for mean squared error

    Returns
    -------
    function
        The corresponding loss function.

    Raises
    ------
    ValueError
        If the loss function name is not recognized.
    """
    if loss_fn_name == 'bce':
        return binary_cross_entropy
    elif loss_fn_name == 'categorical_bce':
        return sparse_categorical_cross_entropy
    elif loss_fn_name == 'mse':
        return mean_squared_error
    else:
        raise ValueError("Unknown loss function: " + loss_fn_name)
