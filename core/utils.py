"""
This module provides a collection of utility functions that is needed for future phases of COCO.

It includes:
- NN Layers Grad Check.
- More to be implemented functions in the future.
"""
import numpy as np
from core import nn
from core import optim
import pickle
def gradient_check(model, X, Y, loss_fn, epsilon=1e-7, threshold=1e-5):
    """
    Performs gradient checking for a given model.
    """
    # Forward pass to compute initial loss
    model(X)
    loss, grad_output = loss_fn(Y, model.last_out, axis=1)

    # Backward pass to get analytical gradients (without optimizer step)
    error_grad = grad_output
    for _, layer in reversed(list(model.layers.items())):
        error_grad = layer.backward(error_grad)

    for _,layer in list(model.layers.items()):
        if not hasattr(layer, 'weights') or layer.weights is None:
            continue

        W, grad_W = layer.weights, layer.grad_w

        # ------ Numerical Gradient for Weights (element-wise) ------
        numerical_grad_W = np.zeros_like(W)
        original_weights = W.copy()

        for idx in np.ndindex(W.shape):
            # Perturb weight at idx by +epsilon
            W[idx] += epsilon
            model(X)
            loss_plus = loss_fn(Y, model.last_out, axis=1)[0]

            # Perturb weight at idx by -epsilon
            W[idx] = original_weights[idx] - epsilon
            model(X)
            loss_minus = loss_fn(Y, model.last_out, axis=1)[0]

            # Compute numerical gradient for this weight
            numerical_grad_W[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            # Restore original weight
            W[idx] = original_weights[idx]

        # Compare analytical and numerical gradients
        diff_W = np.linalg.norm(grad_W - numerical_grad_W) / (np.linalg.norm(grad_W) + np.linalg.norm(numerical_grad_W) + 1e-15)
        print(f"Layer {layer.__class__.__name__} (Weights): Gradient Check {'PASSED' if diff_W < threshold else 'FAILED'} (diff: {diff_W:.8e})")

        # ------ Numerical Gradient for Biases (if present) ------
        if hasattr(layer, 'bias') and layer.bias is not None:
            b, grad_b = layer.bias, layer.grad_b
            numerical_grad_b = np.zeros_like(b)
            original_bias = b.copy()

            for idx in np.ndindex(b.shape):
                b[idx] += epsilon
                model(X)
                loss_plus = loss_fn(Y, model.last_out, axis=1)[0]

                b[idx] = original_bias[idx] - epsilon
                model(X)
                loss_minus = loss_fn(Y, model.last_out, axis=1)[0]

                numerical_grad_b[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                b[idx] = original_bias[idx]

            diff_b = np.linalg.norm(grad_b - numerical_grad_b) / (np.linalg.norm(grad_b) + np.linalg.norm(numerical_grad_b) + 1e-15)
            print(f"Layer {layer.__class__.__name__} (Biases): Gradient Check {'PASSED' if diff_b < threshold else 'FAILED'} (diff: {diff_b:.8e})\n")

def save_model(model, filename="model.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved as {filename}")

def load_model(filename="model.pkl"):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    return model