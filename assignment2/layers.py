import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(np.square(W))
    grad = reg_strength * 2 * W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    predictions_copy = predictions.copy()

    if predictions.ndim == 1:
        predictions_copy -= np.max(predictions_copy)
        exp_pred = np.exp(predictions_copy)
        probs = exp_pred/np.sum(exp_pred)
    else:
        predictions_copy -= np.amax(predictions_copy, axis=1, keepdims=True)
        exp_pred = np.exp(predictions_copy)
        probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    if probs.ndim == 1:
        loss = -np.log(probs[target_index])
    else:
        batch_size = probs.shape[0]
        loss_per_batch = -np.log(probs[np.arange(batch_size), target_index.flatten()])
        loss = np.sum(loss_per_batch) / batch_size
    return loss


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    dprediction = softmax(preds)
    loss = cross_entropy_loss(dprediction, target_index)

    if preds.ndim == 1:
        dprediction[target_index] -= 1
    else:
        batch_size = preds.shape[0]
        dprediction[np.arange(batch_size), target_index.flatten()] -= 1
        dprediction /= batch_size
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self._X_ = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self._X_ = X
        return np.maximum(X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        return d_out * (self._X_ > 0).astype(int)

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        self.X = X
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute
        d_result = np.dot(d_out, self.W.value.T)
        dW = np.dot(self.X.T, d_out)
        dB = np.dot(np.ones((1, d_out.shape[0])), d_out)

        self.W.grad += dW
        self.B.grad += dB
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
