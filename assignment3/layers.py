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


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    dprediction = softmax(predictions)
    loss = cross_entropy_loss(dprediction, target_index)

    if predictions.ndim == 1:
        dprediction[target_index] -= 1
    else:
        batch_size = predictions.shape[0]
        dprediction[np.arange(batch_size), target_index.flatten()] -= 1
        dprediction /= batch_size
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self._X_ = None

    def forward(self, X):
        self._X_ = X
        return np.maximum(X, 0)

    def backward(self, d_out):
        return d_out * (self._X_ > 0).astype(int)

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        d_result = np.dot(d_out, self.W.value.T)
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.dot(np.ones((1, d_out.shape[0])), d_out)

        return d_result

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''
        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(np.random.randn(filter_size, filter_size,
                                       in_channels, out_channels))
        self.B = Param(np.zeros(out_channels))
        self.padding = padding

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        X_padding = np.zeros(shape=(batch_size, height + 2*self.padding,
                                    width + 2*self.padding, channels))
        X_padding[:, self.padding: height+self.padding, self.padding: width+self.padding, :] = X
        self.X = X_padding

        out_height = X_padding.shape[1] - self.filter_size + 1
        out_width = X_padding.shape[2] - self.filter_size + 1
        layer_out = np.zeros(shape=(batch_size, out_height, out_width, self.out_channels))

        # TODO: Implement forward pass
        for x in range(out_height):
            for y in range(out_width):
                receptive_filed = X_padding[:, x:x+self.filter_size, y:y+self.filter_size, :]

                layer_out[:, x:x+self.filter_size, y:y+self.filter_size, :] = (receptive_filed.reshape(
                    (batch_size, -1)).dot(self.W.value.reshape((-1, self.out_channels))) +
                    self.B.value).reshape(batch_size, 1, 1, self.out_channels)
        return layer_out

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        grad = np.zeros(shape=(batch_size, height, width, channels))

        for x in range(out_height):
            for y in range(out_width):
                # TODO: Implement backward pass for specific location
                d_local = d_out[:, x:x+1, y:y+1, :]
                receptive_filed = self.X[:, x:x+self.filter_size, y:y+self.filter_size, :]
                grad[:, x:x+self.filter_size, y:y+self.filter_size, :] += \
                    (d_local.reshape(batch_size, -1).dot(
                        self.W.value.reshape(-1, self.out_channels).T)).reshape(
                        receptive_filed.shape)
                dW = receptive_filed.reshape(batch_size, -1).T.dot(
                    d_local.reshape(batch_size, -1))
                dB = np.dot(np.ones((1, d_local.shape[0])), d_local.reshape(batch_size, -1))

                self.W.grad += dW.reshape(self.W.value.shape)
                self.B.grad += dB.reshape(self.B.value.shape)

        return grad[:, self.padding:(height-self.padding), self.padding:(width-self.padding), :]

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.masks = {}

    def mask(self, x, loc):
        zero_mask = np.zeros_like(x)
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)
        idx = np.argmax(x, axis=1)

        b_idx, c_idx = np.indices((batch_size, channels))
        zero_mask.reshape(batch_size, height * width, channels)[b_idx, idx, c_idx] = 1
        self.masks[loc] = zero_mask

    # TODO: Implement maxpool forward pass (output x/y dimension)
    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        step = self.stride
        self.masks.clear()

        out_height = (X.shape[1] - self.pool_size)//self.stride + 1
        out_width = (X.shape[2] - self.pool_size)//self.stride + 1
        mplayer_out = np.zeros(shape=(batch_size, out_height, out_width, channels))

        for x in range(out_height):
            for y in range(out_width):
                comp_field = X[:, x*step: x*step + self.pool_size, y*step: y*step + self.pool_size, :]
                self.mask(x=comp_field, loc=(x, y))
                mplayer_out[:, x, y, :] = np.max(comp_field, axis=(1, 2))
        return mplayer_out

    # TODO: Implement MaxPool backward pass
    def backward(self, d_out):
        _, out_height, out_width, _ = d_out.shape
        grad = np.zeros_like(self.X)
        step = self.stride

        for x in range(out_height):
            for y in range(out_width):
                grad[:, x*step: x*step + self.pool_size, y*step: y*step + self.pool_size, :] += \
                    d_out[:, x:x+1, y:y+1, :] * self.masks[(x, y)]
        return grad

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement forward pass
        self.X_shape = X.shape
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        return {}
